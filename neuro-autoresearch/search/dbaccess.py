#!/usr/bin/env python3
"""Shared, agent-extensible DB-access library -- the proposer pool's ONLY query surface.

The open-ended search keeps **no notebook**: the SQLite ``models`` table (one row per model,
schema owned by ``search/db.py``) *is* the shared memory, and this module is how every
proposer reads it and learns from the rest of the pool. Three built-in helpers cover the
common cases --

    top_n(db, n)      ranked champions (best ``test_corr`` first), WITH source to build on
    random_n(db, n)   random explorers (idea + score; a spark, not a clone)
    by_id(db, id)     the FULL details + scores + feedback + source of one model, by its
                      five-char alphanumeric id (e.g. ``a3f9k``)

-- and the surface is **extensible**: any proposer can register a new shared query function
for the whole pool with ``register`` (it is appended to ``<namespace>/extensions.py`` and
becomes callable by every worker via ``call``). ``blueprint`` prints the canonical template
(``blueprint_fn``: open a connection, run a *parametrised* query, return rows) to start any
new function from. The freedom is deliberate -- the schema, not a frozen helper API, is the
real contract, so agents get maximum latitude to explore the DB however they want.

Every function takes the namespace DB path
(``search/state/<study>__<roi>/results.db``); the pool shares one DB per namespace under WAL,
so concurrent reads are safe. The proposer's single write is ``enqueue`` -- one ``status=
'queued'`` row per wave, stamped with the current ``iter`` and a ``path`` pointer to the
``model.py`` it just wrote to disk (the source itself never goes into the DB).

CLI (what a headless proposer calls)::

    python search/dbaccess.py schema                 # the full table schema (for the prompt)
    python search/dbaccess.py top_n    <db> [-n 5]
    python search/dbaccess.py random_n <db> [-n 5] [--seed S]
    python search/dbaccess.py by_id    <db> <id>
    python search/dbaccess.py blueprint              # the new-function template
    python search/dbaccess.py list     <db>          # registered shared extensions
    python search/dbaccess.py register <db> --name F --file f.py
    python search/dbaccess.py call     <db> <fn> [json-arg ...]
    python search/dbaccess.py newid    <db>          # a fresh, unused 5-char id
    python search/dbaccess.py enqueue  <db> --id .. --iter .. --idea .. \\
                                            --dataset .. --roi .. --path ..
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import random
import secrets
import sqlite3
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db as _db  # noqa: E402  (schema gateway: db.connect = WAL + busy_timeout; db.NUM = score cols)

# Columns a freshly proposed (queued) row carries; the NUM score columns stay NULL until the
# evaluator's write-back. Mirrors the chunk-01 schema contract without re-declaring it.
QUEUE_COLS = ["id", "iter", "idea", "dataset", "roi", "status", "path"]
ID_ALPHABET = string.ascii_lowercase + string.digits  # five-char alphanumeric model ids
ID_LEN = 5


# --------------------------------------------------------------------------- helpers
def _rows(con, sql, params=()):
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def _read_source(path):
    try:
        with open(path) as f:
            return f.read()
    except (OSError, TypeError):
        return ""


def _score_cols():
    return ", ".join(_db.NUM)


# --------------------------------------------------------------------------- built-in queries
def top_n(db_path, n=5, roi=None, with_code=True):
    """Ranked champions: the ``status='ok'`` rows with the highest reward, best first.

    ``test_corr`` already *is* the reward for the row's reward region, so ranking by it is
    correct within a namespace; pass ``roi`` to rank by that region's ``corr_<roi>`` instead.
    Champions come WITH their source (read from ``path``) so a proposer can build on them.
    """
    order = "test_corr"
    if roi:
        cand = f"corr_{roi}"
        if cand in _db.NUM:
            order = cand
    with _db.connect(db_path) as con:
        rows = _rows(
            con,
            f"SELECT id, iter, idea, status, path, {_score_cols()} FROM models "
            f"WHERE status='ok' AND {order} IS NOT NULL ORDER BY {order} DESC LIMIT ?",
            (int(n),),
        )
    if with_code:
        for r in rows:
            r["code"] = _read_source(r.get("path"))
    return rows


def random_n(db_path, n=5, seed=None):
    """Random explorers: a random draw of ``status='ok'`` rows (idea + score, no source) --
    a per-call diversity spark so the pool studies different prior work, not just the champion.
    """
    cols = f"id, iter, idea, status, {_score_cols()}"
    where = "WHERE status='ok' AND test_corr IS NOT NULL"
    with _db.connect(db_path) as con:
        if seed is None:  # let SQLite sample so we don't pull the whole table just to slice n
            return _rows(con, f"SELECT {cols} FROM models {where} ORDER BY RANDOM() LIMIT ?",
                         (int(n),))
        pool = _rows(con, f"SELECT {cols} FROM models {where}")  # seeded path needs the full set
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[: int(n)]


def by_id(db_path, model_id):
    """Full details + scores + feedback + source of one model, by its five-char id.

    Lets an agent re-read the feedback on its own last model, or weigh in on any other model
    in the DB. Returns the entire row (all columns incl. status/scores) plus the on-disk
    source, or ``None`` if the id is unknown.
    """
    with _db.connect(db_path) as con:
        rows = _rows(con, "SELECT * FROM models WHERE id=?", (model_id,))
    if not rows:
        return None
    row = rows[0]
    row["code"] = _read_source(row.get("path"))
    return row


# --------------------------------------------------------------------------- the blueprint
def blueprint_fn(con, status="ok", limit=5):
    """TEMPLATE for a new shared query function -- the canonical starting point.

    Copy this, rename it, and change the SQL to whatever slice of the DB you want. The rules:
    receive an OPEN connection (from ``db.connect``); run ONE *parametrised* query (pass
    values as ``?`` params -- never string-format them into the SQL); return a list of plain
    ``dict`` rows. Register it with ``register`` and the whole pool can ``call`` it.
    """
    cur = con.execute(
        "SELECT id, iter, idea, status, test_corr FROM models "
        "WHERE status = ? ORDER BY test_corr DESC LIMIT ?",
        (status, int(limit)),
    )
    return [dict(r) for r in cur.fetchall()]


BLUEPRINT_SRC = inspect.getsource(blueprint_fn)


# --------------------------------------------------------------------------- extension mechanism
def _ext_file(db_path):
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "extensions.py")


def load_extensions(db_path):
    """Load the namespace's shared extension functions -> {name: fn}.

    Extensions live in ``<namespace>/extensions.py`` (a plain module any proposer can append
    to); each is a ``fn(con, ...) -> rows`` like ``blueprint_fn``. Returns ``{}`` if there is
    no extension file yet or it fails to import (a broken extension never breaks the pool).
    """
    path = _ext_file(db_path)
    if not os.path.exists(path):
        return {}
    try:
        spec = importlib.util.spec_from_file_location("dbaccess_ext", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001  -- a malformed extension must not crash a proposer
        print(f"[dbaccess] warning: could not load extensions.py: {e}", file=sys.stderr)
        return {}
    return {n: f for n, f in vars(mod).items() if callable(f) and not n.startswith("_")}


def register(db_path, name, src):
    """Append a new shared query function to the namespace's ``extensions.py`` (pool-wide).

    ``src`` is the function's full Python source (must define ``def <name>(con, ...)``). The
    file is created with a header on first use. After this, every worker can ``call`` it.
    """
    if f"def {name}" not in src:
        raise ValueError(f"source must define `def {name}(con, ...)`")
    path = _ext_file(db_path)
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        if new:
            f.write(
                "# Shared, agent-extensible DB-access functions for this search namespace.\n"
                "# Each fn(con, ...) -> list[dict]; registered via `dbaccess.py register`,\n"
                "# called pool-wide via `dbaccess.py call`. Start from `dbaccess.blueprint_fn`.\n"
            )
        f.write("\n\n" + src.rstrip() + "\n")
    return path


def call(db_path, fn_name, args):
    """Run a registered extension (or a built-in ``fn(con, ...)``) and return its rows."""
    registry = {"blueprint_fn": blueprint_fn, **load_extensions(db_path)}
    fn = registry.get(fn_name)
    if fn is None:
        raise KeyError(f"no such function {fn_name!r} (have: {sorted(registry)})")
    with _db.connect(db_path) as con:
        return fn(con, *args)


# --------------------------------------------------------------------------- the proposer's write
def newid(db_path):
    """A fresh five-char alphanumeric id not already in the DB (best-effort; races are caught by
    ``enqueue``'s plain INSERT). Probes one minted id at a time rather than scanning every row."""
    have_db = os.path.exists(db_path)
    for _ in range(10000):
        mid = "".join(secrets.choice(ID_ALPHABET) for _ in range(ID_LEN))
        if not have_db:
            return mid
        with _db.connect(db_path) as con:
            try:
                hit = con.execute("SELECT 1 FROM models WHERE id=? LIMIT 1", (mid,)).fetchone()
            except sqlite3.OperationalError:  # table not created yet -> any id is free
                return mid
        if hit is None:
            return mid
    raise RuntimeError("could not mint a fresh id")


def enqueue(db_path, model_id, it, idea, dataset, roi, path):
    """The proposer's single write: insert ONE ``status='queued'`` row stamped with the wave's
    ``iter`` and a ``path`` pointer to the model file on disk (never the source itself).

    A plain INSERT (not INSERT OR REPLACE) so a colliding id is rejected rather than silently
    clobbering another worker's model -- the proposer mints a new id and retries. This is the
    only write proposers do, and it is what enforces *one model per wave per worker*.
    """
    _db.init(db_path)  # idempotent: ensure the namespace db + table exist
    row = {
        "id": model_id,
        "iter": int(it),
        "idea": (idea or "")[:300],
        "dataset": dataset or "nsd",
        "roi": roi or "ventral",
        "status": "queued",
        "path": os.path.abspath(path),
    }
    cols = ",".join(QUEUE_COLS)
    ph = ",".join("?" * len(QUEUE_COLS))
    with _db.connect(db_path) as con:
        con.execute(f"INSERT INTO models ({cols}) VALUES ({ph})", [row[c] for c in QUEUE_COLS])
    return row


# --------------------------------------------------------------------------- schema (for prompts)
def schema_doc():
    """The full ``models`` schema, embedded verbatim in every proposer's prompt so agents know
    exactly which columns they may query."""
    cols = []
    for c in _db.COLS:
        if c == "iter":
            t = "INTEGER"
        elif c in _db.NUM:
            t = "REAL"
        else:
            t = "TEXT"
        cols.append(f"    {c:<14} {t}")
    return (
        "TABLE models -- one row per model; it is BOTH the full history AND the live queue.\n"
        + "\n".join(cols)
        + "\n\nstatus lifecycle:  queued -> running -> ok | error\n"
        "  queued   a proposer wrote the row; its model.py is on disk at `path`, awaiting eval\n"
        "  running  the evaluator has claimed it (one model on the GPU at a time)\n"
        "  ok       scored cleanly; has a `test_corr` (the reward) + per-region `corr_*`\n"
        "  error    raised / bad shape / timeout, even after repair\n"
        "`path` points to the model.py on disk (the source is NEVER inlined in the DB).\n"
        "`iter` is the wave number; `test_corr` is the held-out reward for the run's region."
    )


# --------------------------------------------------------------------------- rendering (agent-readable)
def _fmt(x):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "?"


def _render_row(r, with_code=False):
    head = f"[{r.get('id')}] test_corr={_fmt(r.get('test_corr'))} iter={r.get('iter')} " \
           f"status={r.get('status')} -- {(r.get('idea') or '').strip()}"
    if with_code and r.get("code"):
        return f"{head}\n```python\n{r['code'].strip()}\n```"
    return head


def _print_rows(rows, with_code, as_json):
    if as_json:
        print(json.dumps(rows, indent=2, default=str))
    elif not rows:
        print("(no matching rows yet)")
    else:
        print("\n\n".join(_render_row(r, with_code) for r in rows) if with_code
              else "\n".join(_render_row(r) for r in rows))


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description="Shared DB-access for the proposer pool.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _db_arg(sp):
        sp.add_argument("db", help="namespace results.db path")

    sp = sub.add_parser("top_n"); _db_arg(sp); sp.add_argument("-n", type=int, default=5)
    sp.add_argument("--roi", default=None); sp.add_argument("--json", action="store_true")
    sp = sub.add_parser("random_n"); _db_arg(sp); sp.add_argument("-n", type=int, default=5)
    sp.add_argument("--seed", type=int, default=None); sp.add_argument("--json", action="store_true")
    sp = sub.add_parser("by_id"); _db_arg(sp); sp.add_argument("id")
    sp.add_argument("--json", action="store_true")
    sub.add_parser("blueprint")
    sub.add_parser("schema")
    sp = sub.add_parser("list"); _db_arg(sp)
    sp = sub.add_parser("register"); _db_arg(sp)
    sp.add_argument("--name", required=True); sp.add_argument("--file", required=True)
    sp = sub.add_parser("call"); _db_arg(sp); sp.add_argument("fn")
    sp.add_argument("args", nargs="*", help="JSON-encoded positional args")
    sp = sub.add_parser("newid"); _db_arg(sp)
    sp = sub.add_parser("enqueue"); _db_arg(sp)
    for f in ("id", "iter", "idea", "dataset", "roi", "path"):
        sp.add_argument(f"--{f}", required=True)

    a = p.parse_args(argv)

    if a.cmd == "schema":
        print(schema_doc()); return
    if a.cmd == "blueprint":
        print("# Canonical template for a new shared DB-access function. Copy, rename, edit "
              "the SQL,\n# then: python search/dbaccess.py register <db> --name <fn> --file "
              "<your.py>\n\n" + BLUEPRINT_SRC); return
    if a.cmd == "top_n":
        _print_rows(top_n(a.db, a.n, a.roi), with_code=True, as_json=a.json); return
    if a.cmd == "random_n":
        _print_rows(random_n(a.db, a.n, a.seed), with_code=False, as_json=a.json); return
    if a.cmd == "by_id":
        r = by_id(a.db, a.id)
        if r is None:
            print(f"(no model {a.id!r} in this DB)")
        elif a.json:
            print(json.dumps(r, indent=2, default=str))
        else:
            print(_render_row(r, with_code=True))
        return
    if a.cmd == "list":
        ext = load_extensions(a.db)
        if not ext:
            print("(no registered extensions yet -- see `dbaccess.py blueprint`)")
        for n, f in sorted(ext.items()):
            doc = (inspect.getdoc(f) or "").splitlines()
            print(f"{n}{inspect.signature(f)} -- {doc[0] if doc else ''}")
        return
    if a.cmd == "register":
        with open(a.file) as f:
            src = f.read()
        print(f"[dbaccess] registered {a.name!r} -> {register(a.db, a.name, src)}"); return
    if a.cmd == "call":
        args = [json.loads(x) for x in a.args]
        print(json.dumps(call(a.db, a.fn, args), indent=2, default=str)); return
    if a.cmd == "newid":
        print(newid(a.db)); return
    if a.cmd == "enqueue":
        try:
            row = enqueue(a.db, a.id, a.iter, a.idea, a.dataset, a.roi, a.path)
        except sqlite3.IntegrityError:
            print(f"[dbaccess] id {a.id!r} already exists -- mint a new one (newid) and retry",
                  file=sys.stderr)
            sys.exit(3)
        print(f"[dbaccess] queued {row['id']} iter={row['iter']} -> {row['path']}"); return


if __name__ == "__main__":
    main()
