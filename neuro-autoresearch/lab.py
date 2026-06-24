#!/usr/bin/env python3
"""``lab`` -- the ONE API every proposer and the evaluator use to talk to a search run.

A *run* is a single open-ended search, isolated in its own directory::

    runs/<dataset>__<date>/
        search.db          # the shared brain: queue + full history + scores (this module owns it)
        models/<id>.py     # shared, IMMUTABLE archive -- one file per model ever tried
        proposers/pNN/     # one sandbox per proposer (its own mutable model.py + this lab symlink)
        eval/              # the single GPU evaluator's workspace

There is NO conductor, NO notebook, and NO wave barrier. ``search.db`` (one ``models`` row per
model) is the only shared memory and doubles as the queue via a ``status`` flag::

    queued   a proposer wrote the row; its model.py is archived at ``path``, awaiting evaluation
      -> running   the single GPU evaluator claimed it (one model at a time)
        -> ok       scored cleanly; has a ``test_corr`` (the reward)
        -> error    raised / bad shape / timeout (even after repair)

**Self-paced, not synchronised.** A proposer pushes ONE model, blocks on *its own* result
(``wait_for_result``), then immediately reads the current DB and proposes again -- no waiting for
siblings. That alone caps the queue at ~one in-flight per proposer and gives each agent the
fastest feedback on whether its last idea worked.

The proposer half is pure sqlite (no torch / no harness import), so ``import lab`` is light.
Scoring (``score`` / the evaluator path) lazy-imports the FIXED harness and runs the model in a
subprocess for isolation.

Which run this process targets is the dir in ``$LAB_RUN`` (else: walk up from cwd to the nearest
``search.db``). ``$LAB_PROPOSER`` (e.g. ``p03``) tags rows with who wrote them.

CLI (what a headless agent calls without writing Python)::

    python lab.py init
    python lab.py propose <model_file> --idea "..."     # -> prints the new id
    python lab.py wait <id>                              # block; prints the scored row as JSON
    python lab.py champions [-n 5]                       # ranked best, WITH source
    python lab.py sample    [-n 5] [--seed S]            # random prior models
    python lab.py get       <id>                         # one model: row + source
    python lab.py query     "SELECT ..."                # read-only arbitrary SQL -> JSON
    python lab.py newid
    python lab.py next                                   # evaluator: claim next queued -> JSON
    python lab.py record    <id> --status ok --test-corr .. [--train-corr ..] [...]
    python lab.py requeue                                # evaluator startup crash recovery
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sqlite3
import string
import sys
import time
from datetime import datetime, timezone

# --------------------------------------------------------------------------- schema
# One row per model; it is BOTH the full history AND the live queue. No ROI, no per-region
# corr_* columns, no wave/iter -- the reward is a single dataset-level ``test_corr``.
NUM = ("test_corr", "train_corr", "gap", "n_params", "seconds")
COLS = ("id", "idea", "summary", "proposer", "dataset", "status",
        "test_corr", "train_corr", "gap", "n_params", "seconds", "created_at", "path")
STATUSES = ("queued", "running", "ok", "error")
_ID_ALPHABET = string.ascii_lowercase + string.digits
_ID_LEN = 5
_POLL = float(os.environ.get("LAB_POLL", "5"))  # seconds between wait/queue polls


# --------------------------------------------------------------------------- run location
def find_run(start: str | None = None) -> str:
    """The run directory this process targets: ``$LAB_RUN`` if set, else the nearest ancestor
    of ``start`` (default cwd) that contains a ``search.db``. Raises if neither resolves."""
    env = os.environ.get("LAB_RUN")
    if env:
        return os.path.abspath(env)
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.exists(os.path.join(d, "search.db")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "no run found: set $LAB_RUN to a runs/<dataset>__<date>/ dir, or run inside one")
        d = parent


def db_path(run: str | None = None) -> str:
    return os.path.join(run or find_run(), "search.db")


def models_dir(run: str | None = None) -> str:
    return os.path.join(run or find_run(), "models")


# --------------------------------------------------------------------------- connection / schema
def connect(run: str | None = None) -> sqlite3.Connection:
    """A WAL-mode connection with a generous busy timeout, so the 10 proposers' inserts and the
    evaluator's updates coexist without ``database is locked`` errors."""
    con = sqlite3.connect(db_path(run), timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.row_factory = sqlite3.Row
    return con


def _coltype(c: str) -> str:
    if c == "n_params":
        return "INTEGER"
    return "REAL" if c in NUM else "TEXT"


def init(run: str | None = None) -> str:
    """Create the run dir + ``models/`` + the ``models`` table (idempotent). Returns the run dir."""
    run = os.path.abspath(run or find_run())
    os.makedirs(models_dir(run), exist_ok=True)
    defs = ", ".join(("id TEXT PRIMARY KEY" if c == "id" else f"{c} {_coltype(c)}") for c in COLS)
    with connect(run) as con:
        con.execute(f"CREATE TABLE IF NOT EXISTS models ({defs})")
    return run


def _ensure(run: str | None = None) -> str:
    run = os.path.abspath(run or find_run())
    if not os.path.exists(db_path(run)):
        init(run)
    return run


# --------------------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None  # drop NaN/inf


def _rows(con, sql, params=()):
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def _read_source(path):
    try:
        with open(path) as f:
            return f.read()
    except (OSError, TypeError):
        return ""


def newid(run: str | None = None) -> str:
    """A fresh 5-char alphanumeric id not already in the DB (races are caught by propose's INSERT)."""
    run = _ensure(run)
    with connect(run) as con:
        for _ in range(10000):
            mid = "".join(secrets.choice(_ID_ALPHABET) for _ in range(_ID_LEN))
            if con.execute("SELECT 1 FROM models WHERE id=? LIMIT 1", (mid,)).fetchone() is None:
                return mid
    raise RuntimeError("could not mint a fresh id")


# --------------------------------------------------------------------------- proposer: write + sync
def propose(model_file: str, idea: str = "", summary: str = "",
            proposer: str | None = None, run: str | None = None) -> str:
    """Register a new model: snapshot ``model_file`` into ``models/<id>.py`` (immutable) and insert
    one ``queued`` row. Returns the new id. This is a proposer's only DB write.

    The snapshot is what the evaluator scores, so the proposer is free to keep editing its own
    working ``model.py`` for the next attempt the instant this returns.
    """
    run = _ensure(run)
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    proposer = proposer or os.environ.get("LAB_PROPOSER", "")
    dataset = os.environ.get("NEURO_DATASET", "")
    for _ in range(50):                       # retry on the rare id collision
        mid = newid(run)
        dst = os.path.join(models_dir(run), f"{mid}.py")
        shutil.copyfile(model_file, dst)
        row = {
            "id": mid, "idea": (idea or "")[:500], "summary": (summary or "")[:1000],
            "proposer": proposer, "dataset": dataset, "status": "queued",
            "test_corr": None, "train_corr": None, "gap": None, "n_params": None,
            "seconds": None, "created_at": _now(), "path": os.path.abspath(dst),
        }
        try:
            with connect(run) as con:
                con.execute(f"INSERT INTO models ({','.join(COLS)}) "
                            f"VALUES ({','.join('?' * len(COLS))})", [row[c] for c in COLS])
            return mid
        except sqlite3.IntegrityError:
            os.path.exists(dst) and os.remove(dst)
            continue
    raise RuntimeError("could not insert a queued row (repeated id collisions)")


def get_result(mid: str, run: str | None = None) -> dict | None:
    """The row for ``mid``, or ``None`` if unknown. Use to check status without blocking."""
    with connect(_ensure(run)) as con:
        r = con.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    return dict(r) if r else None


def wait_for_result(mid: str, poll: float = _POLL, timeout: float | None = None,
                    run: str | None = None) -> dict:
    """BLOCK until model ``mid`` is scored (status ``ok``/``error``); return its row.

    This is the proposer's self-pacing primitive: it only waits for ITS OWN model, never for the
    pool. Raises ``KeyError`` if the id is unknown, ``TimeoutError`` if ``timeout`` elapses first.
    """
    run = _ensure(run)
    deadline = None if timeout is None else time.time() + timeout
    while True:
        row = get_result(mid, run)
        if row is None:
            raise KeyError(f"no such model {mid!r}")
        if row["status"] in ("ok", "error"):
            return row
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"{mid} not scored within {timeout}s (status={row['status']})")
        time.sleep(poll)


# --------------------------------------------------------------------------- proposer: read / learn
def champions(k: int = 5, with_code: bool = True, run: str | None = None) -> list[dict]:
    """The ``ok`` models with the highest reward, best first -- WITH source so you can build on them."""
    with connect(_ensure(run)) as con:
        rows = _rows(con, "SELECT * FROM models WHERE status='ok' AND test_corr IS NOT NULL "
                          "ORDER BY test_corr DESC LIMIT ?", (int(k),))
    if with_code:
        for r in rows:
            r["code"] = _read_source(r.get("path"))
    return rows


def sample(k: int = 5, seed=None, run: str | None = None) -> list[dict]:
    """A random draw of ``ok`` models (idea + score, no source) -- a diversity spark so the pool
    studies different prior work, not just the single champion."""
    with connect(_ensure(run)) as con:
        rows = _rows(con, "SELECT id, idea, proposer, status, test_corr, train_corr, gap "
                          "FROM models WHERE status='ok' AND test_corr IS NOT NULL "
                          "ORDER BY RANDOM() LIMIT ?", (int(k),))
    return rows


def get_model(mid: str, run: str | None = None) -> dict | None:
    """One model's full row + on-disk source, by id, or ``None`` if unknown."""
    with connect(_ensure(run)) as con:
        r = con.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if r is None:
        return None
    row = dict(r)
    row["code"] = _read_source(row.get("path"))
    return row


def query(sql: str, params=(), run: str | None = None) -> list[dict]:
    """Read-only arbitrary SQL over ``search.db`` -- the escape hatch for any custom view an agent
    wants (it can build its own helpers in its sandbox on top of this). Rejects non-SELECT SQL."""
    if sql.lstrip().split(None, 1)[0].lower() not in ("select", "with"):
        raise ValueError("query() is read-only: use a SELECT/WITH statement")
    with connect(_ensure(run)) as con:
        return _rows(con, sql, params)


# --------------------------------------------------------------------------- evaluator: claim / record
def get_next_model(run: str | None = None) -> dict | None:
    """Atomically claim the oldest queued model (``queued -> running``) in ONE statement; return
    ``{id, path, dataset, ...}`` or ``None`` when the queue is empty. One active evaluator only."""
    with connect(_ensure(run)) as con:
        r = con.execute(
            "UPDATE models SET status='running' "
            "WHERE id=(SELECT id FROM models WHERE status='queued' ORDER BY created_at, rowid LIMIT 1) "
            "RETURNING *").fetchone()
    return dict(r) if r else None


def record(mid: str, scores: dict | None = None, status: str = "ok",
           run: str | None = None) -> str:
    """Write a model's result back: ``running -> ok|error``, filling the numeric score columns
    present in ``scores``. ``id``/``idea``/``path``/... are untouched."""
    st = "ok" if str(status).strip().lower().startswith("ok") else "error"
    scores = scores or {}
    cols = [c for c in NUM if scores.get(c) not in (None, "")]
    assignments = [f"{c}=?" for c in cols] + ["status=?"]
    vals = [_num(scores[c]) for c in cols] + [st, mid]
    with connect(_ensure(run)) as con:
        con.execute(f"UPDATE models SET {', '.join(assignments)} WHERE id=?", vals)
    return st


def requeue_orphans(run: str | None = None) -> int:
    """Startup crash recovery: any ``running`` row (the evaluator died mid-score) goes back to
    ``queued`` so it is re-claimed. Returns the number recovered."""
    if not os.path.exists(db_path(os.path.abspath(run or find_run()))):
        return 0
    with connect(run) as con:
        return con.execute("UPDATE models SET status='queued' WHERE status='running'").rowcount


# --------------------------------------------------------------------------- launcher: DB summary
def summary(run: str | None = None) -> str:
    """One-line DB tally + current champion, for the launcher's startup/shutdown log."""
    with connect(run) as con:
        tally = dict(con.execute(
            "SELECT COALESCE(status,'?'), COUNT(*) FROM models GROUP BY 1").fetchall())
        best = con.execute(
            "SELECT id, test_corr FROM models WHERE status='ok' AND test_corr IS NOT NULL "
            "ORDER BY test_corr DESC LIMIT 1").fetchone()
    t = " ".join(f"{k}={v}" for k, v in sorted(tally.items())) or "(empty)"
    champ = f"{best['id']} @ {best['test_corr']:.4f}" if best else "(none)"
    return f"DB: {t} | champion: {champ}"


# --------------------------------------------------------------------------- evaluator: scoring (lazy harness)
def score(model_file: str, dataset: str | None = None, timeout: int = 420) -> dict:
    """Score one model file with the FIXED harness, in a subprocess (isolation against CUDA
    faults). Returns ``{status, test_corr, train_corr, gap, n_params, seconds}``. Never raises on
    a model fault -- a crash/timeout becomes ``status='error'``.

    Reward is the dataset's fixed-region ``test_corr`` (NSD: the ventral stream); there is no
    operator ROI knob.
    """
    import subprocess
    import tempfile
    repo = os.path.dirname(os.path.abspath(__file__))
    dataset = dataset or os.environ.get("NEURO_DATASET", "nsd")
    env = {**os.environ, "NEURO_DATASET": dataset}
    env.pop("NEURO_ROI", None)                # no ROI knob: reward = the preset's fixed region (NSD: ventral)
    # A throwaway out-file the child writes the JSON result to.
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        out = tf.name
    env["LAB_SCORE_OUT"] = out
    try:
        proc = subprocess.run([sys.executable, "-m", "lab", "_score_child", os.path.abspath(model_file)],
                              cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
        log = (proc.stdout or "") + (proc.stderr or "")
        if os.path.getsize(out):
            with open(out) as f:
                res = json.load(f)
        else:
            res = {"status": "error: no result (import/crash)"}
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") + (e.stderr or "") + f"\nTIMEOUT after {timeout}s\n"
        res = {"status": f"error: timeout after {timeout}s"}
    finally:
        os.path.exists(out) and os.remove(out)
    res.setdefault("log", log[-4000:])        # tail of stdout+stderr (the traceback, for repair)
    return res


def _score_child(model_file: str) -> None:
    """Child entrypoint (``python -m lab _score_child <file>``): import the harness, score the
    model on the active dataset (whole-dataset reward), write JSON to ``$LAB_SCORE_OUT``."""
    import importlib.util
    out = os.environ["LAB_SCORE_OUT"]
    res = {"status": "error: unknown"}
    try:
        from harness.datasets import DATASET
        from harness.loaders import load_split
        from harness.reward import evaluate_on_split_safe
        spec = importlib.util.spec_from_file_location("model", model_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        row = evaluate_on_split_safe(mod.MODEL, load_split(DATASET), DATASET, ())
        res = {k: row.get(k) for k in ("status", "test_corr", "train_corr", "gap",
                                       "n_params", "seconds")}
    except Exception as e:  # noqa: BLE001  -- any import/eval fault is a recorded error, not a crash
        import traceback
        traceback.print_exc()
        res = {"status": f"error: {type(e).__name__}: {e}"[:300]}
    with open(out, "w") as f:
        json.dump(res, f)


# --------------------------------------------------------------------------- CLI
def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(description="lab -- the search-run API (see module docstring).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sp = sub.add_parser("propose"); sp.add_argument("model_file")
    sp.add_argument("--idea", default=""); sp.add_argument("--summary", default="")
    sp.add_argument("--proposer", default=None)
    sp = sub.add_parser("wait"); sp.add_argument("id"); sp.add_argument("--timeout", type=float, default=None)
    sp = sub.add_parser("champions"); sp.add_argument("-n", type=int, default=5)
    sp = sub.add_parser("sample"); sp.add_argument("-n", type=int, default=5); sp.add_argument("--seed", default=None)
    sp = sub.add_parser("get"); sp.add_argument("id")
    sp = sub.add_parser("query"); sp.add_argument("sql")
    sub.add_parser("newid")
    sub.add_parser("next")
    sub.add_parser("requeue")
    sub.add_parser("summary")
    sp = sub.add_parser("record"); sp.add_argument("id"); sp.add_argument("--status", default="ok")
    for f in NUM:
        sp.add_argument(f"--{f.replace('_', '-')}", default=None)
    sp = sub.add_parser("_score_child"); sp.add_argument("model_file")  # internal
    a = p.parse_args(argv)

    if a.cmd == "init":
        print(init())
    elif a.cmd == "propose":
        print(propose(a.model_file, a.idea, a.summary, a.proposer))
    elif a.cmd == "wait":
        _print(wait_for_result(a.id, timeout=a.timeout))
    elif a.cmd == "champions":
        _print(champions(a.n))
    elif a.cmd == "sample":
        _print(sample(a.n, a.seed))
    elif a.cmd == "get":
        _print(get_model(a.id))
    elif a.cmd == "query":
        _print(query(a.sql))
    elif a.cmd == "newid":
        print(newid())
    elif a.cmd == "next":
        _print(get_next_model())
    elif a.cmd == "requeue":
        print(requeue_orphans())
    elif a.cmd == "summary":
        print(summary())
    elif a.cmd == "record":
        scores = {f: getattr(a, f) for f in NUM if getattr(a, f) is not None}
        print(record(a.id, scores, a.status))
    elif a.cmd == "_score_child":
        _score_child(a.model_file)


if __name__ == "__main__":
    main()
