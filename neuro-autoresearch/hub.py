#!/usr/bin/env python3
"""``hub`` -- the ONE API every proposer and the evaluator use to talk to a search run.

A *run* is a single open-ended search, isolated in its own directory::

    runs/<dataset>__<date>/
        search.db          # the shared brain: queue + full history + scores (this module owns it)
        models/<id>.py     # shared, IMMUTABLE archive -- one file per model ever tried
        proposers/pNN/     # one sandbox per proposer (its own mutable model.py + this hub symlink)
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

The proposer half is pure sqlite (no torch / no harness import), so ``import hub`` is light.
Scoring (``score`` / the evaluator path) lazy-imports the FIXED harness and runs the model in a
subprocess for isolation.

Which run this process targets is the dir in ``$LAB_RUN`` (else: walk up from cwd to the nearest
``search.db``). ``$LAB_PROPOSER`` (e.g. ``p03``) tags rows with who wrote them.

CLI (what a headless agent calls without writing Python)::

    python hub.py propose-model <model_file> --idea "..."  # -> prints the new id
    python hub.py wait-for-score <id>                       # block; prints the scored row as JSON
    python hub.py top-models [-n 5]                         # CONDUCTOR/launcher only -- global scoreboard (blocked for proposers)
    python hub.py random-models [-n 5] [--seed S]           # random prior models (scores + ideas)
    python hub.py my-best [--no-code]                       # YOUR island's best model (+source, +your own score) to build on
    python hub.py peek-island                               # best of ONE random OTHER island: idea + experiment_notes, NO source, NO score
    python hub.py query-db  "SELECT ..."                   # read-only arbitrary SQL -> JSON (score/source columns hidden from proposers)
    python hub.py reset-island <pNN> [--role "..."]        # CONDUCTOR only: reset an island to the baseline + re-explore (non-destructive)
    python hub.py summary                                   # launcher: one-line tally + champion
    python hub.py requeue                                   # evaluator startup crash recovery
    python hub.py migrate                                   # bring an old search.db to the current schema

The evaluator and setup use the Python API directly (``init`` / ``get_next_model`` / ``record`` /
``score``); those are not exposed as CLI verbs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
# Two proposer-owned complexity fields travel with each model: ``n_params`` (the model's declared
# count of hard-coded / random-but-fixed weights) and ``model_loc`` (lines of code defining the
# model -- a proxy for how elaborate it is). Both are captured from the proposed model.py at propose
# time; ``n_params`` is also refreshed from the live model attribute when the evaluator scores it.
# ``experiment_notes`` is a JSON blob {"description", "insight"} the proposer authors as model.py
# attributes; the evaluator captures it at score time and writes it back with the correlations.
NUM = ("test_corr", "train_corr", "n_params", "seconds")
COLS = ("id", "idea", "experiment_notes", "proposer", "dataset", "status",
        "test_corr", "train_corr", "roi_corr", "n_params", "model_loc",
        "seconds", "created_at", "path")
STATUSES = ("queued", "running", "ok", "error")
_ID_ALPHABET = string.ascii_lowercase + string.digits
_ID_LEN = 5

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))   # hub.py sits at the repo root

# Columns hidden from a sandboxed proposer in ``query``/``top-models``: other islands' scores + the
# on-disk source path, so islands diverge instead of all chasing one champion. (``random-models`` is
# the deliberate exception -- it surfaces scores, anonymized, no code pointers.) Conductor sees all.
_PROPOSER_HIDDEN = ("test_corr", "train_corr", "roi_corr", "path")


def _is_proposer() -> bool:
    """True when running inside a sandboxed proposer island (``LAB_PROPOSER`` set by ``proposer.sh``).
    The conductor (``conduct.sh``) and the manual/launcher paths leave it unset, so they keep full
    score visibility."""
    return bool(os.environ.get("LAB_PROPOSER"))


def _vanilla_guard(cmd: str) -> None:
    """``VANILLA_AUTORESEARCH=1``: close every cross-island channel so each proposer sees ONLY its own
    lineage -- the N-independent-loops baseline (conductor also off). Same boundary as the blinding."""
    if os.environ.get("VANILLA_AUTORESEARCH") == "1" and _is_proposer():
        raise PermissionError(f"{cmd} is OFF in vanilla-autoresearch mode (independent-loops baseline): evolve your OWN `my-best`.")
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
    if c in ("n_params", "model_loc"):
        return "INTEGER"
    return "REAL" if c in NUM else "TEXT"


def _count_loc(text: str) -> int:
    """Lines of code in a model source: non-blank, non-comment lines."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


_ATTR_RE = {
    "n_params": re.compile(r"^\s*n_params\s*=\s*([0-9_]+)\s*(?:#.*)?$", re.M),
    "model_loc": re.compile(r"^\s*model_loc\s*=\s*([0-9_]+)\s*(?:#.*)?$", re.M),
}


def model_complexity(path: str) -> dict:
    """The proposer-owned complexity fields for a proposed model, read from its source:

    - ``n_params``  -- the model's declared ``n_params = <int>`` class attribute (the agent's honest
      count of hard-coded / random-but-fixed weights), or ``None`` if not a plain literal (the
      evaluator will still capture the live ``model.n_params`` when it scores it);
    - ``model_loc`` -- the model's declared ``model_loc = <int>`` if present, else the objective
      line count of the source (so the field is always populated).

    This is a static read (no import / no torch), keeping ``propose`` light."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except (OSError, TypeError):
        return {"n_params": None, "model_loc": None}

    def declared(name):
        m = _ATTR_RE[name].search(text)
        return int(m.group(1).replace("_", "")) if m else None

    return {"n_params": declared("n_params"),
            "model_loc": declared("model_loc") if declared("model_loc") is not None else _count_loc(text)}


def _migrate(con: sqlite3.Connection) -> None:
    """Bring an older ``models`` table to the current schema (idempotent). Renames the legacy
    ``sloc`` column to ``model_loc`` if present (preserving its data), then adds any still-missing
    columns from ``COLS``."""
    existing = {r[1] for r in con.execute("PRAGMA table_info(models)")}
    if not existing:
        return
    if "sloc" in existing and "model_loc" not in existing:
        con.execute("ALTER TABLE models RENAME COLUMN sloc TO model_loc")
        existing.discard("sloc"); existing.add("model_loc")
    for c in COLS:
        if c not in existing:
            con.execute(f"ALTER TABLE models ADD COLUMN {c} {_coltype(c)}")


def init(run: str | None = None) -> str:
    """Create the run dir + ``models/`` + the ``models`` table (idempotent). Returns the run dir."""
    run = os.path.abspath(run or find_run())
    os.makedirs(models_dir(run), exist_ok=True)
    defs = ", ".join(("id TEXT PRIMARY KEY" if c == "id" else f"{c} {_coltype(c)}") for c in COLS)
    with connect(run) as con:
        con.execute(f"CREATE TABLE IF NOT EXISTS models ({defs})")
        _migrate(con)
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


# The reliability predicate shared by every "best/scored model" query: a successfully-scored row
# with a real reward. Kept in one place so the five call sites can't drift apart.
_OK = "status='ok' AND test_corr IS NOT NULL"


def _read_file(path, strip: bool = False) -> str:
    """Read a file's text, returning "" for a missing/None path (used for model source and for the
    per-island marker files). ``strip=True`` for the whitespace-sensitive marker files."""
    try:
        with open(path) as f:
            return f.read().strip() if strip else f.read()
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
def propose(model_file: str, idea: str = "",
            proposer: str | None = None, run: str | None = None) -> str:
    """Register a new model: snapshot ``model_file`` into ``models/<id>.py`` (immutable) + insert one
    ``queued`` row; return the id. A proposer's only DB write -- it can edit its working ``model.py``
    for the next attempt the instant this returns (the snapshot is what gets scored). The
    ``experiment_notes`` (description + insight) are NOT written here: they are authored as
    ``model.py`` attributes and captured by the evaluator at score time."""
    run = _ensure(run)
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    proposer = proposer or os.environ.get("LAB_PROPOSER", "")
    # Stamp created_at NOW, before the handshake + snapshot. If a reset lands anytime during this
    # call, its reset_at will be > this created_at, so the `created_at > reset_at` filter in
    # get_best_model auto-excludes this row -- closing the TOCTOU window between the handshake check
    # and the DB insert. (The handshake below catches the larger reset-before-propose window.)
    created_at = _now()
    if proposer:                       # race handshake: refuse a snapshot taken before a mid-turn reset
        if (_island_reset_at(proposer, run) or "") != _read_file(_sandbox_file(proposer, run, "reset_seen"), strip=True):
            raise RuntimeError(
                "island was RESET since you last read `my-best` -- this model.py is from your old "
                "(now-retired) lineage. Re-read `my-best` (it returns the baseline floor now) and "
                "rebuild your assigned direction.txt role from scratch; do NOT re-propose the old backbone.")
    dataset = os.environ.get("NEURO_DATASET", "")
    for _ in range(50):                       # retry on the rare id collision
        mid = newid(run)
        dst = os.path.join(models_dir(run), f"{mid}.py")
        shutil.copyfile(model_file, dst)
        cx = model_complexity(dst)            # proposer-owned complexity, read from the model source
        row = {
            "id": mid, "idea": (idea or "")[:500], "experiment_notes": None,
            "proposer": proposer, "dataset": dataset, "status": "queued",
            "test_corr": None, "train_corr": None, "roi_corr": None,
            "n_params": cx["n_params"], "model_loc": cx["model_loc"],
            "seconds": None, "created_at": created_at, "path": os.path.abspath(dst),
        }
        try:
            with connect(run) as con:
                _migrate(con)                 # self-heal an older DB missing newer columns
                con.execute(f"INSERT INTO models ({','.join(COLS)}) "
                            f"VALUES ({','.join('?' * len(COLS))})", [row[c] for c in COLS])
            return mid
        except sqlite3.IntegrityError:
            os.path.exists(dst) and os.remove(dst)
            continue
    raise RuntimeError("could not insert a queued row (repeated id collisions)")


def wait_for_result(mid: str, poll: float = _POLL, timeout: float | None = None,
                    run: str | None = None) -> dict:
    """BLOCK until model ``mid`` is scored (status ``ok``/``error``); return its row. The proposer's
    self-pacing primitive -- waits only for ITS OWN model, never the pool. Raises ``KeyError`` if the
    id is unknown, ``TimeoutError`` if ``timeout`` elapses first."""
    run = _ensure(run)
    deadline = None if timeout is None else time.time() + timeout
    while True:
        with connect(run) as con:
            r = con.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
        if r is None:
            raise KeyError(f"no such model {mid!r}")
        row = dict(r)
        if row["status"] in ("ok", "error"):
            return row
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"{mid} not scored within {timeout}s (status={row['status']})")
        time.sleep(poll)


# --------------------------------------------------------------------------- proposer: read / learn
def champions(k: int = 5, run: str | None = None) -> list[dict]:
    """The pool's public scoreboard: the ``ok`` models with the highest reward, best first, as
    scores + ideas (NO source). Source is deliberately withheld -- a proposer builds on its OWN
    island (``get_best_model``) or one random other island (``get_best_model_from_other_island``),
    never a copy of the global champion."""
    with connect(_ensure(run)) as con:
        return _rows(con, f"SELECT * FROM models WHERE {_OK} "
                          "ORDER BY test_corr DESC LIMIT ?", (int(k),))


def sample(k: int = 5, seed=None, run: str | None = None) -> list[dict]:
    """A random draw of ``ok`` models (idea + notes + scores) -- a diversity spark. Returns scores
    (``test_corr``/``train_corr``/``roi_corr``) to proposers too, so they can see which prior ideas
    worked; a proposer only loses the ``id``/``proposer`` code POINTERS (archive + sibling sandbox).
    OFF entirely under ``VANILLA_AUTORESEARCH=1``, so only the full system is affected."""
    _vanilla_guard("random-models")
    with connect(_ensure(run)) as con:
        rows = _rows(con, "SELECT id, idea, proposer, status, test_corr, train_corr, roi_corr, experiment_notes "
                          f"FROM models WHERE {_OK} "
                          "ORDER BY RANDOM() LIMIT ?", (int(k),))
    if _is_proposer():
        rows = [{k: v for k, v in r.items() if k not in ("id", "proposer")} for r in rows]
    return rows


def _sandbox_file(proposer: str, run: str, name: str) -> str:
    return os.path.join(_ensure(run), "proposers", proposer, name)




def _island_reset_at(proposer: str, run: str) -> str | None:
    """The marker of the island's last conductor reset, or ``None``. Scored rows created at or before
    this are excluded from the island's ``my-best`` -- so a reset island genuinely re-explores from
    the baseline floor instead of reloading its old (converged) lineage. The same marker is the
    handshake token: ``my-best`` records it (``reset_seen``) and ``propose`` refuses a snapshot taken
    before a reset that landed mid-turn (see :func:`propose`)."""
    if not proposer:
        return None
    return _read_file(_sandbox_file(proposer, run, "reset_at"), strip=True) or None


def get_best_model(with_code: bool = True, run: str | None = None) -> dict | None:
    """The CALLING proposer's OWN island (``$LAB_PROPOSER``) best ``ok`` model -- the thing to build
    on -- with its source unless ``with_code=False``. Each proposer is its own ISLAND: this never
    returns another proposer's work, so backbones don't all collapse onto one global champion.
    ``None`` if your island has no scored model yet (then just evolve your current ``model.py``).

    Honors a conductor RESET: rows at/before the island's ``reset_at`` marker are excluded, so a
    freshly reset island sees ``None`` and re-explores from the baseline ``model.py`` floor. Calling
    this also records the reset epoch you are syncing to (``reset_seen``), which ``propose`` checks to
    reject a stale snapshot if a reset lands mid-turn."""
    proposer = os.environ.get("LAB_PROPOSER", "")
    run = _ensure(run)
    reset_at = _island_reset_at(proposer, run)
    if proposer:                       # record the reset epoch this turn is built on (race handshake)
        with open(_sandbox_file(proposer, run, "reset_seen"), "w") as fh:
            fh.write(reset_at or "")
    sql = (f"SELECT * FROM models WHERE proposer=? AND {_OK} "
           + ("AND created_at > ? " if reset_at else "")
           + "ORDER BY test_corr DESC LIMIT 1")
    params = (proposer, reset_at) if reset_at else (proposer,)
    with connect(run) as con:
        r = con.execute(sql, params).fetchone()
    if r is None:
        return None
    row = dict(r)
    if with_code:
        row["code"] = _read_file(row.get("path"))
    return row


def get_best_model_from_other_island(run: str | None = None) -> dict | None:
    """Cross-pollination: the best ``ok`` model of ONE RANDOMLY chosen OTHER proposer (island), as
    its ``idea`` + ``experiment_notes`` (description + insight) -- NO source code and NO scores. You
    must RE-IMPLEMENT the concept, not clone a file, and you are not told how well it scored: that is
    what keeps backbones diverse and stops every island chasing the same number. You do NOT get to aim
    it -- the island is auto-picked at random, so it is never just the global champion. ``None`` if no
    other island has scored yet."""
    _vanilla_guard("peek-island")
    proposer = os.environ.get("LAB_PROPOSER", "")
    with connect(_ensure(run)) as con:
        isl = con.execute(
            f"SELECT proposer FROM models WHERE {_OK} "
            "AND proposer<>? AND proposer<>'' GROUP BY proposer ORDER BY RANDOM() LIMIT 1",
            (proposer,)).fetchone()
        if isl is None:
            return None
        r = con.execute(
            "SELECT idea, experiment_notes "
            f"FROM models WHERE proposer=? AND {_OK} "
            "ORDER BY test_corr DESC LIMIT 1", (isl["proposer"],)).fetchone()
    return dict(r)


def reset_island(proposer: str, role: str | None = None, run: str | None = None) -> dict:
    """CONDUCTOR power: hard-reset an island so it RE-EXPLORES its role from scratch.

    Atomically: (1) overwrites ``proposers/<pNN>/model.py`` with the random-projection BASELINE (the
    fresh floor), (2) writes a ``reset_at`` marker so the island's ``my-best`` ignores its old
    (converged) lineage and returns ``None`` until it scores anew, (3) CLEARS the island's ``scratch/``
    notes so the proposer can't re-derive its retired backbone from its own saved analysis, and
    (4) sets ``direction.txt`` to ``role`` if given. NON-DESTRUCTIVE to the science: archived rows +
    the global champion are untouched -- only the island's WORKING lineage + scratch are cleared -- so
    resetting any island (even the champion's) never loses recorded science or the best result. Reset
    is to the BASELINE, never a clone of another island, so the fleet regains genuine diversity.
    Conductor/launcher only (refused for a proposer)."""
    if _is_proposer():
        raise PermissionError("reset-island is a CONDUCTOR power; a proposer cannot reset islands.")
    run = _ensure(run)
    proposer = (proposer or "").strip()
    sandbox = os.path.join(run, "proposers", proposer)
    if not proposer or not os.path.isdir(sandbox):
        raise ValueError(f"reset-island: no such island sandbox {proposer!r} under {run}")
    stamp = _now()
    baseline = os.path.join(REPO_ROOT, "model.py")
    shutil.copyfile(baseline, os.path.join(sandbox, "model.py"))   # back to the floor
    with open(os.path.join(sandbox, "reset_at"), "w") as fh:       # mute the old lineage in my-best
        fh.write(stamp)
    scratch = os.path.join(sandbox, "scratch")                     # wipe the proposer's own notes so it
    if os.path.isdir(scratch):                                     # can't re-derive its retired backbone
        shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    if role:
        with open(os.path.join(sandbox, "direction.txt"), "w") as fh:
            fh.write(role.strip() + "\n")
    return {"reset": proposer, "model_py": "baseline", "scratch": "cleared",
            "reset_at": stamp, "role": role or "(unchanged)"}


def query(sql: str, params=(), run: str | None = None) -> list[dict]:
    """Read-only arbitrary SQL over ``search.db`` -- the escape hatch for any custom view an agent
    wants (it can build its own helpers in its sandbox on top of this). Rejects non-SELECT SQL.

    For a PROPOSER (``LAB_PROPOSER`` set) the global scoreboard is hidden: SQL referencing another
    island's score / source columns (``test_corr``/``train_corr``/``roi_corr``/``path``) is rejected,
    and those keys are stripped from results so a ``SELECT *`` can't surface them either. The
    conductor/launcher (no ``LAB_PROPOSER``) get unrestricted access."""
    if sql.lstrip().split(None, 1)[0].lower() not in ("select", "with"):
        raise ValueError("query() is read-only: use a SELECT/WITH statement")
    proposer = _is_proposer()
    if proposer:
        low = sql.lower()
        hit = next((c for c in _PROPOSER_HIDDEN if c in low), None)
        if hit is not None:
            raise ValueError(
                f"query: column {hit!r} is hidden from a proposer -- you cannot read other islands' "
                f"scores or source. Use `my-best` for your OWN island's score and `peek-island` for "
                f"one random other island's idea (no scores). Query idea/status/notes/proposer freely.")
    with connect(_ensure(run)) as con:
        rows = _rows(con, sql, params)
    if proposer:                       # strip score/path keys a `SELECT *` would otherwise surface
        rows = [{k: v for k, v in r.items() if k not in _PROPOSER_HIDDEN} for r in rows]
    return rows


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
    present in ``scores``, plus the ``experiment_notes`` JSON the proposer authored in model.py
    (captured by the evaluator at score time). ``id``/``idea``/``path``/... are untouched."""
    st = "ok" if str(status).strip().lower().startswith("ok") else "error"
    scores = scores or {}
    cols = [c for c in NUM if scores.get(c) not in (None, "")]
    vals = [_num(scores[c]) for c in cols]
    if scores.get("roi_corr"):                    # TEXT json blob, only when the breakdown is non-empty
        cols.append("roi_corr")
        vals.append(json.dumps(scores["roi_corr"]))
    if scores.get("experiment_notes"):            # TEXT json blob {"description","insight"} from model.py
        cols.append("experiment_notes")
        vals.append(scores["experiment_notes"])
    assignments = [f"{c}=?" for c in cols] + ["status=?"]
    vals = vals + [st, mid]
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


def migrate(run: str | None = None) -> int:
    """Bring an older ``search.db`` up to the current schema: add missing columns, then BACKFILL
    the proposer-owned complexity fields (``model_loc`` always; ``n_params`` where the archived
    model.py declares one) for rows that predate them. Returns the number of rows touched. Safe to
    re-run (only fills rows still missing the value)."""
    run = _ensure(run)
    with connect(run) as con:
        _migrate(con)
        touched = 0
        rows = con.execute(
            "SELECT id, path, n_params, model_loc FROM models WHERE path IS NOT NULL "
            "AND (model_loc IS NULL OR n_params IS NULL)").fetchall()
        for r in rows:
            cx = model_complexity(r["path"])
            sets, vals = [], []
            if r["model_loc"] is None and cx["model_loc"] is not None:
                sets.append("model_loc=?"); vals.append(cx["model_loc"])
            if r["n_params"] is None and cx["n_params"] is not None:
                sets.append("n_params=?"); vals.append(cx["n_params"])
            if sets:
                con.execute(f"UPDATE models SET {', '.join(sets)} WHERE id=?", vals + [r["id"]])
                touched += 1
    return touched


# --------------------------------------------------------------------------- launcher: DB summary
def summary(run: str | None = None) -> str:
    """One-line DB tally + current champion, for the launcher's startup/shutdown log."""
    with connect(run) as con:
        tally = dict(con.execute(
            "SELECT COALESCE(status,'?'), COUNT(*) FROM models GROUP BY 1").fetchall())
        best = con.execute(
            f"SELECT id, test_corr FROM models WHERE {_OK} "
            "ORDER BY test_corr DESC LIMIT 1").fetchone()
    t = " ".join(f"{k}={v}" for k, v in sorted(tally.items())) or "(empty)"
    champ = f"{best['id']} @ {best['test_corr']:.4f}" if best else "(none)"
    return f"DB: {t} | champion: {champ}"


# --------------------------------------------------------------------------- evaluator: scoring (lazy harness)
def score(model_file: str, dataset: str | None = None, timeout: int = 420) -> dict:
    """Score one model file with the FIXED harness in a subprocess (isolation against CUDA faults).
    Returns ``{status, test_corr, train_corr, roi_corr, experiment_notes, n_params, seconds}``;
    never raises on a model
    fault -- a crash/timeout becomes ``status='error'``. Reward is the dataset's fixed-region
    ``test_corr`` (NSD: the ventral stream); no operator ROI knob."""
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
        proc = subprocess.run([sys.executable, "-m", "hub", "_score_child", os.path.abspath(model_file)],
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
    """Child entrypoint (``python -m hub _score_child <file>``): import the harness, score the
    model on the active dataset (whole-dataset reward), write JSON to ``$LAB_SCORE_OUT``."""
    import importlib.util
    out = os.environ["LAB_SCORE_OUT"]
    res = {"status": "error: unknown"}
    try:
        from harness.datasets import DATASET
        from harness.reward import evaluate_subjects
        spec = importlib.util.spec_from_file_location("model", model_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        row = evaluate_subjects(mod.MODEL, DATASET)   # NEURO_SUBJECT list -> mean over subjects; else single
        res = {k: row.get(k) for k in ("status", "test_corr", "train_corr", "roi_corr",
                                       "experiment_notes", "n_params", "seconds")}
    except Exception as e:  # noqa: BLE001  -- any import/eval fault is a recorded error, not a crash
        import traceback
        traceback.print_exc()
        res = {"status": f"error: {type(e).__name__}: {e}"[:300]}
    with open(out, "w") as f:
        json.dump(res, f)


# --------------------------------------------------------------------------- CLI
def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def _top_models_cli(a):
    """``top-models`` is for the conductor/launcher only. A proposer calling it is refused -- the
    global scoreboard is hidden so islands diverge instead of all converging on one champion."""
    if _is_proposer():
        raise ValueError(
            "top-models is unavailable to proposers (the global scoreboard is hidden so islands stay "
            "diverse). Use `peek-island` for ONE random other island's idea (no scores) and "
            "`my-best` for your OWN island's best + score.")
    return champions(a.n)


def main(argv=None):
    p = argparse.ArgumentParser(description="hub -- the search-run API (see module docstring).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Each subcommand registers its handler (a function of the parsed args) and whether its result
    # is printed as JSON (_print) or a bare line (print) -- so there is ONE dispatch, no if/elif.
    def add(name, fn, json_out=False):
        sp = sub.add_parser(name); sp.set_defaults(_fn=fn, _json=json_out); return sp

    sp = add("propose-model", lambda a: propose(a.model_file, a.idea, a.proposer))
    sp.add_argument("model_file"); sp.add_argument("--idea", default="")
    sp.add_argument("--proposer", default=None)
    sp = add("wait-for-score", lambda a: wait_for_result(a.id, timeout=a.timeout), json_out=True)
    sp.add_argument("id"); sp.add_argument("--timeout", type=float, default=None)
    sp = add("top-models", _top_models_cli, json_out=True); sp.add_argument("-n", type=int, default=5)
    sp = add("random-models", lambda a: sample(a.n, a.seed), json_out=True)
    sp.add_argument("-n", type=int, default=5); sp.add_argument("--seed", default=None)
    sp = add("my-best", lambda a: get_best_model(not a.no_code), json_out=True)
    sp.add_argument("--no-code", action="store_true", help="return the row only, without source")
    add("peek-island", lambda a: get_best_model_from_other_island(), json_out=True)
    add("query-db", lambda a: query(a.sql), json_out=True).add_argument("sql")
    sp = add("reset-island", lambda a: reset_island(a.proposer, a.role), json_out=True)
    sp.add_argument("proposer", help="island to reset, e.g. p04")
    sp.add_argument("--role", default=None, help="new direction.txt to set on reset (optional)")
    add("requeue", lambda a: requeue_orphans())
    add("migrate", lambda a: f"migrated schema; filled complexity for {migrate()} rows")
    add("summary", lambda a: summary())
    sub.add_parser("_score_child").add_argument("model_file")  # internal: scores in a subprocess

    a = p.parse_args(argv)
    if a.cmd == "_score_child":
        _score_child(a.model_file)
        return
    (_print if a._json else print)(a._fn(a))


if __name__ == "__main__":
    main()
