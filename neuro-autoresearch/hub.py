#!/usr/bin/env python3
"""Search queue, model archive, and feedback API.

Each run contains ``search.db``, immutable ``models/<id>.py`` snapshots, and
``proposers/pNN/`` workspaces. Rows move through queued -> running -> ok/error.
``test_corr`` stores RSA search reward; final encoding never enters this queue.

Proposers submit with ``propose-model`` and await their result with
``wait-for-score``. They can inspect their own ``my-best``, anonymized
``random-models``, or one other island's ideas via ``peek-island``. The optional
conductor sees the full scoreboard and can steer or reset individual islands.

``LAB_RUN`` selects the run, otherwise the nearest ancestor with ``search.db``
is used. ``LAB_PROPOSER`` identifies an island and restricts score visibility.
Run ``hub.py --help`` for commands. Numerical dependencies load only when scoring
or checking model equivalence; queue operations remain lightweight.
"""
from __future__ import annotations

import argparse
import hashlib
import math
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

# Schema: one row per archived model, including queue status and score provenance.
# n_params is captured statically on submission and refreshed during evaluation.
# experiment_notes stores the model's description/insight JSON.
NUM = ("test_corr", "train_corr", "n_params", "seconds")
COLS = ("id", "idea", "experiment_notes", "proposer", "dataset", "status",
        "test_corr", "train_corr", "roi_corr", "n_params", "model_loc",
        "seconds", "created_at", "path", "metric", "stage", "protocol_version",
        "subject_scores", "evaluation_metadata", "model_sha256")
STATUSES = ("queued", "running", "ok", "error")
_ID_ALPHABET = string.ascii_lowercase + string.digits
_ID_LEN = 5

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))   # hub.py sits at the repo root

# Columns hidden from a sandboxed proposer in ``query``/``top-models``: other islands' scores + the
# on-disk source path, so islands diverge instead of all chasing one champion. (``random-models`` is
# the deliberate exception -- it surfaces scores, anonymized, no code pointers.) Conductor sees all.
_PROPOSER_HIDDEN = ("test_corr", "train_corr", "roi_corr", "path", "subject_scores", "evaluation_metadata")


def _is_proposer() -> bool:
    """Whether this process belongs to an island (LAB_PROPOSER is set)."""
    return bool(os.environ.get("LAB_PROPOSER"))


def _vanilla_guard(cmd: str) -> None:
    """Disable cross-island learning for the independent-loops baseline."""
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
    """Open a WAL connection with a busy timeout for concurrent queue workers."""
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
    lines = (line.strip() for line in text.splitlines())
    return sum(bool(line) and not line.startswith("#") for line in lines)


_ATTR_RE = {
    "n_params": re.compile(r"^\s*n_params\s*=\s*([0-9_]+)\s*(?:#.*)?$", re.M),
    "model_loc": re.compile(r"^\s*model_loc\s*=\s*([0-9_]+)\s*(?:#.*)?$", re.M),
}


def model_complexity(path: str) -> dict:
    """Read declared n_params/model_loc without importing candidate code.

    Fall back to a source line count for model_loc; the evaluator measures
    materialized parameters after forward."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except (OSError, TypeError):
        return {"n_params": None, "model_loc": None}

    def declared(name):
        m = _ATTR_RE[name].search(text)
        return int(m.group(1).replace("_", "")) if m else None

    model_loc = declared("model_loc")
    return {"n_params": declared("n_params"),
            "model_loc": model_loc if model_loc is not None else _count_loc(text)}


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


def search_contract(run: str, dataset: str | None = None) -> dict:
    """Reject legacy runs and final evaluation at the search API boundary."""
    from harness.protocols import protocol_for, validate_search_dataset
    path = os.path.join(run, "env.json")
    if not os.path.exists(path):
        raise ValueError("Search requires a versioned env.json; scaffold a new laion or triplen run")
    with open(path) as f:
        env = json.load(f)
    key = validate_search_dataset(env.get("NEURO_DATASET", ""))
    protocol = protocol_for(key)
    if dataset and validate_search_dataset(dataset) != key:
        raise ValueError("Dataset does not match this run's evaluation contract")
    if (env.get("NEURO_METRIC") != "rsa" or env.get("NEURO_STAGE") != "search"
            or env.get("protocol", {}).get("version") != protocol.version
            or env.get("NEURO_SUBJECT") != ",".join(protocol.search_subjects)):
        raise ValueError("Run is incompatible with the current RSA search protocol")
    return env


# --------------------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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
    """Archive a candidate and enqueue its RSA evaluation; return its ID.

    Check source, declared size, run contract, and reset acknowledgement first.
    The evaluator scores the snapshot and captures live complexity and notes."""
    run = _ensure(run)
    contract = search_contract(run, os.environ.get("NEURO_DATASET"))
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    cheat = _scan_source(model_file)              # reject reward-hacks BEFORE they queue
    if cheat:
        raise RuntimeError(
            f"model.py rejected -- reward-hack: {cheat}. Models must be HAND-WRITTEN and UNTRAINED: "
            f"analytic / fixed-seed weights in your own code, receiving only the stimulus. No "
            f"pretrained/downloaded weights, no network, no reading the neural target or the readout.")
    declared = model_complexity(model_file).get("n_params")   # cheap early filter; the harness enforces the REAL count at score
    if declared and declared > _MAX_PARAMS:
        raise RuntimeError(
            f"model.py rejected -- declared n_params {declared:,} exceeds the {_MAX_PARAMS:,} cap. "
            f"Keep the model small and hand-written -- fewer, interpretable parameters.")
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
    dataset = contract["NEURO_DATASET"]
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
            "metric": "rsa", "stage": "search", "protocol_version": contract["protocol"]["version"],
            "subject_scores": None, "evaluation_metadata": None,
            "model_sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
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
    """Wait for this model to reach ok/error and return its row.

    Raise KeyError for an unknown ID or TimeoutError if the deadline expires."""
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
    """Return the highest-scoring successful rows, including archive paths.

    This trusted Python API is exposed to conductors through top-models;
    proposer access is restricted by the CLI."""
    with connect(_ensure(run)) as con:
        return _rows(con, f"SELECT * FROM models WHERE {_OK} "
                          "ORDER BY test_corr DESC LIMIT ?", (int(k),))


def champion(run: str | None = None) -> dict | None:
    """Return the best scored row, or None. Used to select a frozen model."""
    rows = champions(1, run)
    return rows[0] if rows else None


def sample(k: int = 5, seed=None, run: str | None = None) -> list[dict]:
    """Sample scored ideas and notes, hiding IDs and proposer names from islands.

    Unlike query-db, this channel exposes anonymized scores. It is disabled
    for proposers in vanilla mode. The seed argument is retained for compatibility."""
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
    """Read the reset marker used to exclude old lineage and reject stale proposals."""
    if not proposer:
        return None
    return _read_file(_sandbox_file(proposer, run, "reset_at"), strip=True) or None


def get_best_model(with_code: bool = True, run: str | None = None) -> dict | None:
    """Return this island's best model since its last reset, optionally with code.

    Acknowledge the current reset marker so propose can reject a model built
    across a concurrent reset. Return None until the island scores a new model."""
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
    """Return the best idea/notes from one random other island, without scores or code."""
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
    """Restore an island's baseline, clear scratch, and optionally set its direction.

    Mark its previous lineage retired without changing archived models or
    scores. Only a conductor or operator may reset an island."""
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
    """Run a SELECT/WITH query under proposer score-visibility rules.

    Proposers cannot name hidden score/source columns; those columns are
    also removed from SELECT * results. Conductors see the complete rows."""
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
    """Atomically claim the oldest queued model; return None if the queue is empty."""
    with connect(_ensure(run)) as con:
        r = con.execute(
            "UPDATE models SET status='running' "
            "WHERE id=(SELECT id FROM models WHERE status='queued' ORDER BY created_at, rowid LIMIT 1) "
            "RETURNING *").fetchone()
    return dict(r) if r else None


def record(mid: str, scores: dict | None = None, status: str = "ok",
           run: str | None = None) -> str:
    """Validate and persist evaluation feedback without changing model identity.

    Only finite, same-dataset, full-cohort RSA results can become search rewards."""
    st = "ok" if str(status).strip().lower().startswith("ok") else "error"
    scores = scores or {}
    run = _ensure(run)
    with connect(run) as con:
        _migrate(con)
        queued = con.execute("SELECT dataset,metric,stage,protocol_version,path FROM models WHERE id=?", (mid,)).fetchone()
    if queued is None:
        raise KeyError(mid)
    if st == "ok" and (_num(scores.get("test_corr")) is None
                        or not str(scores.get("status", "ok")).startswith("ok")):
        st = "error"
    if st == "ok" and queued["metric"] == "rsa":
        from harness.protocols import protocol_for, SOURCE_REVISION
        protocol = protocol_for(queued["dataset"])
        required = set(protocol.search_subjects)
        subject_scores = scores.get("subject_scores") or {}
        subjects = scores.get("subjects") or []
        cohort_ok = (isinstance(subject_scores, dict) and set(subject_scores) == required
                     and len(subjects) == len(required) and set(subjects) == required
                     and all(_num(value) is not None for value in subject_scores.values()))
        mean_ok = cohort_ok and math.isclose(float(scores["test_corr"]),
            sum(float(value) for value in subject_scores.values()) / len(required), abs_tol=1e-10)
        if (scores.get("metric") != "rsa" or scores.get("stage") != "search"
                or scores.get("protocol_version") != queued["protocol_version"]
                or scores.get("dataset") != queued["dataset"]
                or scores.get("source_revision") != SOURCE_REVISION or not mean_ok):
            st = "error"
            scores = {"status": "error: incompatible evaluation result; expected same-dataset full-cohort RSA"}
    updates = {c: _num(scores[c]) for c in NUM if scores.get(c) not in (None, "")}
    if scores.get("roi_corr"):
        updates["roi_corr"] = json.dumps(scores["roi_corr"])
    if scores.get("experiment_notes"):
        updates["experiment_notes"] = scores["experiment_notes"]
    if scores.get("subject_scores"):
        updates["subject_scores"] = json.dumps(scores["subject_scores"], allow_nan=False)
    metadata = {k: scores[k] for k in (
        "dataset", "metric", "stage", "protocol_version", "aggregation", "subjects", "search_image_policy",
        "source_revision", "subject_roi_scores", "n_train", "n_test", "n_targets", "status") if k in scores}
    updates["evaluation_metadata"] = json.dumps(metadata, allow_nan=False)
    if queued["path"] and os.path.isfile(queued["path"]):
        with open(queued["path"], "rb") as source:
            updates["model_sha256"] = hashlib.sha256(source.read()).hexdigest()
    updates["status"] = st
    assignments = ", ".join(f"{column}=?" for column in updates)
    with connect(run) as con:
        con.execute(f"UPDATE models SET {assignments} WHERE id=?", [*updates.values(), mid])
    return st


def requeue_orphans(run: str | None = None) -> int:
    """Startup crash recovery: any ``running`` row (the evaluator died mid-score) goes back to
    ``queued`` so it is re-claimed. Returns the number recovered."""
    if not os.path.exists(db_path(os.path.abspath(run or find_run()))):
        return 0
    with connect(run) as con:
        return con.execute("UPDATE models SET status='queued' WHERE status='running'").rowcount


def migrate(run: str | None = None) -> int:
    """Upgrade the schema and backfill missing source complexity; return rows updated."""
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


# --------------------------------------------------------------------------- reward-integrity guard
# The reward is real only if the model is HAND-WRITTEN & UNTRAINED (analytic/fixed-seed weights in its
# own code, fed only the stimulus). Reject two cheat families -- loading PRETRAINED weights, and
# reading the target/readout (leakage) -- statically (source scan) + at runtime (no network, no
# pretrained-lib imports), and cap the params. Nothing hand-written needs any of these.
_BANNED_IMPORTS = ("transformers", "huggingface_hub", "timm", "open_clip",
                   "sentence_transformers", "sonar", "fairseq", "diffusers")
_CHEAT_PATTERNS = (
    (r"from_pretrained|AutoModel|AutoTokenizer|\bGPT2\w*|\btransformers\b|huggingface|\btimm\b|"
     r"open_clip|sentence_transformers|torch\.hub\.load|torchvision\.models", "loads a pretrained model"),
    (r"\burllib\b|\brequests\b|https?://|socket\.|\bwget\b|\bcurl\b|snapshot_download", "network / download"),
    (r"harness\.(reward|ridge|loaders|metrics|vendor|datasets)|looped\.data|looped\.metrics|\bload_split\b|reward_region_mask", "reads the target/readout"),
    (r"corrcoef\s*=|evaluate_on_split\s*=|\bridge\.\w+\s*=|\breward\.\w+\s*=", "monkeypatches the readout"),
)
_MAX_PARAMS = int(os.environ.get("NEURO_MAX_PARAMS") or 100_000)   # propose-time filter; harness enforces robustly at score
_GUARD_INSTALLED = False


def _scan_source(path):
    """Reason string if the model SOURCE contains an obvious reward-hack, else None."""
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    return next((why for pat, why in _CHEAT_PATTERNS if re.search(pat, src)), None)


def _install_integrity_guard():
    """Block the pretrained/download cheat at runtime: no network, no pretrained-lib imports.
    Idempotent + process-wide (the harness needs neither)."""
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    import socket, importlib.abc
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

    def _no_net(*_a, **_k):
        raise RuntimeError("network disabled while scoring -- hand-written models only (no downloads)")
    socket.socket.connect = socket.create_connection = _no_net

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BANNED_IMPORTS:
                raise ImportError(f"import of {name!r} blocked -- hand-written models only")
    sys.meta_path.insert(0, _Block())
    _GUARD_INSTALLED = True


# --------------------------------------------------------------------------- evaluator: scoring (lazy harness)
def score(model_file: str, dataset: str | None = None) -> dict:
    """Score one model with the FIXED harness IN-PROCESS, reusing the per-process split cache: the
    torch import, CUDA init, and the split load are paid ONCE across every model this process scores
    (~50x the old subprocess-per-model throughput). Never raises on a model fault -- an exception
    becomes ``status='error'``. Returns ``{status, test_corr, train_corr, roi_corr, experiment_notes,
    n_params, seconds, log}`` (``log`` = captured-output tail, for repair). ONE dataset per process
    (``DATASET`` is import-bound). A model that HANGS (native CUDA/C, uninterruptible in-process) or
    hard-crashes is handled by the evaluator's watchdog, which records the error and restarts the
    process (``requeue_orphans`` recovers the row)."""
    import contextlib, importlib.util, io, traceback
    from harness.protocols import validate_search_dataset
    dataset = validate_search_dataset(dataset or os.environ.get("NEURO_DATASET", "laion"))
    if os.environ.get("NEURO_STAGE", "search") != "search" or os.environ.get("NEURO_METRIC", "rsa") != "rsa":
        raise ValueError("hub.score is search-only; final encoding goes through experiments/evaluation/final.py")
    if os.environ.get("LAB_RUN"):
        search_contract(find_run(), dataset)
        from search.setup_run import write_env_json
        write_env_json(find_run(), dataset)
    os.environ["NEURO_DATASET"] = dataset
    os.environ.pop("NEURO_ROI", None)             # no ROI knob: reward = the preset's fixed region
    _install_integrity_guard()                    # no network / no pretrained-model imports while scoring
    cheat = _scan_source(model_file)              # static reject (also catches already-queued cheats)
    if cheat:
        return {"status": f"error: reward-hack blocked -- {cheat}"[:300], "log": ""}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            from harness.datasets import get_dataset
            from harness.reward import evaluate_subjects
            spec = importlib.util.spec_from_file_location("model", model_file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            row = evaluate_subjects(mod.MODEL, get_dataset(dataset), stage="search")
                                                          # (the harness robustly enforces the param cap)
        res = dict(row)
    except Exception as e:  # noqa: BLE001  -- any eval fault is a recorded error, not a crash
        traceback.print_exc(file=buf)
        res = {"status": f"error: {type(e).__name__}: {e}"[:300]}
    res.setdefault("log", buf.getvalue()[-4000:])
    return res


# --------------------------------------------------------------------------- CLI
def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def _top_models_cli(a):
    """Expose the global scoreboard to conductors/operators only."""
    if _is_proposer():
        raise ValueError(
            "top-models is unavailable to proposers (the global scoreboard is hidden so islands stay "
            "diverse). Use `peek-island` for ONE random other island's idea (no scores) and "
            "`my-best` for your OWN island's best + score.")
    return champions(a.n)


def _champion_cli(a):
    """Expose the global champion and source path to tooling, never proposers."""
    if _is_proposer():
        raise ValueError(
            "champion is unavailable to proposers (the global scoreboard is hidden so islands stay "
            "diverse). Use `my-best` for your own island and `peek-island` for one random other.")
    return champion()


# --------------------------------------------------------------------------- simplify: output equivalence
def _load_model_instance(model_file: str):
    """Load a candidate MODEL for a direct activation-equivalence check."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_eqmodel", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def _synthetic_batch(spec: dict):
    """Generate a reproducible batch with the captured stimulus shape, dtype, and range."""
    import numpy as np
    import torch
    g = torch.Generator().manual_seed(int(spec.get("seed", 0)))
    shape = (int(spec["batch"]), *[int(d) for d in spec["stim_shape"]])
    dtype = getattr(torch, str(spec["dtype"]).replace("torch.", ""))
    lo, hi = float(spec.get("low", 0.0)), float(spec.get("high", 1.0))
    if dtype.is_floating_point:
        x = torch.rand(shape, generator=g, dtype=torch.float64) * (hi - lo) + lo
        return x.to(dtype)
    return torch.randint(int(np.floor(lo)), int(np.floor(hi)) + 1, shape, generator=g).to(dtype)


def check_equivalence(candidate_file: str, reference_file: str, spec_file: str,
                      rtol: float = 1e-4, atol: float = 1e-6) -> dict:
    """Compare candidate/reference activations on a fixed synthetic batch.

    Return numerical differences and an equivalence flag, or a captured error.
    This checks the sampled inputs; it is not a proof for every possible input."""
    import json as _json
    import torch
    try:
        with torch.no_grad():
            spec = _json.load(open(spec_file))
            X = _synthetic_batch(spec)
            a_ref = torch.as_tensor(_load_model_instance(reference_file)(X)).float().cpu()
            a_cand = torch.as_tensor(_load_model_instance(candidate_file)(X)).float().cpu()
        if a_ref.shape != a_cand.shape:
            return {"equivalent": False, "reason": "shape mismatch",
                    "ref_shape": list(a_ref.shape), "cand_shape": list(a_cand.shape)}
        diff = (a_ref - a_cand).abs()
        denom = a_ref.abs().clamp_min(1e-12)
        equivalent = bool(torch.allclose(a_cand, a_ref, rtol=rtol, atol=atol))
        return {"equivalent": equivalent,
                "max_abs_diff": float(diff.max()), "max_rel_diff": float((diff / denom).max()),
                "ref_shape": list(a_ref.shape), "cand_shape": list(a_cand.shape),
                "rtol": rtol, "atol": atol}
    except Exception as e:  # noqa: BLE001 -- a candidate that won't import/run is a failed check, not a crash
        import traceback
        return {"equivalent": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:]}


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
    add("champion", _champion_cli, json_out=True)
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
    sp = add("check-equivalence",
             lambda a: check_equivalence(a.candidate, a.reference, a.spec, a.rtol, a.atol),
             json_out=True)
    sp.add_argument("candidate", help="the refactored model.py to verify")
    sp.add_argument("--reference", required=True, help="the frozen original model.py")
    sp.add_argument("--spec", required=True, help="input_spec.json (synthetic-batch shape/dtype/range)")
    sp.add_argument("--rtol", type=float, default=1e-4)
    sp.add_argument("--atol", type=float, default=1e-6)
    add("summary", lambda a: summary())

    a = p.parse_args(argv)
    (_print if a._json else print)(a._fn(a))


if __name__ == "__main__":
    main()
