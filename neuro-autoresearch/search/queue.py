#!/usr/bin/env python3
"""Trusted search database, immutable model archive, and evaluation queue.

Rows move through queued -> running -> ok/error. Only validated full-cohort RSA
results enter this database. Role-scoped feedback and access rules live in search.api.
"""
from __future__ import annotations

import hashlib
import math
import json
import os
import re
import secrets
import shutil
import sqlite3
import string
from datetime import datetime, timezone

from neuro_eval.model import scan_source, model_complexity

# Schema: one row per archived model, including queue status and score provenance.
# n_params is captured statically on submission and refreshed during evaluation.
# experiment_notes stores the model's description/insight JSON.
NUM = ("test_corr", "train_corr", "n_params", "seconds")
COLS = ("id", "idea", "experiment_notes", "proposer", "dataset", "status",
        "test_corr", "train_corr", "roi_corr", "n_params", "model_loc",
        "seconds", "created_at", "path", "metric", "stage", "protocol_version",
        "subject_scores", "evaluation_metadata", "model_sha256", "parent_id", "repair_depth")
_ID_ALPHABET = string.ascii_lowercase + string.digits
_ID_LEN = 5

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
                "no run found: set $LAB_RUN to a search_runs/<dataset>__<date>/ dir, or run inside one")
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
    if c == "repair_depth":
        return "INTEGER DEFAULT 0"
    if c in ("n_params", "model_loc"):
        return "INTEGER"
    return "REAL" if c in NUM else "TEXT"


def init(run: str | None = None) -> str:
    """Create the run dir + ``models/`` + the ``models`` table (idempotent). Returns the run dir."""
    run = os.path.abspath(run or find_run())
    os.makedirs(models_dir(run), exist_ok=True)
    defs = ", ".join(("id TEXT PRIMARY KEY" if c == "id" else f"{c} {_coltype(c)}") for c in COLS)
    with connect(run) as con:
        con.execute(f"CREATE TABLE IF NOT EXISTS models ({defs})")
        existing = {row[1] for row in con.execute("PRAGMA table_info(models)")}
        if existing != set(COLS):
            raise ValueError("Incompatible queue schema; start a fresh run")
    return run


def _ensure(run: str | None = None) -> str:
    run = os.path.abspath(run or find_run())
    if not os.path.exists(db_path(run)):
        init(run)
    return run


def search_contract(run: str, dataset: str | None = None) -> dict:
    """Reject legacy runs and final evaluation at the search API boundary."""
    from neuro_eval.benchmarks import protocol_for, validate_search_dataset
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
            proposer: str | None = None, run: str | None = None, *, parent_id: str | None = None,
            repair_depth: int = 0) -> str:
    """Archive a candidate and enqueue its RSA evaluation; return its ID.

    Check source, declared size, run contract, and reset acknowledgement first.
    The evaluator scores the snapshot and captures live complexity and notes."""
    if type(repair_depth) is not int or repair_depth < 0:
        raise ValueError("repair_depth must be a nonnegative integer")
    run = _ensure(run)
    contract = search_contract(run, os.environ.get("NEURO_DATASET"))
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    cheat = scan_source(model_file)              # reject reward-hacks BEFORE they queue
    if cheat:
        raise RuntimeError(
            f"model.py rejected -- reward-hack: {cheat}. Models must be HAND-WRITTEN and UNTRAINED: "
            f"analytic / fixed-seed weights in your own code, receiving only the stimulus. No "
            f"pretrained/downloaded weights, no network, no reading the neural target or the readout.")
    declared = model_complexity(model_file).get("n_params")   # cheap early filter; the evaluator enforces the REAL count at score
    if declared and declared > _MAX_PARAMS:
        raise RuntimeError(
            f"model.py rejected -- declared n_params {declared:,} exceeds the {_MAX_PARAMS:,} cap. "
            f"Keep the model small and hand-written -- fewer, interpretable parameters.")
    proposer = proposer or os.environ.get("LAB_PROPOSER", "")
    # Stamp created_at NOW, before the handshake + snapshot. If a reset lands anytime during this
    # call, its reset_at will be > this created_at, so the `created_at > reset_at` filter in
    # the API auto-excludes this row -- closing the TOCTOU window between the handshake check
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
        try:
            with open(model_file, "rb") as source, open(dst, "xb") as snapshot:
                shutil.copyfileobj(source, snapshot)
            os.chmod(dst, 0o444)
        except FileExistsError:
            continue
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
            "parent_id": parent_id, "repair_depth": repair_depth,
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


def champion(run: str | None = None) -> dict | None:
    """Read the best successful RSA archive row for trusted integrity checks."""
    with connect(_ensure(run)) as con:
        row = con.execute("SELECT * FROM models WHERE status='ok' AND test_corr IS NOT NULL "
                          "AND metric='rsa' AND stage='search' ORDER BY test_corr DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def _sandbox_file(proposer: str, run: str, name: str) -> str:
    """Private control state; never follow files in agent-writable workspaces."""
    if not re.fullmatch(r"p[0-9]{2,}", proposer or ""):
        raise ValueError("Invalid proposer identity")
    if name not in {"reset_at", "reset_seen", "direction.txt"}:
        raise ValueError("Invalid control marker")
    directory = os.path.join(_ensure(run), "state", proposer)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def _island_reset_at(proposer: str, run: str) -> str | None:
    """Read the reset marker used to exclude old lineage and reject stale proposals."""
    if not proposer:
        return None
    return _read_file(_sandbox_file(proposer, run, "reset_at"), strip=True) or None


def reset_island(proposer: str, role: str | None = None, run: str | None = None) -> dict:
    """Retire a lineage and set private direction; leave agent files untouched."""
    run = _ensure(run)
    proposer = (proposer or "").strip()
    marker = _sandbox_file(proposer, run, "reset_at")
    if not os.path.isdir(os.path.join(run, "proposers", proposer)):
        raise ValueError(f"reset-island: no such island {proposer!r}")
    stamp = _now()
    with open(marker, "w") as fh:
        fh.write(stamp)
    if role:
        with open(_sandbox_file(proposer, run, "direction.txt"), "w") as fh:
            fh.write(role.strip()[:2000] + "\n")
    return {"reset": proposer, "reset_at": stamp, "role": role or "(unchanged)"}


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
        queued = con.execute("SELECT dataset,metric,stage,protocol_version,path,model_sha256 FROM models WHERE id=?", (mid,)).fetchone()
    if queued is None:
        raise KeyError(mid)
    if st == "ok" and (_num(scores.get("test_corr")) is None
                        or not str(scores.get("status", "ok")).startswith("ok")):
        st = "error"
    if st == "ok" and queued["metric"] == "rsa":
        from neuro_eval.benchmarks import protocol_for, SOURCE_REVISION
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
        "source_revision", "subject_roi_scores", "n_train", "n_test", "n_targets", "rsa_diagnostics", "status", "candidate_error") if k in scores}
    updates["evaluation_metadata"] = json.dumps(metadata, allow_nan=False)
    if not queued["path"] or not os.path.isfile(queued["path"]):
        st = "error"
        updates["test_corr"] = None
    else:
        with open(queued["path"], "rb") as source:
            actual_hash = hashlib.sha256(source.read()).hexdigest()
        if actual_hash != queued["model_sha256"]:
            st = "error"
            updates["test_corr"] = None
            updates["evaluation_metadata"] = json.dumps({"status": "error: archived source identity changed"})
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


_MAX_PARAMS = int(os.environ.get("NEURO_MAX_PARAMS") or 100_000)
