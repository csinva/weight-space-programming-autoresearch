"""Shared read-only access to a search run's ``search.db``.

This is the keystone the dashboard backend is built on. It resolves which run we are
serving, opens its ``search.db`` read-only, and maps our ``models`` rows onto the shape
CORAL's frontend expects (an ``Attempt``). Every endpoint handler builds on the helpers
here so the mapping lives in exactly one place.

Run dirs live at ``runs/<task>__<run>/`` (e.g. ``runs/nsd__2026-06-24``): task = dataset,
run = date. ``search.db`` has one ``models`` row per model ever proposed; its ``status`` is
the queue flag (queued -> running -> ok / error).
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"

# A proposer/run counts as "alive" if something happened this recently (seconds).
ALIVE_WINDOW = 180.0
# A proposer is "active" only while actively composing a proposal (it appends to its log
# on every reasoning step / tool call). Once it submits a model it blocks waiting on the
# evaluator -> idle. This window guards against brief pauses (e.g. a slow local command).
LOG_ACTIVE_WINDOW = 180.0

# our model.status -> CORAL StatusBadge vocabulary (improved is set per-row below)
_STATUS_MAP = {"error": "crashed", "running": "active", "queued": "idle"}


# --------------------------------------------------------------------------- run discovery
def list_run_dirs() -> list[Path]:
    """Every run dir with a ``search.db``, oldest dir name first."""
    if not RUNS_ROOT.exists():
        return []
    return sorted(p for p in RUNS_ROOT.glob("*__*") if (p / "search.db").exists())


def split_run_name(p: Path) -> tuple[str, str]:
    """``runs/nsd__2026-06-24`` -> ``("nsd", "2026-06-24")``."""
    task, _, run = p.name.partition("__")
    return task, run


def run_dir(task: str, run: str) -> Path:
    return RUNS_ROOT / f"{task}__{run}"


def default_run_dir() -> Path | None:
    """Latest run dir (by name) — what the dashboard opens on by default."""
    dirs = list_run_dirs()
    return dirs[-1] if dirs else None


# --------------------------------------------------------------------------- db access
def connect(rd: Path) -> sqlite3.Connection:
    """Read-only connection to ``rd/search.db`` (never mutate a live run)."""
    c = sqlite3.connect(f"file:{(rd / 'search.db').as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def fetch_models(rd: Path) -> list[dict[str, Any]]:
    """All ``models`` rows, oldest first (proposal order)."""
    with connect(rd) as c:
        return [dict(r) for r in c.execute("SELECT * FROM models ORDER BY created_at, ROWID")]


# --------------------------------------------------------------------------- time helpers
def parse_ts(s: str | None) -> float:
    """ISO8601 (e.g. ``2026-06-24T09:37:50+00:00``) -> epoch seconds; 0 on failure."""
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0


def now() -> float:
    return datetime.now(timezone.utc).timestamp()


# --------------------------------------------------------------------------- the core mapping
def model_to_attempt(row: dict[str, Any], status: str) -> dict[str, Any]:
    """One ``models`` row -> one CORAL ``Attempt`` dict.

    ``status`` is passed in because "improved" vs "baseline" depends on the running best,
    which only :func:`attempts_for` can see across the whole ordered list.
    """
    idea = row.get("idea") or ""
    summary = row.get("summary") or ""
    bits = []
    if idea:
        bits.append(idea)
    if summary and summary != idea:
        bits.append(summary)
    metrics = []
    for k in ("test_corr", "train_corr", "gap", "n_params", "model_loc", "seconds"):
        v = row.get(k)
        if v is not None:
            metrics.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    if metrics:
        bits.append("  ".join(metrics))
    return {
        "commit_hash": row["id"],
        "agent_id": row.get("proposer") or "?",
        "title": summary or idea or row["id"],
        "score": round(row["test_corr"], 5) if row.get("test_corr") is not None else None,
        "status": status,
        "parent_hash": None,  # our search keeps no lineage
        "timestamp": row.get("created_at") or "",
        "feedback": "\n\n".join(bits),
    }


def attempts_for(rd: Path) -> list[dict[str, Any]]:
    """Full mapped ``Attempt`` list for a run, newest first.

    Walks rows in proposal order to flag each ``ok`` model that set a new running-best
    as ``improved`` (highlighted in the chart); other ``ok`` rows are ``baseline``.
    """
    rows = fetch_models(rd)
    best = float("-inf")
    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("status") or "queued"
        if raw == "ok":
            sc = row.get("test_corr")
            if sc is not None and sc > best:
                best = sc
                status = "improved"
            else:
                status = "baseline"
        else:
            status = _STATUS_MAP.get(raw, "baseline")
        out.append(model_to_attempt(row, status))
    out.reverse()  # newest first
    return out


def list_proposers(rd: Path) -> list[str]:
    """Proposer ids (``p00``, ``p01``, …) that have a sandbox in this run."""
    pdir = rd / "proposers"
    if not pdir.exists():
        return []
    return sorted(p.name for p in pdir.iterdir() if p.is_dir() and p.name.startswith("p"))


def proposer_direction(rd: Path, pid: str) -> str:
    """A proposer's self-chosen specialization (``direction.txt``), or ""."""
    try:
        return (rd / "proposers" / pid / "direction.txt").read_text().strip()
    except OSError:
        return ""


def read_model_source(rd: Path, model_id: str) -> str:
    """The never-trained ``models/<id>.py`` source, or "" if missing."""
    try:
        return (rd / "models" / f"{model_id}.py").read_text(errors="replace")
    except OSError:
        return ""


def direction_history(rd: Path) -> dict[str, list[str]]:
    """Per-proposer history of standing directions, oldest first.

    The conductor rewrites each ``direction.txt`` every epoch via
    ``echo "<dir>" > .../pNN/direction.txt``; ``conductor.log`` is the only record of the
    earlier ones. We reconstruct the rendered command lines and pull out each clean
    (fully-quoted) direction; truncated log entries are simply skipped.
    """
    try:
        log = (rd / "conductor.log").read_text(errors="replace")
    except OSError:
        return {}
    # rejoin the rendered "  │ " command-body wrap so each echo is on one logical line
    cleaned = re.sub(r"\n\s*│\s?", " ", log)
    hist: dict[str, list[str]] = {}
    for m in re.finditer(
        r'echo "(.*?)"\s*>\s*"?\$?[^"]*?/(p\d\d)/direction\.txt', cleaned
    ):
        hist.setdefault(m.group(2), []).append(m.group(1).strip())
    return hist


def previous_direction(history: list[str], current: str) -> str:
    """The most recent direction that differs from ``current`` (the prior front), or ""."""
    cur = current.strip()
    for d in reversed(history):
        if d.strip() != cur:
            return d
    return ""


def proposer_log_mtime(rd: Path, pid: str) -> float:
    """Epoch mtime of a proposer's ``proposer.log`` (how recently it did anything); 0 if none.

    This is the ground-truth activity signal: a running proposer appends to its log on
    every reasoning step and tool call, so a fresh mtime means the process is working.
    """
    try:
        return (rd / "proposers" / pid / "proposer.log").stat().st_mtime
    except OSError:
        return 0.0


def inflight_proposers(rd: Path) -> set[str]:
    """Proposers that have a model queued or running (i.e. waiting on the GPU evaluator)."""
    try:
        with connect(rd) as c:
            rows = c.execute(
                "SELECT DISTINCT proposer FROM models WHERE status IN ('queued', 'running')"
            ).fetchall()
        return {r[0] for r in rows if r[0]}
    except Exception:
        return set()


def agent_activity(rd: Path, pid: str, run_alive: bool, inflight: set[str]) -> tuple[str, float]:
    """Accurate ``(status, last_activity_epoch)`` for one proposer.

    A proposer's loop is: compose a model -> submit it -> *block* waiting for its score ->
    repeat. So:
      - ``idle``   = blocked waiting on the evaluator (it has a queued/running model), or
                     its log has gone quiet (stalled) — it is not doing work right now;
      - ``active`` = actively composing its next proposal (no in-flight model and it wrote
                     to its log within ``LOG_ACTIVE_WINDOW``);
      - ``stopped``= the whole search is down.
    """
    log_mtime = proposer_log_mtime(rd, pid)
    if not run_alive:
        return "stopped", log_mtime
    if pid in inflight:
        return "idle", log_mtime  # submitted a model, waiting on its evaluation
    if log_mtime > 0 and (now() - log_mtime) < LOG_ACTIVE_WINDOW:
        return "active", log_mtime
    return "idle", log_mtime


def run_is_alive(rd: Path) -> bool:
    """Alive if a model is in-flight, a model was scored recently, or any proposer log is fresh."""
    try:
        rows = fetch_models(rd)
    except Exception:
        rows = []
    if any((r.get("status") or "") in ("queued", "running") for r in rows):
        return True
    latest_model = max((parse_ts(r.get("created_at")) for r in rows), default=0.0)
    if latest_model and (now() - latest_model) < ALIVE_WINDOW:
        return True
    pdir = rd / "proposers"
    if pdir.exists():
        for p in pdir.iterdir():
            try:
                if (now() - (p / "proposer.log").stat().st_mtime) < LOG_ACTIVE_WINDOW:
                    return True
            except OSError:
                pass
    return False
