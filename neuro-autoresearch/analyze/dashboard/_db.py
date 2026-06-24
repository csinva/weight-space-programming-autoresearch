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

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"

# A proposer/run counts as "alive" if something happened this recently (seconds).
ALIVE_WINDOW = 180.0

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
        "score": row.get("test_corr"),
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


def run_is_alive(rd: Path) -> bool:
    """Heuristic: a queued/running model, or any activity within ALIVE_WINDOW."""
    rows = fetch_models(rd)
    if any((r.get("status") or "") in ("queued", "running") for r in rows):
        return True
    latest = max((parse_ts(r.get("created_at")) for r in rows), default=0.0)
    return latest > 0 and (now() - latest) < ALIVE_WINDOW
