"""Data layer for ``dashboard_next`` — reuses the proven read-only access in
``dashboard._db`` and adds the two shapes the redesign needs:

  * per-proposer **model + effort** (parsed from each ``proposer.log`` session header,
    with the launch roster as a round-robin fallback), and
  * each proposer's **best scored model** with its ``experiment_notes`` unpacked and its
    archived ``model.py`` source — for the new Knowledge gallery.

Nothing here writes; a live search run can be served safely.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dashboard import _db  # reuse the keystone read-only layer unchanged

# The launch roster (mirror of search/propose.sh DEFAULT_ROSTER) — only a *fallback* for
# model/effort when a proposer.log header is missing; the log is the run-accurate source.
DEFAULT_ROSTER = [
    "claude-opus-4.8", "claude-opus-4.7", "claude-opus-4.6",
    "gpt-5.5", "gemini-3.1-pro-preview:high",
]

# "[proposer] p00 session start 2026-06-26T04:04:06EDT (claude-opus-4.8/medium)"
_HEADER_RE = re.compile(r"\(([^()/]+)/([^()/]+)\)\s*$")


# --------------------------------------------------------------------------- model/effort
def proposer_model_effort(rd: Path, pid: str) -> tuple[str, str]:
    """``(model, effort)`` for one proposer, from the LAST session header in its log.

    Falls back to the round-robin roster assignment (model only) when no log/header is
    found, so a freshly scaffolded run still shows something sensible.
    """
    log = rd / "proposers" / pid / "proposer.log"
    model = effort = ""
    try:
        with log.open(errors="replace") as fh:
            for line in fh:
                if "session start" in line:
                    m = _HEADER_RE.search(line.strip())
                    if m:
                        model, effort = m.group(1).strip(), m.group(2).strip()
    except OSError:
        pass
    if not model:
        try:
            idx = int(pid[1:])
            entry = DEFAULT_ROSTER[idx % len(DEFAULT_ROSTER)]
            model, _, e = entry.partition(":")
            effort = effort or e
        except (ValueError, IndexError):
            pass
    return model, effort


# --------------------------------------------------------------------------- notes parse
def _as_float(v: Any) -> float | None:
    """Coerce DB values that are occasionally strings to float; None if not numeric."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_notes(experiment_notes: str | None) -> dict[str, str]:
    """``experiment_notes`` JSON -> ``{"description","insight"}`` (best-effort)."""
    if not experiment_notes:
        return {"description": "", "insight": ""}
    try:
        d = json.loads(experiment_notes)
        if isinstance(d, dict):
            return {
                "description": str(d.get("description", "")),
                "insight": str(d.get("insight", "")),
            }
    except (ValueError, TypeError):
        pass
    return {"description": str(experiment_notes), "insight": ""}


# --------------------------------------------------------------------------- aggregates
def _scored_rows(rd: Path) -> list[dict[str, Any]]:
    return [r for r in _db.fetch_models(rd) if (r.get("status") == "ok"
                                                and r.get("test_corr") is not None)]


def overview_meta(rd: Path) -> dict[str, Any]:
    """Header numbers: totals, best score and the best model's 1-based proposal index."""
    rows = _db.fetch_models(rd)  # oldest first (proposal order)
    scored = [(i, r) for i, r in enumerate(rows, start=1)
              if r.get("status") == "ok" and r.get("test_corr") is not None]
    best_idx, best_row = (None, None)
    if scored:
        best_idx, best_row = max(scored, key=lambda t: t[1]["test_corr"])
    return {
        "total": len(rows),
        "scored": len(scored),
        "best_score": round(best_row["test_corr"], 4) if best_row else None,
        "best_title": (best_row.get("idea") or "") if best_row else "",
        "best_index": best_idx,
    }


def top_status(rd: Path) -> str:
    """One word for the header chip: stopped / active / idle."""
    if not _db.run_is_alive(rd):
        return "stopped"
    inflight = _db.inflight_proposers(rd)
    for pid in _db.list_proposers(rd):
        status, _ = _db.agent_activity(rd, pid, True, inflight)
        if status == "active":
            return "active"
    return "idle"


def attempts(rd: Path) -> list[dict[str, Any]]:
    """Every attempt, newest first, with a stable 1-based proposal ``index``."""
    rows = _db.fetch_models(rd)  # oldest first
    best = float("-inf")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        raw = row.get("status") or "queued"
        sc = row.get("test_corr")
        if raw == "ok":
            if sc is not None and sc > best:
                best = sc
                status = "improved"
            else:
                status = "baseline"
        else:
            status = _db._STATUS_MAP.get(raw, "baseline")
        out.append({
            "index": i,
            "hash": row["id"],
            "agent": row.get("proposer") or "?",
            "title": row.get("idea") or row["id"],
            "score": round(sc, 4) if sc is not None else None,
            "status": status,
            "time": row.get("created_at") or "",
        })
    out.reverse()
    return out


def agents(rd: Path) -> list[dict[str, Any]]:
    """Per-proposer summary card data, ordered p00..p09 (model + effort + direction)."""
    rows = _db.fetch_models(rd)
    by_agent: dict[str, list[dict]] = {}
    for r in rows:
        by_agent.setdefault(r.get("proposer") or "?", []).append(r)

    alive = _db.run_is_alive(rd)
    inflight = _db.inflight_proposers(rd)
    history = _db.direction_history(rd)

    pids = _db.list_proposers(rd) or sorted(p for p in by_agent if p.startswith("p"))
    out: list[dict[str, Any]] = []
    for pid in pids:
        mine = by_agent.get(pid, [])
        scored = [r["test_corr"] for r in mine
                  if r.get("status") == "ok" and r.get("test_corr") is not None]
        model, effort = proposer_model_effort(rd, pid)
        status, last = _db.agent_activity(rd, pid, alive, inflight)
        direction = _db.proposer_direction(rd, pid)
        out.append({
            "id": pid,
            "model": model,
            "effort": effort,
            "status": status,
            "models_count": len(mine),
            "best_score": round(max(scored), 4) if scored else None,
            "direction": direction,
            "prev_direction": _db.previous_direction(history.get(pid, []), direction),
            "last_activity": last,
        })
    return out


def knowledge(rd: Path) -> list[dict[str, Any]]:
    """Each proposer's BEST scored model: code + unpacked stats, sorted by score desc."""
    rows = _scored_rows(rd)
    best_by: dict[str, dict] = {}
    for r in rows:
        pid = r.get("proposer") or "?"
        if pid not in best_by or r["test_corr"] > best_by[pid]["test_corr"]:
            best_by[pid] = r

    me = {a["id"]: a for a in agents(rd)}
    out: list[dict[str, Any]] = []
    for pid, r in best_by.items():
        notes = parse_notes(r.get("experiment_notes"))
        a = me.get(pid, {})
        out.append({
            "agent": pid,
            "model_id": r["id"],
            "title": r.get("idea") or r["id"],
            "insight": notes["insight"],
            "description": notes["description"],
            "n_params": r.get("n_params"),
            "seconds": _as_float(r.get("seconds")),
            "test_corr": round(r["test_corr"], 4),
            "roi_corr": (round(_as_float(r.get("roi_corr")), 4)
                         if _as_float(r.get("roi_corr")) is not None else None),
            "model": a.get("model", ""),
            "effort": a.get("effort", ""),
            "created_at": r.get("created_at") or "",
            "source": _db.read_model_source(rd, r["id"]),
        })
    out.sort(key=lambda d: d["test_corr"], reverse=True)
    return out
