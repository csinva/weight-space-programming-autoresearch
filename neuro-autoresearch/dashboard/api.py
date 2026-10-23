"""Starlette async endpoint handlers for the CORAL neural-search dashboard.

Each handler reads the current run's SQLite DB via ``_db`` helpers and returns
JSON shaped for the React frontend.  No handler ever writes to the DB.
"""
from __future__ import annotations

from collections import defaultdict
import json

from starlette.requests import Request
from starlette.responses import JSONResponse

from . import _db


# --------------------------------------------------------------------------- helpers

def _rd(request: Request):
    """Return the current run dir, or None if not set / no DB."""
    rd = getattr(getattr(request.app, "state", None), "run_dir", None)
    if rd and (rd / "search.db").exists():
        return rd
    return None


def _safe_attempts(rd) -> list[dict]:
    if rd is None:
        return []
    try:
        return _db.attempts_for(rd)
    except Exception:
        return []


# --------------------------------------------------------------------------- endpoints

async def get_config(request: Request) -> JSONResponse:
    rd = _rd(request)
    task_name = _db.split_run_name(rd)[0] if rd else "unknown"
    attempts = _safe_attempts(rd)
    agent_count = len({a["agent_id"] for a in attempts})
    # Read the persisted contract, never the monitoring process's environment or
    # today's protocol defaults: older runs used a two-region RSA objective.
    env = {}
    if rd:
        try:
            value = json.loads((rd / "env.json").read_text())
            if isinstance(value, dict):
                env = value
        except (OSError, ValueError):
            pass
    protocol = env.get("protocol")
    protocol = protocol if isinstance(protocol, dict) else {}
    metric = env.get("NEURO_METRIC", protocol.get("search_metric", "rsa"))
    metric = metric if metric in ("rsa", "encoding") else "unknown"
    region = env.get("NEURO_REGION", protocol.get("search_region"))
    region = region if region in ("early", "ventral") else None
    target = f"{region} visual cortex" if region else "early and ventral visual cortex"
    measure = "RSA" if metric == "rsa" else metric
    return JSONResponse({
        "task": {
            "name": task_name,
            "description": f"Discover computational principles of {target} through {measure} model search",
            "region": region,
        },
        "grader": {"type": metric, "direction": "maximize"},
        "agents": {"count": agent_count, "model": "", "max_turns": 0, "reflect_every": 0},
    })


async def get_attempts(request: Request) -> JSONResponse:
    rd = _rd(request)
    return JSONResponse(_safe_attempts(rd))


async def get_leaderboard(request: Request) -> JSONResponse:
    rd = _rd(request)
    try:
        top = int(request.query_params.get("top", 20))
    except (ValueError, TypeError):
        top = 20
    scored = [a for a in _safe_attempts(rd) if a["score"] is not None]
    scored.sort(key=lambda a: a["score"], reverse=True)
    return JSONResponse(scored[:top])


async def get_attempt_detail(request: Request) -> JSONResponse:
    rd = _rd(request)
    h = request.path_params["hash"]
    for attempt in _safe_attempts(rd):
        if attempt["commit_hash"] == h:
            result = dict(attempt)
            src_path = rd / "models" / f"{h}.py"
            if rd and src_path.exists():
                try:
                    source = src_path.read_text(errors="replace")
                    result["feedback"] = result["feedback"] + f"\n\n--- model.py ---\n{source}"
                except OSError:
                    pass
            return JSONResponse(result)
    return JSONResponse({"error": "not found"}, status_code=404)


async def get_agent_attempts(request: Request) -> JSONResponse:
    rd = _rd(request)
    agent_id = request.path_params["id"]
    filtered = [a for a in _safe_attempts(rd) if a["agent_id"] == agent_id]
    return JSONResponse(filtered)


async def get_dag(request: Request) -> JSONResponse:
    rd = _rd(request)
    attempts = _safe_attempts(rd)
    scored = [a for a in attempts if a["score"] is not None]
    best_hash = max(scored, key=lambda a: a["score"])["commit_hash"] if scored else None
    # cap at 300 most-recent scored nodes
    nodes_src = scored[-300:] if len(scored) > 300 else scored
    nodes = [
        {
            "id": a["commit_hash"],
            "parent": None,
            "is_root": True,
            "agent_id": a["agent_id"],
            "score": a["score"],
            "status": a["status"],
            "title": a["title"],
            "timestamp": a["timestamp"],
            "is_best": a["commit_hash"] == best_hash,
        }
        for a in nodes_src
    ]
    return JSONResponse({"nodes": nodes, "edges": []})


async def get_status(request: Request) -> JSONResponse:
    rd = _rd(request)
    attempts = _safe_attempts(rd)
    alive = _db.run_is_alive(rd) if rd else False

    scored = [a for a in attempts if a["score"] is not None]
    crashed = [a for a in attempts if a["status"] == "crashed"]
    best = max(scored, key=lambda a: a["score"]) if scored else None

    # per-agent grouping
    by_agent: dict[str, list] = defaultdict(list)
    for a in attempts:
        by_agent[a["agent_id"]].append(a)

    inflight = _db.inflight_proposers(rd) if rd else set()
    agent_statuses = []
    for aid, agent_attempts in sorted(by_agent.items()):
        agent_scored = [a["score"] for a in agent_attempts if a["score"] is not None]
        status, last_activity = _db.agent_activity(rd, aid, alive, inflight)
        direction = _db.proposer_direction(rd, aid) if rd else ""
        agent_statuses.append({
            "agent_id": aid,
            "status": status,
            "sessions": 1,
            "last_activity": last_activity,
            "attempts": len(agent_attempts),
            "best_score": max(agent_scored) if agent_scored else None,
            "direction": direction,
            "prev_direction": "",
        })

    return JSONResponse({
        "manager_alive": alive,
        "manager_pid": None,
        "eval_count": len(scored),
        "total_attempts": len(attempts),
        "scored_attempts": len(scored),
        "crashed_attempts": len(crashed),
        "best_score": best["score"] if best else None,
        "best_title": best["title"] if best else None,
        "agents": agent_statuses,
    })


async def get_runs(request: Request) -> JSONResponse:
    rd = getattr(getattr(request.app, "state", None), "run_dir", None)
    current_task, current_run = _db.split_run_name(rd) if rd else ("", "")

    # group all run dirs by task
    task_runs: dict[str, list[Path]] = defaultdict(list)
    for p in _db.list_run_dirs():
        task, _ = _db.split_run_name(p)
        task_runs[task].append(p)

    tasks = []
    for task, dirs in sorted(task_runs.items()):
        dirs_sorted = sorted(dirs, reverse=True)  # newest first by name
        newest = dirs_sorted[0]
        runs = []
        for d in dirs_sorted:
            _, run_ts = _db.split_run_name(d)
            try:
                attempts = len(_db.fetch_models(d))
            except Exception:
                attempts = 0
            runs.append({
                "timestamp": run_ts,
                "status": "running" if _db.run_is_alive(d) else "stopped",
                "attempts": attempts,
                "is_latest": d == newest,
            })
        tasks.append({"slug": task, "runs": runs})

    return JSONResponse({
        "current": {"task": current_task, "run": current_run},
        "tasks": tasks,
    })


async def switch_run(request: Request) -> JSONResponse:
    body = await request.json()
    task = body.get("task", "")
    run = body.get("run", "")
    target = _db.run_dir(task, run)
    if not (target / "search.db").exists():
        return JSONResponse({"ok": False, "error": "no such run"}, status_code=404)
    async with request.app.state.switch_lock:
        request.app.state.run_dir = target
    return JSONResponse({"ok": True, "task": task, "run": run})


# --------------------------------------------------------------------------- stubs

async def get_notes(request: Request) -> JSONResponse:
    """Knowledge > Notes: the champion models (top by score), with their source."""
    rd = _rd(request)
    if rd is None:
        return JSONResponse([])
    scored = [a for a in _safe_attempts(rd) if a["score"] is not None]
    scored.sort(key=lambda a: a["score"], reverse=True)
    top = scored[:15]
    top.reverse()  # frontend reverses within a category -> this shows best first
    notes = []
    for i, a in enumerate(top):
        body = a["feedback"]
        src = _db.read_model_source(rd, a["commit_hash"])
        if src:
            body += f"\n\n--- model.py ---\n{src}"
        notes.append({
            "date": (a["timestamp"] or "")[:10],
            "title": a["title"],
            "body": body,
            "creator": a["agent_id"],
            "relative_path": f"models/{a['commit_hash']}.py",
            "category": "experiments",
            "index": i,
        })
    return JSONResponse(notes)


async def get_skills(request: Request) -> JSONResponse:
    """Knowledge > Skills: each proposer's self-chosen search direction."""
    rd = _rd(request)
    if rd is None:
        return JSONResponse([])
    skills = []
    for pid in _db.list_proposers(rd):
        direction = _db.proposer_direction(rd, pid)
        if not direction:
            continue
        skills.append({
            "name": pid,
            "description": direction,
            "creator": pid,
            "created": "",
            "path": "",
        })
    return JSONResponse(skills)


async def get_skill_detail(request: Request) -> JSONResponse:
    return JSONResponse({"content": "", "metadata": {}, "body": "", "files": []})


async def get_steering(request: Request) -> JSONResponse:
    return JSONResponse({"actions": [], "pending_count": 0})


async def post_steer(request: Request) -> JSONResponse:
    return JSONResponse({"action": {"applied": False}})
