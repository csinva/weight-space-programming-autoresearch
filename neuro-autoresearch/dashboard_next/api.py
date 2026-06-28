"""Starlette JSON handlers for ``dashboard_next``.

Thin wrappers over :mod:`dashboard_next._data` (the new shapes) and a few reused handlers
from the original ``dashboard`` package (runs list / switch / logs / SSE). Read-only.
"""
from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from dashboard import _db
from . import _data


def _rd(request: Request):
    rd = getattr(getattr(request.app, "state", None), "run_dir", None)
    if rd and (rd / "search.db").exists():
        return rd
    return None


def _runs_payload(rd) -> dict:
    from collections import defaultdict
    cur_task, cur_run = _db.split_run_name(rd) if rd else ("", "")
    task_runs: dict[str, list] = defaultdict(list)
    for p in _db.list_run_dirs():
        t, _ = _db.split_run_name(p)
        task_runs[t].append(p)
    tasks = []
    for t, dirs in sorted(task_runs.items()):
        dirs_sorted = sorted(dirs, reverse=True)
        newest = dirs_sorted[0]
        runs = []
        for d in dirs_sorted:
            _, ts = _db.split_run_name(d)
            try:
                n = len(_db.fetch_models(d))
            except Exception:
                n = 0
            runs.append({
                "run": ts,
                "status": "running" if _db.run_is_alive(d) else "stopped",
                "attempts": n,
                "is_latest": d == newest,
            })
        tasks.append({"slug": t, "runs": runs})
    return {"current": {"task": cur_task, "run": cur_run}, "tasks": tasks}


async def get_meta(request: Request) -> JSONResponse:
    rd = _rd(request)
    if rd is None:
        return JSONResponse({"task": "", "run": "", "status": "stopped", "alive": False,
                             "total": 0, "scored": 0, "best_score": None, "best_index": None,
                             "runs": _runs_payload(None)})
    task, run = _db.split_run_name(rd)
    meta = _data.overview_meta(rd)
    return JSONResponse({
        "task": task, "run": run,
        "status": _data.top_status(rd),
        "alive": _db.run_is_alive(rd),
        "total": meta["total"], "scored": meta["scored"],
        "best_score": meta["best_score"], "best_title": meta["best_title"],
        "best_index": meta["best_index"],
        "runs": _runs_payload(rd),
    })


async def get_attempts(request: Request) -> JSONResponse:
    rd = _rd(request)
    return JSONResponse(_data.attempts(rd) if rd else [])


async def get_agents(request: Request) -> JSONResponse:
    rd = _rd(request)
    return JSONResponse(_data.agents(rd) if rd else [])


async def get_knowledge(request: Request) -> JSONResponse:
    rd = _rd(request)
    return JSONResponse(_data.knowledge(rd) if rd else [])


async def switch_run(request: Request) -> JSONResponse:
    body = await request.json()
    task, run = body.get("task", ""), body.get("run", "")
    target = _db.run_dir(task, run)
    if not (target / "search.db").exists():
        return JSONResponse({"ok": False, "error": "no such run"}, status_code=404)
    async with request.app.state.switch_lock:
        request.app.state.run_dir = target
    return JSONResponse({"ok": True, "task": task, "run": run})
