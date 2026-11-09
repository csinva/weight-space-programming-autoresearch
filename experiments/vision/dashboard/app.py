"""Starlette app wiring the live dashboard over a search run's ``search.db``.

Mirrors CORAL's ``coral/web/app.py`` but the data source is our SQLite ``search.db``
instead of a ``.coral/`` directory. The same prebuilt React frontend (vendored under
``static/``) is served unchanged; we only swap the backend it talks to.

Run it with::

    python -m dashboard --run artifacts/search/laion__2026-10-22   # serve one run
    python -m dashboard                              # serve the latest run

The current run can be switched live from the UI (the run selector); everything is
read-only, so it is safe to run alongside an active search.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import _db
from .api import (
    get_agent_attempts,
    get_attempt_detail,
    get_attempts,
    get_config,
    get_dag,
    get_leaderboard,
    get_notes,
    get_runs,
    get_skill_detail,
    get_skills,
    get_status,
    get_steering,
    post_steer,
    switch_run,
)
from .events import DBWatcher, sse_endpoint
from .logs import get_logs, get_logs_list

STATIC_DIR = Path(__file__).parent / "static"


def create_app(run_dir: Path | None = None) -> Starlette:
    """Build the dashboard app, serving ``run_dir`` (default: the latest run)."""
    run_dir = Path(run_dir).resolve() if run_dir else _db.default_run_dir()

    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.run_dir = run_dir
        app.state.switch_lock = asyncio.Lock()
        app.state.watcher = DBWatcher(lambda: app.state.run_dir)
        app.state._watcher_task = asyncio.create_task(app.state.watcher.run())
        try:
            yield
        finally:
            app.state.watcher.stop()
            app.state._watcher_task.cancel()
            try:
                await app.state._watcher_task
            except asyncio.CancelledError:
                pass

    async def spa_fallback(request: Request) -> Response:
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return Response("Dashboard bundle missing under dashboard/static/.", status_code=404)

    routes = [
        Route("/api/config", get_config),
        Route("/api/attempts", get_attempts),
        Route("/api/leaderboard", get_leaderboard),
        Route("/api/attempts/agent/{id}", get_agent_attempts),
        Route("/api/attempts/{hash}", get_attempt_detail),
        Route("/api/dag", get_dag),
        Route("/api/notes", get_notes),
        Route("/api/skills", get_skills),
        Route("/api/skills/{name}", get_skill_detail),
        Route("/api/logs", get_logs_list),
        Route("/api/logs/{agent_id}", get_logs),
        Route("/api/status", get_status),
        Route("/api/steer", get_steering),
        Route("/api/steer", post_steer, methods=["POST"]),
        Route("/api/runs", get_runs),
        Route("/api/runs/switch", switch_run, methods=["POST"]),
        Route("/api/events", sse_endpoint),
    ]

    if (STATIC_DIR / "assets").exists():
        routes.append(Mount("/assets", app=StaticFiles(directory=STATIC_DIR / "assets"), name="assets"))

    routes.append(Route("/{path:path}", spa_fallback))

    middleware = [
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ]
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
