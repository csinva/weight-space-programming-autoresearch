"""Starlette app for ``dashboard_next`` — the redesigned dashboard.

Serves a self-contained Preact SPA from ``static/`` and the JSON API in :mod:`api`.
Reuses the original ``dashboard`` package's SSE watcher and per-proposer log handlers
verbatim, so the only new server code is the two redesigned shapes (agents / knowledge).
Runs independently of the original ``dashboard`` (different default port) — the old one
stays fully functional until the redesign is signed off.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from dashboard import _db
from dashboard.events import DBWatcher, sse_endpoint
from dashboard.logs import get_logs, get_logs_list
from .api import get_agents, get_attempts, get_knowledge, get_meta, switch_run

STATIC_DIR = Path(__file__).parent / "static"


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Tell browsers never to cache our assets, so a reload always loads the latest
    JS/CSS. The dashboard is local + read-only, so there's nothing to gain from caching
    and a stale ES-module graph is a real footgun while the UI is iterated on."""

    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp


def create_app(run_dir: Path | None = None) -> Starlette:
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
        return Response("dashboard_next bundle missing under dashboard_next/static/.",
                        status_code=404)

    routes = [
        Route("/api/meta", get_meta),
        Route("/api/attempts", get_attempts),
        Route("/api/agents", get_agents),
        Route("/api/knowledge", get_knowledge),
        Route("/api/logs", get_logs_list),
        Route("/api/logs/{agent_id}", get_logs),
        Route("/api/runs/switch", switch_run, methods=["POST"]),
        Route("/api/events", sse_endpoint),
    ]
    if (STATIC_DIR / "assets").exists():
        routes.append(Mount("/assets", app=StaticFiles(directory=STATIC_DIR / "assets"), name="assets"))
    routes.append(Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"))
    routes.append(Route("/{path:path}", spa_fallback))

    middleware = [
        Middleware(NoCacheMiddleware),
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
    ]
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
