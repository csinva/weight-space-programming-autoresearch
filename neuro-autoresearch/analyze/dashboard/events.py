"""Server-Sent Events endpoint for the neuro-autoresearch dashboard.

Polls the current run's ``search.db`` every ~2 s and pushes SSE events to
the React frontend when models are added, updated, or scored.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from starlette.requests import Request
from starlette.responses import StreamingResponse

from . import _db


class DBWatcher:
    """Polls search.db and broadcasts SSE events to subscribed queues."""

    def __init__(
        self,
        get_run_dir: Callable[[], Path | None],
        poll_interval: float = 2.0,
        subscribers: list[asyncio.Queue] | None = None,
    ) -> None:
        self.get_run_dir = get_run_dir
        self.poll_interval = poll_interval
        self._subscribers: list[asyncio.Queue] = subscribers if subscribers is not None else []
        self._running = False
        self._state: dict = {}

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def _broadcast(self, event: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def _snapshot(self) -> dict:
        rd = self.get_run_dir()
        if rd is None or not (rd / "search.db").exists():
            return {"count": 0, "scored": 0, "latest_mtime": 0.0, "status_sig": ()}
        try:
            with _db.connect(rd) as conn:
                rows = conn.execute("SELECT status, created_at FROM models").fetchall()
        except Exception:
            return {"count": 0, "scored": 0, "latest_mtime": 0.0, "status_sig": ()}

        count = len(rows)
        scored = sum(1 for r in rows if r[0] == "ok")
        latest_mtime = max((_db.parse_ts(r[1]) for r in rows), default=0.0)
        status_sig = tuple(sorted(Counter(r[0] for r in rows).items()))
        return {
            "count": count,
            "scored": scored,
            "latest_mtime": latest_mtime,
            "status_sig": status_sig,
        }

    async def run(self) -> None:
        self._running = True
        try:
            self._state = self._snapshot()
        except Exception:
            self._state = {"count": 0, "scored": 0, "latest_mtime": 0.0, "status_sig": ()}

        while self._running:
            await asyncio.sleep(self.poll_interval)
            try:
                new = self._snapshot()
                if new["count"] > self._state.get("count", 0):
                    self._broadcast(
                        {
                            "event": "attempt:new",
                            "data": {"count": new["count"], "previous": self._state.get("count", 0)},
                        }
                    )
                if (
                    new["latest_mtime"] > self._state.get("latest_mtime", 0)
                    or new["status_sig"] != self._state.get("status_sig")
                ):
                    self._broadcast(
                        {"event": "attempt:update", "data": {"mtime": new["latest_mtime"]}}
                    )
                if new["scored"] != self._state.get("scored", 0):
                    self._broadcast(
                        {"event": "eval:update", "data": {"count": new["scored"]}}
                    )
                self._state = new
            except Exception:
                pass  # transient DB error — keep the loop alive

    def stop(self) -> None:
        self._running = False


async def sse_endpoint(request: Request) -> StreamingResponse:
    """GET /api/events — streams SSE events to the React frontend."""
    watcher: DBWatcher = request.app.state.watcher
    queue = watcher.subscribe()

    async def event_generator():
        try:
            yield f"event: connected\ndata: {json.dumps({'status': 'ok'})}\n\n"
            heartbeat_interval = 15.0
            last_heartbeat = time.time()
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    etype = event.get("event", "message")
                    data = json.dumps(event.get("data", {}))
                    yield f"event: {etype}\ndata: {data}\n\n"
                except TimeoutError:
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        yield f"event: heartbeat\ndata: {json.dumps({'time': now})}\n\n"
                        last_heartbeat = now
                if await request.is_disconnected():
                    break
        finally:
            watcher.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
