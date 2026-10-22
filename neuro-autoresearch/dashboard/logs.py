"""Expose native provider logs from a search run to the dashboard."""
from __future__ import annotations

from collections import deque
import json
import re
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

_MAX_LINES = 2000
_AGENT_ID = re.compile(r"p[0-9]+|coordinator")


def _file_meta(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "index": 0,
            "size_bytes": stat.st_size, "modified": stat.st_mtime}


def _parse_log(lines) -> tuple[str, list[dict]]:
    """Preserve provider events and ordinary errors without assuming one schema."""
    entries, session_ts = [], ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        timestamp = str(event.get("timestamp", "")) if isinstance(event, dict) else ""
        session_ts = session_ts or timestamp
        entries.append({"type": "text", "timestamp": timestamp, "details": {},
                        "content": json.dumps(event, indent=2, ensure_ascii=False) if event is not None else line})
    return session_ts, entries


async def get_logs_list(request: Request) -> JSONResponse:
    run = request.app.state.run_dir
    if not run:
        return JSONResponse({})
    return JSONResponse({p.stem: [_file_meta(p)] for p in sorted((run / "logs").glob("*.jsonl"))
                         if _AGENT_ID.fullmatch(p.stem) and p.is_file()})


async def get_logs(request: Request) -> JSONResponse:
    agent_id = request.path_params["agent_id"]
    run = request.app.state.run_dir
    empty = {"agent_id": agent_id, "log_files": [], "turns": []}
    if not run or not _AGENT_ID.fullmatch(agent_id):
        return JSONResponse(empty)
    path = run / "logs" / f"{agent_id}.jsonl"
    if not path.is_file():
        return JSONResponse(empty)
    with path.open(errors="replace") as log:
        session_ts, entries = _parse_log(deque(log, maxlen=_MAX_LINES))
    return JSONResponse({"agent_id": agent_id, "log_files": [_file_meta(path)],
                         "turns": [{"index": 0, "entries": entries, "usage": {}, "timestamp": session_ts}]})
