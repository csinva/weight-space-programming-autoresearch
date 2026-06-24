"""Async Starlette handlers that expose per-proposer agent logs to the CORAL dashboard."""

from __future__ import annotations

import os
import re
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

from . import _db

# Only parse the last N lines of very large logs to keep responses small.
_MAX_LINES = 2000

_SESSION_RE = re.compile(
    r"\[proposer\].*session start\s+(\S+)", re.IGNORECASE
)


def _file_meta(path: Path, index: int) -> dict:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "index": index,
        "size_bytes": st.st_size,
        "modified": st.st_mtime,
    }


def _parse_log(text: str) -> tuple[str, list[dict]]:
    """Return (session_timestamp, entries_list)."""
    lines = text.splitlines()
    if len(lines) > _MAX_LINES:
        lines = lines[-_MAX_LINES:]

    session_ts = ""
    entries: list[dict] = []

    def _is_continuation(line: str) -> bool:
        return bool(line) and (
            line[0] in (" ", "\t") or line.startswith("│") or line.startswith("└")
        )

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        # Session header
        if line.startswith("[proposer]"):
            if not session_ts:
                m = _SESSION_RE.search(line)
                if m:
                    session_ts = m.group(1)
            entries.append({"type": "system", "content": line, "details": {}, "timestamp": ""})
            i += 1
            continue

        # Tool call
        if line.startswith("●"):
            content = line[1:].strip()
            i += 1
            body_lines: list[str] = []
            while i < len(lines) and _is_continuation(lines[i]):
                body_lines.append(lines[i])
                i += 1
            details = {"body": "\n".join(body_lines)} if body_lines else {}
            entries.append({"type": "tool_call", "content": content, "details": details, "timestamp": ""})
            continue

        # Continuation line appearing without a preceding tool_call (shouldn't happen, but be safe)
        if _is_continuation(line):
            i += 1
            continue

        # Prose: collect consecutive non-special, non-blank lines
        prose: list[str] = []
        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                break
            if ln.startswith("●") or ln.startswith("[proposer]"):
                break
            if _is_continuation(ln):
                break
            prose.append(ln)
            i += 1
        if prose:
            entries.append({"type": "text", "content": "\n".join(prose), "details": {}, "timestamp": ""})

    return session_ts, entries


# ---------------------------------------------------------------------------

async def get_logs_list(request: Request) -> JSONResponse:
    run_dir: Path | None = request.app.state.run_dir
    if not run_dir:
        return JSONResponse({})

    proposers_dir = run_dir / "proposers"
    if not proposers_dir.exists():
        return JSONResponse({})

    result: dict = {}
    for p_dir in sorted(proposers_dir.iterdir()):
        if not p_dir.is_dir():
            continue
        log_file = p_dir / "proposer.log"
        if not log_file.exists():
            continue
        result[p_dir.name] = [_file_meta(log_file, 0)]

    return JSONResponse(result)


async def get_logs(request: Request) -> JSONResponse:
    agent_id: str = request.path_params["agent_id"]
    run_dir: Path | None = request.app.state.run_dir

    empty = {"agent_id": agent_id, "log_files": [], "turns": []}

    if not run_dir:
        return JSONResponse(empty)

    log_file = run_dir / "proposers" / agent_id / "proposer.log"
    if not log_file.exists():
        return JSONResponse(empty)

    text = log_file.read_text(errors="replace")
    session_ts, entries = _parse_log(text)

    meta = _file_meta(log_file, 0)
    turn = {
        "index": 0,
        "entries": entries,
        "usage": {},
        "timestamp": session_ts,
    }

    return JSONResponse({
        "agent_id": agent_id,
        "log_files": [meta],
        "turns": [turn],
    })
