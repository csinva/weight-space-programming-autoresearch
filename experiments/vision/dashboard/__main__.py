"""``python -m dashboard`` — serve the live dashboard for a search run."""
from __future__ import annotations

import argparse
import socket
from pathlib import Path

import uvicorn

from . import _db
from .app import create_app


def _free_port(host: str, port: int, tries: int = 20) -> int:
    """Return ``port`` if free, else the next free port (so launches never crash)."""
    for p in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex((host, p)) != 0:  # nothing listening -> free
                return p
    return port


def main() -> None:
    p = argparse.ArgumentParser(description="Live dashboard over a search run's search.db.")
    p.add_argument("--run", type=Path, default=None,
                   help="run dir to serve (e.g. artifacts/search/laion__2026-10-22); default: the latest run")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    run_dir = args.run.resolve() if args.run else _db.default_run_dir()
    if run_dir is None or not (run_dir / "search.db").exists():
        raise SystemExit(
            f"No search.db found (looked in {args.run or _db.RUNS_ROOT}). "
            "Pass --run artifacts/search/<task>__<date>."
        )

    task, run = _db.split_run_name(run_dir)
    port = _free_port(args.host, args.port)
    if port != args.port:
        print(f"port {args.port} busy -> using {port}")
    print(f"serving {task}/{run}  ->  http://{args.host}:{port}")
    uvicorn.run(create_app(run_dir), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
