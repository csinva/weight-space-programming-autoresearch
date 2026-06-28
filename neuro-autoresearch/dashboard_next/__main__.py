"""``python -m dashboard_next`` — serve the redesigned dashboard for a search run.

Runs on port 8090 by default so it sits alongside the original ``python -m dashboard``
(8080) for side-by-side comparison.
"""
from __future__ import annotations

import argparse
import socket
from pathlib import Path

import uvicorn

from dashboard import _db
from .app import create_app


def _free_port(host: str, port: int, tries: int = 20) -> int:
    for p in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex((host, p)) != 0:
                return p
    return port


def main() -> None:
    p = argparse.ArgumentParser(description="Redesigned live dashboard over a search run.")
    p.add_argument("--run", type=Path, default=None,
                   help="run dir to serve (e.g. runs/nsd__2026-06-26); default: latest")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()

    run_dir = args.run.resolve() if args.run else _db.default_run_dir()
    if run_dir is None or not (run_dir / "search.db").exists():
        raise SystemExit(
            f"No search.db found (looked in {args.run or _db.RUNS_ROOT}). "
            "Pass --run runs/<task>__<date>."
        )

    task, run = _db.split_run_name(run_dir)
    port = _free_port(args.host, args.port)
    if port != args.port:
        print(f"port {args.port} busy -> using {port}")
    print(f"[next] serving {task}/{run}  ->  http://{args.host}:{port}")
    uvicorn.run(create_app(run_dir), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
