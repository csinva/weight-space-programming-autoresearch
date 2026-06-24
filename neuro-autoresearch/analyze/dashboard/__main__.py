"""``python -m analyze.dashboard`` — serve the live dashboard for a search run."""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from . import _db
from .app import create_app


def main() -> None:
    p = argparse.ArgumentParser(description="Live dashboard over a search run's search.db.")
    p.add_argument("--run", type=Path, default=None,
                   help="run dir to serve (e.g. runs/nsd__2026-06-24); default: the latest run")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    run_dir = args.run.resolve() if args.run else _db.default_run_dir()
    if run_dir is None or not (run_dir / "search.db").exists():
        raise SystemExit(
            f"No search.db found (looked in {args.run or _db.RUNS_ROOT}). "
            "Pass --run runs/<task>__<date>."
        )

    task, run = _db.split_run_name(run_dir)
    print(f"serving {task}/{run}  ->  http://{args.host}:{args.port}")
    uvicorn.run(create_app(run_dir), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
