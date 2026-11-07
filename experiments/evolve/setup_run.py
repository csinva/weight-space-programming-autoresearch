"""
Create a fresh run folder for an autoresearch experiment.

Usage:
    uv run experiments/evolve/setup_run.py <tag>

Result:
    Creates  experiments/runs/<tag>/  containing:
      src/                                            → symlinks to evolve/*
      interpretable_transformer.py                    → fresh local copy
      results/                                        → empty dir
      interpretable_transformers_lib/                 → empty dir

The agent then `cd`s into the run folder and works only there:
no git branch, no commits — every change is local to that folder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


SYMLINK_NAMES = ["src"]
COPY_NAMES = ["interpretable_transformer.py", "results", "program.md"]


def setup_run(tag: str, repo_root: str | None = None, runs_subdir: str = "runs") -> str:
    evolve_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_root is None:
        repo_root = os.path.dirname(evolve_dir)

    runs_dir = os.path.join(repo_root, runs_subdir)
    run_dir = os.path.join(runs_dir, f"{tag}")

    if os.path.exists(run_dir):
        raise FileExistsError(f"Run folder already exists: {run_dir}")

    os.makedirs(run_dir)

    for name in SYMLINK_NAMES:
        src_path = os.path.join(evolve_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        dst_path = os.path.join(run_dir, name)
        rel = os.path.relpath(src_path, run_dir)
        os.symlink(rel, dst_path)

    for name in COPY_NAMES:
        src_path = os.path.join(evolve_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        elif os.path.isdir(src_path):
            dst_path = os.path.join(run_dir, name)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, os.path.join(run_dir, name))

    os.makedirs(os.path.join(run_dir, "interpretable_transformers_lib"))

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="Run tag, e.g. 'may15-run1'.")
    args = parser.parse_args()

    try:
        run_dir = setup_run(args.tag)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Created run folder: {run_dir}")
    print()
    print("Next steps:")
    print(f"  cd {run_dir}")
    print(f"  uv run interpretable_transformer.py  # one experiment iteration")
