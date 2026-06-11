"""Create a fresh, isolated run folder for an autoresearch experiment.

Usage::

    python scripts/setup_run.py <tag>

Result -- ``<repo_root>/runs/<tag>/`` containing:

  ``harness/``      -> relative symlink to the repo's ``harness/`` (FIXED code)
  ``prompts/``      -> relative symlink to the repo's ``prompts/`` (FIXED code)
  ``model.py``      fresh local copy (EDITABLE: the evolving model)
  ``experiment.py`` fresh local copy (EDITABLE: LOG_SLICES)
  ``results/``      empty dir -- the per-run ``<study>.csv`` lands here
  ``models_lib/``   empty dir -- per-attempt ``model.py`` snapshots

The agent then ``cd``s into the run folder and works only there: editing the
local ``model.py``, snapshotting each attempt into ``models_lib/``, and running
``NEURO_DATASET=<key> python -m harness.run``. The symlinked ``harness`` package
imports, the local ``model.py``/``experiment.py`` import, and the harness appends
to the run-local ``results/<study>.csv`` (its results dir is cwd-relative). No git
branch, no commits -- every change stays local to the folder.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

SYMLINK_NAMES = ["harness", "prompts"]
COPY_NAMES = ["model.py", "experiment.py"]
EMPTY_DIRS = ["results", "models_lib"]


def setup_run(tag: str, repo_root: str | None = None) -> str:
    """Create ``runs/<tag>/`` and return its path. Errors if it already exists."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_root is None:
        repo_root = os.path.dirname(scripts_dir)

    run_dir = os.path.join(repo_root, "runs", tag)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run folder already exists: {run_dir}")

    os.makedirs(run_dir)

    for name in SYMLINK_NAMES:
        src_path = os.path.join(repo_root, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        rel = os.path.relpath(src_path, run_dir)
        os.symlink(rel, os.path.join(run_dir, name))

    for name in COPY_NAMES:
        src_path = os.path.join(repo_root, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        shutil.copy2(src_path, os.path.join(run_dir, name))

    for name in EMPTY_DIRS:
        os.makedirs(os.path.join(run_dir, name))

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tag", help="Run tag, e.g. 'demo-run1'.")
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
    print("  NEURO_DATASET=nsd python -m harness.run  # one evaluation pass")
