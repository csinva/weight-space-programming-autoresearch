#!/usr/bin/env python3
"""Scaffold a search RUN and its per-proposer sandboxes.

A run lives at ``runs/<dataset>__<date>/`` and is the unit of one open-ended search::

    runs/<dataset>__<date>/
        search.db            # the shared brain (created by hub.init)
        models/              # shared immutable model archive (created by hub.init)
        proposers/p00 … pNN/ # one sandbox per proposer (its own model.py + symlinked harness/hub)
        eval/                # the single GPU evaluator's sandbox

Each *sandbox* is a runnable workspace: relative symlinks to the FIXED ``harness/`` package and
``hub.py`` (so ``import harness`` / ``import hub`` resolve), a local editable ``model.py`` seeded
from the repo-root baseline model.py, a ``CLAUDE.md`` (symlink to ``search/proposer.CLAUDE.md``, the
agent's ambient orientation), and a ``scratch/`` for the agent. Nothing is copied that should be
shared -- the DB and the model archive live once at the run root.

    python scripts/setup_run.py <dataset> [--date YYYY-MM-DD] [--n 10]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX_SYMLINKS = ["harness", "hub.py"]          # FIXED code, imported (never copied)


def make_sandbox(path: str, repo_root: str = REPO) -> str:
    """Create one runnable sandbox at ``path``: symlinked harness/hub, a seed ``model.py`` copied
    from the repo-root baseline model.py, and a ``scratch/`` dir. Idempotent (a resumed run keeps
    each proposer's evolved model.py)."""
    os.makedirs(path, exist_ok=True)
    for name in SANDBOX_SYMLINKS:
        src = os.path.join(repo_root, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing source for symlink: {src}")
        link = os.path.join(path, name)
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(os.path.relpath(src, path), link)
    model_py = os.path.join(path, "model.py")
    if not os.path.exists(model_py):                       # seed the baseline (don't clobber a resumed run)
        shutil.copy2(os.path.join(repo_root, "model.py"), model_py)
    # ambient orientation for the proposer agent (Copilot auto-loads CLAUDE.md from the cwd):
    # the hub.py DB interface, the model contract, and the loop. Symlinked so it stays in sync.
    claude_src = os.path.join(repo_root, "search", "proposer.CLAUDE.md")
    claude_link = os.path.join(path, "CLAUDE.md")
    if os.path.exists(claude_src) and not os.path.islink(claude_link) and not os.path.exists(claude_link):
        os.symlink(os.path.relpath(claude_src, path), claude_link)
    os.makedirs(os.path.join(path, "scratch"), exist_ok=True)
    return path


def scaffold_run(dataset: str, date: str | None = None, repo_root: str = REPO,
                 n_proposers: int = 10) -> str:
    """Create (or idempotently complete) ``runs/<dataset>__<date>/`` + N proposer sandboxes + eval.

    Returns the run dir. Safe to re-run: existing sandboxes/DB are left intact (a resumed search).
    """
    date = date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")  # ET-dated run
    run_dir = os.path.join(repo_root, "runs", f"{dataset}__{date}")
    sys.path.insert(0, repo_root)
    import hub
    hub.init(run_dir)                              # search.db + models/
    for i in range(n_proposers):
        make_sandbox(os.path.join(run_dir, "proposers", f"p{i:02d}"), repo_root)
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)   # evaluator scratch (scores via hub.score in-repo)
    return run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="dataset key, e.g. nsd / text / meg / tvsd / nnn")
    p.add_argument("--date", default=None, help="run date (default: today, UTC)")
    p.add_argument("--n", type=int, default=10, help="number of proposer sandboxes")
    a = p.parse_args()
    run_dir = scaffold_run(a.dataset, a.date, n_proposers=a.n)
    print(f"scaffolded run: {run_dir}")
    print(f"  search.db + models/ + proposers/p00..p{a.n - 1:02d}/ + eval/")
