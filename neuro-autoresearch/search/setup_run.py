#!/usr/bin/env python3
"""Create a versioned search run and writable proposer workspaces.

The host owns search.db, immutable models, private state and evaluator logs.
Provider CLIs see only their workspace and the bounded feedback client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from harness.protocols import protocol_for, validate_search_dataset, SOURCE_REVISION
SANDBOX_SYMLINKS = []          # FIXED code, imported (never copied)

# The eval-env knobs that change what the reward SCORES (subject set + readout + reliability cuts).
# Persisted into env.json at scaffold time so a later approximate simplification reproduces the
# EXACT evaluation this run used, instead of guessing the defaults.
EVAL_ENV_KEYS = (
    "NEURO_SUBJECT", "NEURO_IMSIZE", "NEURO_MAX_PARAMS", "NEURO_DATA_DIR",
    "LAION_DATA_DIR", "LAION_FMRI_DIR", "TRIPLEN_DATA_DIR", "NSD_ROOT",
    "NSD_STIMULI_HDF5_PATH", "NSD_DATA_DIR", "NSD_DATA_FILE", "NSD_BETAS_DIR",
    "TVSD_DATA_DIR", "TVSD_STIMULI_HDF5_PATH", "LAION_NCSNR_MIN", "NSD_NCSNR_MIN",
    "NNN_MIN_RELIABILITY", "NNN_UNIT_TYPE", "TRIPLEN_MIN_RELIABILITY", "TVSD_ORACLE_MIN",
)
PROVIDER_ENV_KEYS = (
    "NEURO_MODEL_ROSTER", "NEURO_CODEX_MODEL", "NEURO_CLAUDE_MODEL", "NEURO_AGY_MODEL", "NEURO_EFFORT",
    "NEURO_REPAIR_MODEL", "NEURO_REPAIR_EFFORT", "NEURO_CONDUCTOR_MODEL",
    "NEURO_CONDUCTOR_EFFORT", "AGY_ADC_AUTH",
)


def harness_sha256() -> str:
    """Fingerprint fixed scientific code so a resumed queue cannot change its reward."""
    from pathlib import Path
    root = Path(REPO) / "harness"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_versions() -> dict:
    """Track numerical dependencies as part of the reproducible run contract."""
    from importlib.metadata import version
    import platform
    return {"python": platform.python_version(), **{name: version(name) for name in (
        "torch", "numpy", "scipy", "h5py", "pillow", "nibabel", "xarray", "pandas")}}


def evaluation_environment(dataset: str) -> dict:
    """Resolve the full search contract before creating or opening a run."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    dataset = validate_search_dataset(dataset)
    protocol = protocol_for(dataset)
    subjects = tuple(s.strip() for s in os.environ.get("NEURO_SUBJECT", "").split(",") if s.strip())
    if subjects and subjects != protocol.search_subjects:
        raise ValueError(f"{dataset} requires all search subjects: {','.join(protocol.search_subjects)}")
    if os.environ.get("NEURO_METRIC", "rsa") != "rsa" or os.environ.get("NEURO_STAGE", "search") != "search":
        raise ValueError("Search runs require NEURO_METRIC=rsa and NEURO_STAGE=search")
    env = {k: os.environ[k] for k in EVAL_ENV_KEYS if os.environ.get(k)}
    env.update(NEURO_DATASET=dataset, NEURO_SUBJECT=",".join(protocol.search_subjects),
               NEURO_METRIC="rsa", NEURO_STAGE="search",
               NEURO_IMSIZE=os.environ.get("NEURO_IMSIZE", "128"),
               NEURO_MAX_PARAMS=os.environ.get("NEURO_MAX_PARAMS", "100000"))
    # JSON canonicalization changes dataclass tuples to lists, matching subsequent disk reads.
    env["protocol"] = json.loads(json.dumps(protocol.to_dict()))
    env["source_revision"] = SOURCE_REVISION
    env["harness_sha256"] = harness_sha256()
    env["runtime_versions"] = runtime_versions()
    from pathlib import Path
    from harness.datasets import get_dataset
    from harness.loaders.vision.looped import dataset_source_paths
    env["data_files"] = {}
    for role, value in dataset_source_paths(get_dataset(dataset, protocol.search_subjects[0])).items():
        path = Path(value).expanduser().resolve()
        identity = {"path": str(path)}
        if path.exists():
            stat = path.stat()
            identity.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        else:
            identity["missing"] = True
        env["data_files"][role] = identity
    return env


def write_env_json(run_dir: str, dataset: str) -> str:
    """Write the run contract, or reject any scientific change on resume."""
    path = os.path.join(run_dir, "env.json")
    env = evaluation_environment(dataset)
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f)
        changes = [k for k in set(env) | (set(previous) - {"providers"}) if previous.get(k) != env.get(k)]
        if changes:
            raise ValueError("Cannot resume with a changed evaluation contract: " + ", ".join(sorted(changes))
                             + ". Use a new run date; historical encoding scores cannot mix with RSA.")
        return path
    if os.path.exists(os.path.join(run_dir, "search.db")):
        raise ValueError("Existing run has no versioned env.json; start a new run date")
    env["providers"] = {k: os.environ[k] for k in PROVIDER_ENV_KEYS if os.environ.get(k)}
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "x") as f:
        json.dump(env, f, indent=2)
    return path


def make_sandbox(path: str, repo_root: str = REPO) -> str:
    """Seed a workspace without replacing an existing candidate or scratch."""
    os.makedirs(path, exist_ok=True)
    for name in SANDBOX_SYMLINKS:
        src = os.path.join(repo_root, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing source for symlink: {src}")
        link = os.path.join(path, name)
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(os.path.relpath(src, path), link)
    # The agent-facing hub is a small client, never the trusted SQLite module.
    client = os.path.join(repo_root, "search", "client.py")
    hub_path = os.path.join(path, "hub.py")
    if os.path.islink(hub_path):
        os.unlink(hub_path)
    if os.path.exists(client):
        shutil.copyfile(client, hub_path)
    model_py = os.path.join(path, "model.py")
    if not os.path.exists(model_py):                       # seed the baseline (don't clobber a resumed run)
        shutil.copy2(os.path.join(repo_root, "model.py"), model_py)
    prompt = os.path.join(repo_root, "search", "proposer", "prompt.md")
    for filename in ("AGENTS.md", "GEMINI.md", "CLAUDE.md"):
        target = os.path.join(path, filename)
        if os.path.islink(target):
            os.unlink(target)
        if os.path.isfile(prompt):
            shutil.copyfile(prompt, target)
    os.makedirs(os.path.join(path, "scratch"), exist_ok=True)
    return path


def scaffold_run(dataset: str, date: str | None = None, repo_root: str = REPO,
                 n_proposers: int = 10) -> str:
    """Create (or idempotently complete) ``runs/<dataset>__<date>/`` + N proposer sandboxes + eval.

    Returns the run dir. Safe to re-run: existing sandboxes/DB are left intact (a resumed search).
    """
    dataset = validate_search_dataset(dataset)
    if n_proposers < 1:
        raise ValueError("n_proposers must be positive")
    date = date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    datetime.strptime(date, "%Y-%m-%d")
    run_dir = os.path.join(repo_root, "runs", f"{dataset}__{date}")
    sys.path.insert(0, repo_root)
    import hub
    write_env_json(run_dir, dataset)               # eval-env snapshot for later re-scoring
    hub.init(run_dir)                              # search.db + models/
    from search.public_inputs import export_public_inputs
    export_public_inputs(run_dir, dataset)
    for i in range(n_proposers):
        make_sandbox(os.path.join(run_dir, "proposers", f"p{i:02d}"), repo_root)
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)   # evaluator scratch (scores via hub.score in-repo)
    return run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="search dataset: laion or triplen (nnn alias accepted)")
    p.add_argument("--date", default=None, help="run date (default: today, ET)")
    p.add_argument("--n", type=int, default=10, help="number of proposer sandboxes")
    a = p.parse_args()
    run_dir = scaffold_run(a.dataset, a.date, n_proposers=a.n)
    print(f"scaffolded run: {run_dir}")
    print(f"  search.db + models/ + proposers/p00..p{a.n - 1:02d}/ + eval/")
