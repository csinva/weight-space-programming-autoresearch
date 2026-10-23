#!/usr/bin/env python3
"""Create a versioned search run and writable proposer workspaces.

The host owns search.db, immutable models, private state and evaluator logs.
Provider CLIs see only their workspace and the bounded feedback client.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from neuro_eval.benchmarks import protocol_for, validate_search_dataset, SOURCE_REVISION

from neuro_eval.model import EVAL_ENV_KEYS, evaluation_sha256, runtime_versions
PROVIDER_ENV_KEYS = (
    "NEURO_MODEL_ROSTER", "NEURO_CODEX_MODEL", "NEURO_CLAUDE_MODEL", "NEURO_AGY_MODEL", "NEURO_EFFORT",
    "NEURO_REPAIR_MODEL", "NEURO_REPAIR_EFFORT", "NEURO_COORDINATOR_MODEL",
    "NEURO_COORDINATOR_EFFORT", "AGY_ADC_AUTH",
)


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
    env["evaluation_sha256"] = evaluation_sha256()
    env["runtime_versions"] = runtime_versions()
    from neuro_eval.datasets import get_dataset
    from neuro_eval.datasets import dataset_source_paths
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


def scaffold_run(dataset: str, date: str | None = None, repo_root: str = REPO,
                 n_proposers: int = 3) -> str:
    """Create (or idempotently complete) ``search_runs/<dataset>__<date>/`` + N proposer sandboxes + eval.

    Returns the run dir. Safe to re-run: existing sandboxes/DB are left intact (a resumed search).
    """
    dataset = validate_search_dataset(dataset)
    if n_proposers < 1:
        raise ValueError("n_proposers must be positive")
    date = date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    datetime.strptime(date, "%Y-%m-%d")
    run_dir = os.path.join(repo_root, "search_runs", f"{dataset}__{date}")
    from search.agents import make_sandbox
    sys.path.insert(0, repo_root)
    from search import queue
    write_env_json(run_dir, dataset)               # eval-env snapshot for later re-scoring
    queue.init(run_dir)                              # search.db + models/
    from search.agents import export_public_inputs
    export_public_inputs(run_dir, dataset)
    for i in range(n_proposers):
        make_sandbox(os.path.join(run_dir, "proposers", f"p{i:02d}"), repo_root)
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)   # evaluator scratch
    return run_dir


def validate_sources(dataset: str) -> dict[str, str]:
    from neuro_eval.datasets import get_dataset
    from neuro_eval.datasets import dataset_source_paths
    from neuro_eval.benchmarks import validate_search_dataset

    paths = dataset_source_paths(get_dataset(validate_search_dataset(dataset)))
    for kind, raw_path in paths.items():
        path = Path(raw_path).expanduser()
        if not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
            raise ValueError(f"Search {kind} file missing, empty, or unreadable: {path}")
    return paths


def configure():
    """Load declarative defaults and retain machine login authentication."""
    from dotenv import dotenv_values
    for name, value in dotenv_values(os.path.join(REPO, 'search/config.env')).items():
        if value is not None:
            os.environ.setdefault(name, value)
    for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN',
                 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_BASE_URL'):
        os.environ.pop(name, None)


def main(argv=None):
    import math
    import shlex
    from search.providers import resolve, command, preflight
    configure()
    parser = argparse.ArgumentParser(description='Launch fixed-roster RSA search.')
    parser.add_argument('dataset', nargs='?', default='laion')
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--epoch-mins', type=float, default=30,
                        help='Minutes after a coordinator session finishes before the next (default: 30; 0 disables)')
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    parser.add_argument('--date', default=None)
    parser.add_argument('--preflight', '--dry-run', action='store_true')
    parser.add_argument('--show-config', action='store_true')
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.hours) or args.hours <= 0:
            raise ValueError('--hours must be positive')
        if not math.isfinite(args.epoch_mins) or args.epoch_mins < 0:
            raise ValueError('--epoch-mins must be nonnegative')
        if args.gpu and any(c not in '0123456789,' for c in args.gpu):
            raise ValueError('--gpu must be an index or comma list')
        if args.date:
            datetime.strptime(args.date, '%Y-%m-%d')
        dataset = validate_search_dataset(args.dataset)
        roster = os.environ['NEURO_MODEL_ROSTER'].split()
        if not roster:
            raise ValueError('Empty provider roster')
        evaluators = int(os.environ['NEURO_EVALUATORS'])
        if evaluators < 1:
            raise ValueError('NEURO_EVALUATORS must be positive')
        if os.environ.get('VANILLA_AUTORESEARCH') == '1':
            args.epoch_mins = 0
        roles = [(f'p{i:02d}', spec, 'medium') for i, spec in enumerate(roster)]
        roles += [('repair', os.environ['NEURO_REPAIR_MODEL'], os.environ['NEURO_REPAIR_EFFORT'])]
        if args.epoch_mins:
            roles += [('coordinator', os.environ['NEURO_COORDINATOR_MODEL'], os.environ['NEURO_COORDINATOR_EFFORT'])]
        agents = [resolve(spec, effort) for _, spec, effort in roles]
        if args.show_config:
            for (role, _, _), agent in zip(roles, agents):
                print(f'{role}: {shlex.join(command(agent, "<workspace>", "<role prompt>", []))}')
            return 0
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        environment = evaluation_environment(dataset)
        configure()  # .env loading must not reintroduce conflicting Claude authentication routes.
        os.environ.update({k: str(v) for k, v in environment.items() if k.startswith('NEURO_')})
        from neuro_eval.model import check_isolation
        check_isolation()
        validate_sources(dataset)
        preflight(agents)
        if args.preflight:
            print(f'[search] preflight passed: dataset={dataset} subjects={environment["NEURO_SUBJECT"]}')
            return 0
        run = scaffold_run(dataset, args.date, n_proposers=len(roster))
        os.environ.update(LAB_RUN=run, NEURO_DATASET=dataset, NEURO_FLEET=str(len(roster)))
        print(f'[search] run dir: {run}', flush=True)
        from search.agents import supervise
        return supervise(run, dataset, roster, args.hours, args.epoch_mins, evaluators)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
