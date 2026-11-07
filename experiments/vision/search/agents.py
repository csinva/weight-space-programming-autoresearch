"""Minimal filesystem view for provider CLIs; queue access goes through a socket.

Provider network access stays enabled for their APIs. Host files/processes are
hidden; only the workspace, installed runtime, and copied login files are visible.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import hashlib
import json

import numpy as np

from neuro_eval.model import sandbox_command, check_isolation

REPO = Path(__file__).resolve().parents[1]


def _login_files(provider: str, destination: Path) -> None:
    home = Path.home()
    locations = {
        'codex': [(Path(os.environ.get('CODEX_HOME', home / '.codex')) / 'auth.json', '.codex/auth.json')],
        'claude': [],  # Interactive refresh credentials must never be cloned.
        'agy': [(Path(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', home / '.config/gcloud/application_default_credentials.json')),
                 '.config/gcloud/application_default_credentials.json')],
    }
    for source, relative in locations[provider]:
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o600)
    if provider == 'claude':
        # Avoid interactive onboarding; authentication uses an automation token.
        (destination / '.claude.json').write_text('{"hasCompletedOnboarding":true}')


def _cli_mounts(executable: str) -> tuple[str, list[str], str]:
    found = shutil.which(executable)
    if not found:
        raise ValueError(f'Missing provider executable: {executable}')
    path = Path(found).resolve()
    mounts = ['--ro-bind', str(path), str(path)]
    path_dirs = ['/usr/bin', '/bin']
    # npm Codex needs its package and the matching node runtime, not all of HOME.
    if path.suffix == '.js':
        package = path.parent.parent
        mounts = ['--ro-bind', str(package), str(package)]
        node = shutil.which('node')
        if not node:
            raise ValueError('Node-based provider requires node')
        node = str(Path(node).resolve())
        mounts += ['--ro-bind', node, node]
        path_dirs.insert(0, str(Path(node).parent))
        # Optional platform packages contain the native Codex executable.
        optional = package.parent / 'codex-linux-x64'
        if optional.is_dir():
            mounts += ['--ro-bind', str(optional), str(optional)]
    return str(path), mounts, ':'.join(path_dirs)


@contextmanager
def agent_command(agent, argv: list[str], workspace: str, role: str, usage=None):
    """Yield an isolated command while its role-bound broker is alive."""
    check_isolation()
    workspace = Path(workspace).resolve(strict=True)
    run = os.environ.get('LAB_RUN')
    if role in {'proposer', 'coordinator'} and not run:
        raise ValueError(f'{role} requires LAB_RUN')
    if role in {'proposer', 'coordinator'}:
        from search.notes import seed_notes
        seed_notes(workspace)
    executable, cli_mounts, path = _cli_mounts(agent.executable)
    with tempfile.TemporaryDirectory(prefix='neuro-agent-') as temp:
        root = Path(temp)
        home = root / 'home'
        home.mkdir()
        _login_files(agent.provider, home)
        provider_argv = [executable, *argv[1:]]
        if agent.provider == 'claude':
            from search.auth import claude_token
            token = claude_token()
            # Transfer through a private temporary file, never argv or run artifacts.
            token_path = home / '.automation-token'
            token_path.touch(mode=0o600)
            token_path.write_text(token)
            bootstrap = (
                "import os, pathlib, sys; "
                "p = pathlib.Path('/home/agent/.automation-token'); "
                "os.environ['CLAUDE_CODE_OAUTH_TOKEN'] = p.read_text(); "
                "p.unlink(); os.execv(sys.argv[1], sys.argv[1:])"
            )
            provider_argv = [sys.executable, '-c', bootstrap, *provider_argv]
        public = root / 'reference'
        public.mkdir()
        for name in ('neuro_eval/rsa.py', 'neuro_eval/encoding.py', 'neuro_eval/benchmarks.py', 'README.md', 'model.py'):
            source = REPO / name
            if source.is_file():
                shutil.copyfile(source, public / source.name)
        socket_dir = root / 'broker'
        socket_dir.mkdir()
        if role in {'proposer', 'coordinator'}:
            from search.api import Broker
            broker = Broker(run, role, os.environ.get('LAB_PROPOSER'), socket_dir / 'api.sock')
        else:
            broker = nullcontext()
        with broker:
            args = [*sandbox_command(), '--share-net', *cli_mounts,
                    '--bind', str(home), '/home/agent',
                    '--ro-bind', str(public), '/reference',
                    '--bind', str(workspace), str(workspace),
                    '--ro-bind', str(REPO / 'search/api.py'), str(workspace / 'api.py'),
                    '--ro-bind', str(socket_dir), '/broker',
                    '--setenv', 'HOME', '/home/agent', '--setenv', 'PATH', path,
                    '--setenv', 'NEURO_PYTHON', sys.executable,
                    '--setenv', 'NEURO_BROKER_SOCKET', '/broker/api.sock',
                    '--setenv', 'CODEX_HOME', '/home/agent/.codex',
                    '--setenv', 'AGY_ADC_AUTH', 'true',
                    '--setenv', 'GOOGLE_APPLICATION_CREDENTIALS', '/home/agent/.config/gcloud/application_default_credentials.json',
                    '--chdir', str(workspace)]
            if role == 'proposer':
                for name in ('AGENTS.md', 'CLAUDE.md', 'GEMINI.md'):
                    if (workspace / name).is_file():
                        args += ['--ro-bind', str(REPO / 'search/prompts/proposer.md'), str(workspace / name)]
            if run and (Path(run) / 'public').is_dir():
                args += ['--ro-bind', str(Path(run) / 'public'), '/inputs']
            for name in ('resolv.conf', 'hosts', 'nsswitch.conf', 'ssl/certs', 'pki'):
                source = Path('/etc') / name
                if source.exists():
                    args += ['--ro-bind', str(source.resolve()), str(source)]
            # Public protocol facts only; no inherited dataset paths, tokens, or host hooks.
            for name in ('NEURO_DATASET', 'NEURO_REGION', 'NEURO_METRIC', 'NEURO_MAX_PARAMS', 'NEURO_IMSIZE', 'LANG'):
                if name in os.environ:
                    args += ['--setenv', name, os.environ[name]]
            with capture_native_usage(home, agent.provider, usage):
                yield [*args, *provider_argv]


@contextmanager
def capture_native_usage(home, provider, usage):
    """Persist sanitized usage while private provider telemetry still exists."""
    if usage is None or provider not in {'codex', 'claude'}:
        yield
        return
    import threading
    stopped = threading.Event()
    offsets = {}

    def collect():
        directory = home / ('.codex/sessions' if provider == 'codex' else '.claude/projects')
        for path in directory.glob('**/*.jsonl'):
            try:
                with path.open('rb') as stream:
                    stream.seek(offsets.get(path, 0))
                    while line := stream.readline():
                        # Retry an unfinished record when the CLI completes its write.
                        if not line.endswith(b'\n'):
                            break
                        offsets[path] = stream.tell()
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if isinstance(event, dict):
                            if provider == 'codex':
                                usage.observe_codex(event)
                            else:
                                usage.observe(event)
            except FileNotFoundError:
                continue

    def poll():
        while not stopped.wait(0.25):
            collect()

    thread = threading.Thread(target=poll, name='provider-usage', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        collect()


def export_public_inputs(run: str | Path, dataset, *, split=None) -> Path:
    """Write eight examples without exposing responses, metadata or source paths.

    Scaffolding uses deterministic synthetic pixels and never opens neural data.
    A trusted caller with an already loaded search split may instead supply it;
    only its protocol-filtered X_test images are exported. Existing examples are
    reused on resume, without loading data or regenerating their bytes.
    """
    from neuro_eval.datasets import get_dataset
    from neuro_eval.benchmarks import validate_search_dataset

    if isinstance(dataset, str):
        dataset = get_dataset(validate_search_dataset(dataset))
    name = validate_search_dataset(dataset.loader)
    if dataset.stage != "search" or dataset.metric not in {"rsa", "encoding"}:
        raise ValueError("Public examples require a fixed search dataset")
    root = Path(run) / "public"
    image_path, spec_path = root / "images.npy", root / "input_spec.json"
    kind = "synthetic_debug_images" if split is None else "permitted_search_images"
    contract = {"schema_version": 1, "dataset": name, "kind": kind,
                "layout": "flattened_HWC_RGB", "imsize": dataset.imsize,
                "dtype": "float32", "value_range": [0, 1],
                "output": "finite numeric (images, features) with fewer than 10000 features",
                "images_file": "images.npy"}
    if spec_path.exists():
        existing = json.loads(spec_path.read_text())
        if any(existing.get(key) != value for key, value in contract.items()):
            raise ValueError("Public input contract changed; use a fresh run")
        if (not image_path.is_file() or image_path.is_symlink()
                or hashlib.sha256(image_path.read_bytes()).hexdigest() != existing["images_sha256"]):
            raise ValueError("Public debug images changed after export")
        return root
    if split is None:
        images = np.random.default_rng(0).random((8, dataset.imsize ** 2 * 3), dtype=np.float32)
    else:
        values = split[1][:8]
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        images = np.array(values, dtype=np.float32, copy=True)
    if (images.ndim != 2 or images.shape[1] != dataset.imsize ** 2 * 3 or not len(images)
            or not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 1)):
        raise ValueError("Debug images must satisfy the fixed flattened RGB input contract")
    root.mkdir(parents=True, exist_ok=True)
    with image_path.open("xb") as file:
        np.save(file, images, allow_pickle=False)
    contract.update(n_images=len(images), images_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
    with spec_path.open("x") as file:
        json.dump(contract, file, indent=2, allow_nan=False)
        file.write("\n")
    return root

def make_sandbox(path: str, repo_root: str = REPO) -> str:
    """Seed a workspace without replacing an existing candidate or scratch."""
    os.makedirs(path, exist_ok=True)
    # Expose only the bounded socket client in provider workspaces.
    client = os.path.join(repo_root, "search", "api.py")
    api_path = os.path.join(path, "api.py")
    if os.path.islink(api_path):
        os.unlink(api_path)
    if os.path.exists(client):
        shutil.copyfile(client, api_path)
    model_py = os.path.join(path, "model.py")
    if not os.path.exists(model_py):                       # seed the baseline (don't clobber a resumed run)
        shutil.copy2(os.path.join(repo_root, "model.py"), model_py)
    prompt = os.path.join(repo_root, "search", "prompts", "proposer.md")
    for filename in ("AGENTS.md", "GEMINI.md", "CLAUDE.md"):
        target = os.path.join(path, filename)
        if os.path.islink(target):
            os.unlink(target)
        if os.path.isfile(prompt):
            shutil.copyfile(prompt, target)
    os.makedirs(os.path.join(path, "scratch"), exist_ok=True)
    return path


def stop_process(child, grace=2):
    """Reap a session and every remaining tool before its replacement starts."""
    import signal
    import time
    descendants = set()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # Repairs/provider tools can create their own sessions. Capture their
        # ancestry before terminating the leader, then kill them individually.
        parents = {}
        for entry in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = entry.read_text().rsplit(')', 1)[1].split()
                parents[int(entry.parent.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
        pending = {child.pid, *descendants}
        while pending:
            children = {pid for pid, parent in parents.items() if parent in pending} - descendants
            descendants.update(children)
            pending = children
        for pid in descendants:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            time.sleep(grace)
    child.wait()


def supervise(run, dataset, roster, hours, epoch_mins, evaluators=1):
    """Supervise isolated provider sessions and one recoverable evaluator pool."""
    import signal
    import time
    from search import queue
    root = Path(run)
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'coordinator').mkdir(exist_ok=True)
    started = time.monotonic()
    deadline = started + hours * 3600
    stopping = False
    sessions = {}
    from search.costs import ensure_pricing, summarize_run
    ensure_pricing(root)
    summarize_run(root)
    next_usage = time.monotonic() + 5
    disabled = set()
    failures = {}
    retry_at = {}
    launched_at = {}
    next_coordinator = started

    def halt(*_):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, halt) for sig in (signal.SIGTERM, signal.SIGINT)}

    def launch(name, spec=None):
        env = os.environ.copy()
        env.pop('LAB_PROPOSER', None)
        env['LAB_RUN'] = str(root.resolve())
        if name.startswith('eval'):
            cards = os.environ.get('NEURO_EVAL_GPUS', '').split(',')
            if cards[0]:
                env['CUDA_VISIBLE_DEVICES'] = cards[int(name[4:]) % len(cards)]
            argv = [sys.executable, '-u', '-m', 'search.worker', '--dataset', dataset]
            prompt = None
        else:
            proposer = name.startswith('p')
            role = 'proposer' if proposer else 'coordinator'
            workspace = root / 'proposers' / name if proposer else root / 'coordinator'
            effort = 'medium' if proposer else os.environ['NEURO_COORDINATOR_EFFORT']
            if proposer:
                env['LAB_PROPOSER'] = name
            argv = [sys.executable, '-m', 'search.providers', 'run', '--agent', spec,
                    '--effort', effort, '--role', role, '--cwd', str(workspace)]
            remaining = max(0.0, (deadline - time.monotonic()) / 60)
            prompt = (f'This run: dataset={dataset}; region={env["NEURO_REGION"]}; metric={env["NEURO_METRIC"]}; role={name}; '
                      f'remaining session budget at launch={remaining:.1f} minutes. '
                      'The launcher stops this session at its deadline. Optimize only the named region and metric. '
                      'Final NSD/TVSD encoding is unavailable. Use "$NEURO_PYTHON" api.py for feedback.\n\n'
                      + (REPO / 'search/prompts' / f'{role}.md').read_text())
        with (root / 'logs' / f'{name}.jsonl').open('a') as log:
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.PIPE if prompt else subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        sessions[name] = child
        launched_at[name] = time.monotonic()
        if prompt:
            try:
                child.stdin.write(prompt)
                child.stdin.close()
            except BrokenPipeError:
                pass  # Poll the session status; terminal failures disable this slot.

    queue.requeue_orphans(run)
    try:
        for i, spec in enumerate(roster):
            launch(f'p{i:02d}', spec)
        for i in range(evaluators):
            launch(f'eval{i}')
        while not stopping and time.monotonic() < deadline:
            if time.monotonic() >= next_usage:
                summarize_run(root)
                next_usage = time.monotonic() + 5
            for name, child in list(sessions.items()):
                code = child.poll()
                if code is None:
                    continue
                stop_process(child, grace=0.05)
                del sessions[name]
                runtime = time.monotonic() - launched_at.pop(name)
                if name.startswith('eval'):
                    # Stop peers before requeueing: a live claim must never be reset.
                    for peer in [key for key in sessions if key.startswith('eval')]:
                        stop_process(sessions.pop(peer))
                        launched_at.pop(peer)
                    queue.requeue_orphans(run)
                    for i in range(evaluators):
                        launch(f'eval{i}')
                    break
                if code == 78:
                    disabled.add(name)
                    print(f'[search] {name} disabled after terminal provider failure', flush=True)
                elif name.startswith('p'):
                    if code == 0 or runtime >= 300:
                        failures[name] = 0
                    if code != 0:
                        failures[name] = failures.get(name, 0) + 1
                        if failures[name] >= 5:
                            disabled.add(name)
                            print(f'[search] {name} disabled after {failures[name]} consecutive provider failures (exit {code})', flush=True)
                        else:
                            delay = min(60, 2 ** (failures[name] - 1))
                            retry_at[name] = time.monotonic() + delay
                            print(f'[search] {name} failed (exit {code}); retry in {delay}s (failure {failures[name]}/5)', flush=True)
                    elif deadline - time.monotonic() <= 120:
                        print(f'[search] {name} finished; no restart in final two minutes', flush=True)
                    else:
                        launch(name, roster[int(name[1:])])
            if all(f'p{i:02d}' in disabled for i in range(len(roster))):
                return 78
            now = time.monotonic()
            for name, ready_at in list(retry_at.items()):
                if ready_at <= now < deadline:
                    del retry_at[name]
                    launch(name, roster[int(name[1:])])
            if (epoch_mins > 0 and 'coordinator' not in disabled and now >= next_coordinator):
                if 'coordinator' not in sessions:
                    launch('coordinator', os.environ['NEURO_COORDINATOR_MODEL'])
                # Fixed launch slots from run startup. Skip occupied/missed slots;
                # never overlap reviews or replay a backlog after an overrun.
                interval = epoch_mins * 60
                next_coordinator = started + ((now - started) // interval + 1) * interval
            time.sleep(0.25)
        return 0
    finally:
        for child in sessions.values():
            stop_process(child)
        queue.requeue_orphans(run)
        summarize_run(root)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
