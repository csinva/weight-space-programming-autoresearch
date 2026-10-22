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

from harness.isolation import sandbox_command, check_isolation

REPO = Path(__file__).resolve().parents[1]


def _login_files(provider: str, destination: Path) -> None:
    home = Path.home()
    locations = {
        'codex': [(Path(os.environ.get('CODEX_HOME', home / '.codex')) / 'auth.json', '.codex/auth.json')],
        'claude': [(home / '.claude/.credentials.json', '.claude/.credentials.json')],
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
        # Avoid interactive onboarding; authentication still comes from the saved login.
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
def agent_command(agent, argv: list[str], workspace: str, role: str, *, reference=None, input_spec=None):
    """Yield an isolated command while its role-bound broker is alive."""
    check_isolation()
    workspace = Path(workspace).resolve(strict=True)
    run = os.environ.get('LAB_RUN')
    if role in {'proposer', 'conductor', 'simplifier'} and not run:
        raise ValueError(f'{role} requires LAB_RUN')
    executable, cli_mounts, path = _cli_mounts(agent.executable)
    with tempfile.TemporaryDirectory(prefix='neuro-agent-') as temp:
        root = Path(temp)
        home = root / 'home'
        home.mkdir()
        _login_files(agent.provider, home)
        public = root / 'reference'
        public.mkdir()
        for name in ('harness/metrics.py', 'harness/protocols.py', 'docs/protocol.md', 'docs/access-feedback.md', 'model.py'):
            source = REPO / name
            if source.is_file():
                shutil.copyfile(source, public / source.name)
        socket_dir = root / 'broker'
        socket_dir.mkdir()
        if role in {'proposer', 'conductor', 'simplifier'}:
            from search.broker import Broker
            broker = Broker(run, role, os.environ.get('LAB_PROPOSER'), socket_dir / 'api.sock',
                            reference=reference, input_spec=input_spec)
        else:
            broker = nullcontext()
        with broker:
            args = [*sandbox_command(), '--share-net', *cli_mounts,
                    '--bind', str(home), '/home/agent',
                    '--ro-bind', str(public), '/reference',
                    '--bind', str(workspace), str(workspace),
                    '--ro-bind', str(REPO / 'search/client.py'), str(workspace / 'hub.py'),
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
                        args += ['--ro-bind', str(REPO / 'search/proposer/prompt.md'), str(workspace / name)]
            # Preserve exact simplification references even though the workspace is writable.
            for source in (reference, input_spec):
                if source:
                    source = Path(source).resolve(strict=True)
                    snapshot = root / source.name
                    snapshot.write_bytes(source.read_bytes())
                    args += ['--ro-bind', str(snapshot), str(source)]
            if run and (Path(run) / 'public').is_dir():
                args += ['--ro-bind', str(Path(run) / 'public'), '/inputs']
            for name in ('resolv.conf', 'hosts', 'nsswitch.conf', 'ssl/certs', 'pki'):
                source = Path('/etc') / name
                if source.exists():
                    args += ['--ro-bind', str(source.resolve()), str(source)]
            # Public protocol facts only; no inherited dataset paths, tokens, or host hooks.
            for name in ('NEURO_DATASET', 'NEURO_MAX_PARAMS', 'NEURO_IMSIZE', 'LANG'):
                if name in os.environ:
                    args += ['--setenv', name, os.environ[name]]
            yield [*args, executable, *argv[1:]]


def run_command(argv: list[str], cwd: str) -> int:
    """Forward termination while keeping the broker's context alive."""
    import signal
    with subprocess.Popen(argv, cwd=cwd) as child:
        previous = {sig: signal.signal(sig, lambda signum, _frame: child.send_signal(signum))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = child.wait()
            return code if code >= 0 else 128 - code
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
