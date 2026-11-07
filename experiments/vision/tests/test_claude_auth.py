"""Automation credentials cross the sandbox without cloning interactive login state."""
import os
from pathlib import Path
import subprocess

import pytest

from search import agents, providers, run


def test_configure_preserves_only_supported_claude_auth(monkeypatch):
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'test-automation-token')
    for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'):
        monkeypatch.setenv(name, 'conflicting-value')
    run.configure()
    assert os.environ['CLAUDE_CODE_OAUTH_TOKEN'] == 'test-automation-token'
    assert not any(name in os.environ for name in
                   ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'))


def test_claude_never_copies_interactive_credentials(tmp_path, monkeypatch):
    host = tmp_path / 'host'
    (host / '.claude').mkdir(parents=True)
    saved = host / '.claude/.credentials.json'
    saved.write_text('interactive-refresh-secret')
    monkeypatch.setattr(Path, 'home', lambda: host)
    destination = tmp_path / 'private'
    destination.mkdir()
    agents._login_files('claude', destination)
    assert not (destination / '.claude/.credentials.json').exists()
    assert saved.read_text() == 'interactive-refresh-secret'


def test_automation_token_reaches_only_claude_without_argv_or_artifacts(tmp_path, monkeypatch):
    token = 'dummy-automation-secret'
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', token)
    executable = tmp_path / 'fake-claude'
    executable.write_text('''#!/usr/bin/python3
import os, pathlib
assert os.environ['CLAUDE_CODE_OAUTH_TOKEN'] == 'dummy-automation-secret'
assert not pathlib.Path('/home/agent/.automation-token').exists()
assert not pathlib.Path('/home/agent/.claude/.credentials.json').exists()
print('authenticated')
''')
    executable.chmod(0o755)
    monkeypatch.setenv('NEURO_CLAUDE_BIN', str(executable))
    agent = providers.resolve('claude:opus5.5')
    with agents.agent_command(agent, [str(executable)], str(tmp_path), 'repair') as command:
        assert token not in '\0'.join(command)
        private_home = Path(command[command.index('--bind') + 1])
        assert (private_home / '.automation-token').stat().st_mode & 0o777 == 0o600
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'authenticated'
    assert not private_home.exists()
    assert not any(token in p.read_text() for p in tmp_path.iterdir() if p.is_file() and p != executable)


def test_claude_without_automation_token_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv('CLAUDE_CODE_OAUTH_TOKEN', raising=False)
    monkeypatch.setattr('search.auth.token_path', lambda: tmp_path / 'missing-token')
    monkeypatch.setattr(agents, '_cli_mounts', lambda _: ('/usr/bin/true', [], '/usr/bin'))
    with pytest.raises(ValueError, match='claude setup-token'):
        with agents.agent_command(providers.resolve('claude:opus5.5'), ['claude'], str(tmp_path), 'repair'):
            pytest.fail('launched without token')


def test_other_provider_does_not_receive_claude_token(tmp_path, monkeypatch):
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'dummy-automation-secret')
    executable = tmp_path / 'fake-codex'
    executable.write_text('''#!/usr/bin/python3
import os, pathlib
assert 'CLAUDE_CODE_OAUTH_TOKEN' not in os.environ
assert not pathlib.Path('/home/agent/.automation-token').exists()
''')
    executable.chmod(0o755)
    monkeypatch.setenv('NEURO_CODEX_BIN', str(executable))
    with agents.agent_command(providers.resolve('codex:test'), [str(executable)], str(tmp_path), 'repair') as command:
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_saved_token_permissions_loading_and_environment_precedence(tmp_path, monkeypatch):
    from search import auth
    path = tmp_path / 'config/token'
    monkeypatch.setattr(auth, 'token_path', lambda: path)
    monkeypatch.delenv('CLAUDE_CODE_OAUTH_TOKEN', raising=False)
    assert auth.save_token('test-saved-token') == path
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert auth.claude_token() == 'test-saved-token'
    auth.save_token('replacement-token')
    assert auth.claude_token() == 'replacement-token'
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'environment-token')
    assert auth.claude_token() == 'environment-token'
    monkeypatch.delenv('CLAUDE_CODE_OAUTH_TOKEN')
    path.chmod(0o644)
    with pytest.raises(ValueError, match='mode 600'):
        auth.claude_token()


def test_saved_token_reaches_claude_sandbox(tmp_path, monkeypatch):
    from search import auth
    path = tmp_path / 'config/token'
    monkeypatch.setattr(auth, 'token_path', lambda: path)
    monkeypatch.delenv('CLAUDE_CODE_OAUTH_TOKEN', raising=False)
    auth.save_token('saved-test-token')
    executable = tmp_path / 'fake-claude'
    executable.write_text("#!/usr/bin/python3\nimport os\nassert os.environ['CLAUDE_CODE_OAUTH_TOKEN'] == 'saved-test-token'\n")
    executable.chmod(0o755)
    monkeypatch.setenv('NEURO_CLAUDE_BIN', str(executable))
    with agents.agent_command(providers.resolve('claude:opus5.5'), [str(executable)], str(tmp_path), 'repair') as command:
        assert 'saved-test-token' not in str(command)
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    assert auth.claude_token() == 'saved-test-token'
