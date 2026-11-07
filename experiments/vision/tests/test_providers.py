"""Exercise real provider wrappers and isolation with fake CLIs; no model requests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


def fake_cli(tmp_path, name, result=None):
    if result is None:
        result = {"event": "result", "result": {"status": "SUCCESS"}}
    executable = tmp_path / name
    help_text = "--model --sandbox --add-dir --cd --json --config --skip-git-repo-check --print --effort --mode --output-format --dangerously-bypass-approvals-and-sandbox --dangerously-skip-permissions --verbose --ignore-user-config --disable --include-partial-messages --safe-mode --tools --strict-mcp-config --mcp-config --disable-slash-commands"
    executable.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
if "--help" in sys.argv:
    print({help_text!r})
elif sys.argv[1:] == ["models"]:
    print("gemini-4-argon-high\\tGemini 4 Argon (High)")
else:
    print({json.dumps(result)!r})
    pathlib.Path("capture.json").write_text(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}))
''')
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize("provider,spec", [("codex", "codex:gpt-6-astra"), ("claude", "claude:opus5.5"), ("agy", "agy:argon4")])
def test_real_adapter_executes_one_prompt_without_shell(tmp_path, provider, spec):
    executable = fake_cli(tmp_path, provider)
    capture = tmp_path / "capture.json"
    prompt_file = tmp_path / "prompt.txt"
    prompt = 'Check literal $(touch NOT_ALLOWED) and `touch NOT_ALLOWED`\nKeep two lines.'
    prompt_file.write_text(prompt)
    expected_model = {"codex": "gpt-6-astra", "claude": "claude-opus-5-5", "agy": "gemini-4-argon-high"}[provider]
    env = {**os.environ, f"NEURO_{provider.upper()}_BIN": str(executable),
           f"NEURO_{provider.upper()}_MODEL": expected_model}
    if provider == "claude":
        env["CLAUDE_CODE_OAUTH_TOKEN"] = "fake-automation-token"
    result = subprocess.run([sys.executable, "-m", "search.providers", "run",
                             "--agent", spec, "--cwd", str(tmp_path), "--write-dir", str(tmp_path),
                             "--prompt-file", str(prompt_file)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["cwd"] == str(tmp_path)
    assert record["argv"][record["argv"].index("--model") + 1] == expected_model
    assert record["argv"][-1] == prompt
    assert not (tmp_path / "NOT_ALLOWED").exists()
    if provider == "codex":
        assert record["argv"][0] == "exec"
        assert "--dangerously-bypass-approvals-and-sandbox" in record["argv"]
        disabled = [record["argv"][i + 1] for i, flag in enumerate(record["argv"]) if flag == "--disable"]
        assert {"apps", "plugins", "remote_plugin", "browser_use", "computer_use",
                "image_generation", "multi_agent"} <= set(disabled)
        assert 'web_search="disabled"' in record["argv"]
    else:
        assert "--print" in record["argv"]
        assert "--dangerously-skip-permissions" in record["argv"]
        if provider == "claude":
            assert "--verbose" in record["argv"]
            assert "--safe-mode" in record["argv"]
            assert "--bare" not in record["argv"]
            assert "--strict-mcp-config" in record["argv"]
            assert record["argv"][record["argv"].index("--tools") + 1] == "Bash,Read,Edit,Write,Glob,Grep"
        assert "--disable-slash-commands" in record["argv"]
    assert "--sandbox" not in record["argv"]


@pytest.mark.parametrize("result,expected", [
    ({"status": "SUCCESS", "denied_actions": [{"action": "command", "display_name": "RunCommand"}]}, 78),
    ({"status": "ERROR", "error": "invalid model selection"}, 78),
    ({"status": "ERROR", "error": "temporary service unavailable"}, 1),
    ({}, 78),
])
def test_agy_native_terminal_outcome_controls_exit(tmp_path, capsys, result, expected):
    from search.providers import run_provider
    event = {"event": "result", "result": result}
    executable = fake_cli(tmp_path, "agy", event)
    assert run_provider([str(executable), "--print", "fake"], str(tmp_path)) == expected
    assert json.loads(capsys.readouterr().out) == event


def test_agy_wrapper_forwards_termination(tmp_path):
    import select
    import signal
    executable = tmp_path / "agy"
    executable.write_text(f"#!{sys.executable}\nimport time\nprint('ready', flush=True)\ntime.sleep(60)\n")
    executable.chmod(0o755)
    code = "from search.providers import run_provider; import sys; sys.exit(run_provider(sys.argv[1:2], sys.argv[2]))"
    child = subprocess.Popen([sys.executable, "-c", code, str(executable), str(tmp_path)],
                             cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
    try:
        ready, _, _ = select.select([child.stdout], [], [], 5)
        assert ready, "fake provider did not become ready"
        assert os.read(child.stdout.fileno(), 1024).strip() == b"ready"
        child.send_signal(signal.SIGTERM)
        child.communicate(timeout=5)
        assert child.returncode == 128 + signal.SIGTERM
    finally:
        # Kill the group even if the wrapper exited but left its CLI holding pipes.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.communicate(timeout=5)


def test_provider_os_boundary_and_bounded_submission(tmp_path):
    from search import queue
    from neuro_eval.benchmarks import protocol_for
    from neuro_eval.model import check_isolation
    check_isolation()
    run = tmp_path / 'run'
    workspace = run / 'proposers/p00'
    workspace.mkdir(parents=True)
    sentinel = tmp_path / 'final-targets-secret'
    sentinel.write_text('must never be visible')
    protocol = protocol_for('laion')
    (run / 'env.json').write_text(json.dumps({
        'NEURO_DATASET': 'laion', 'NEURO_REGION': 'ventral', 'NEURO_METRIC': 'rsa', 'NEURO_STAGE': 'search',
        'NEURO_SUBJECT': ','.join(protocol.search_subjects), 'protocol': protocol.to_dict()}))
    queue.init(str(run))
    executable = fake_cli(tmp_path, 'codex')
    text = executable.read_text()
    attack = f'''
    assert not pathlib.Path({str(sentinel)!r}).exists()
    assert not pathlib.Path({str(run / 'search.db')!r}).exists()
    assert not pathlib.Path('/proc/{os.getpid()}/root').exists()
    assert 'NEURO_NSD_ROOT' not in os.environ
    assert 'LAB_RUN' not in os.environ
    import importlib.util
    spec = importlib.util.spec_from_file_location('client', 'api.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.request({{'command': 'my-best'}})
    mid = client.request({{'command': 'propose-model', 'source': 'MODEL = None\\n', 'idea': 'boundary probe'}})
    assert client.request({{'command': 'status', 'id': mid}})['status'] == 'queued'
    try:
        client.request({{'command': 'record', 'id': mid, 'test_corr': 1}})
    except RuntimeError:
        pass
    else:
        raise AssertionError('untrusted score write accepted')
    pathlib.Path('submitted.json').write_text(json.dumps(mid))
'''
    text = text.replace('    pathlib.Path("capture.json")', attack + '    pathlib.Path("capture.json")')
    executable.write_text(text)
    env = {**os.environ, 'NEURO_CODEX_BIN': str(executable), 'LAB_RUN': str(run),
           'LAB_PROPOSER': 'p00', 'NEURO_DATASET': 'laion', 'NEURO_REGION': 'ventral', 'NEURO_METRIC': 'rsa', 'NEURO_NSD_ROOT': str(sentinel)}
    result = subprocess.run([sys.executable, '-m', 'search.providers', 'run',
                             '--role', 'proposer', '--agent', 'codex:astra', '--cwd', str(workspace)],
                            input='test', env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    mid = json.loads((workspace / 'submitted.json').read_text())
    with queue.connect(str(run)) as connection:
        row = dict(connection.execute('SELECT * FROM models WHERE id=?', (mid,)).fetchone())
    assert row['status'] == 'queued' and row['proposer'] == 'p00'
    assert row['test_corr'] is None


def test_native_usage_harvests_before_home_cleanup(tmp_path):
    from search.agents import capture_native_usage

    class Usage:
        def __init__(self):
            self.events = []

        def observe_codex(self, event):
            self.events.append(event)

    usage = Usage()
    directory = tmp_path / '.codex/sessions/2026/10/22'
    directory.mkdir(parents=True)
    path = directory / 'rollout.jsonl'
    event = {'type': 'event_msg', 'payload': {'type': 'token_count',
             'info': {'total_token_usage': {'input_tokens': 12, 'output_tokens': 3}}}}
    with capture_native_usage(tmp_path, 'codex', usage):
        path.write_text(json.dumps(event) + '\n' + '{"unfinished":')
    assert usage.events == [event]


def test_native_claude_usage_harvest(tmp_path):
    from search.agents import capture_native_usage

    class Usage:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    usage = Usage()
    directory = tmp_path / '.claude/projects/project'
    directory.mkdir(parents=True)
    event = {'type': 'assistant', 'message': {'id': 'message-1',
             'usage': {'input_tokens': 12, 'output_tokens': 3}}}
    with capture_native_usage(tmp_path, 'claude', usage):
        (directory / 'session.jsonl').write_text(json.dumps(event) + '\n')
    assert usage.events == [event]


def test_non_agy_usage_relay_preserves_native_events(tmp_path, capsys):
    from search.providers import run_provider

    class Usage:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    event = {'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 3}}
    executable = fake_cli(tmp_path, 'codex', event)
    usage = Usage()
    assert run_provider([str(executable), 'prompt'], str(tmp_path), usage, validate_agy=False) == 0
    assert usage.events == [event]
    assert json.loads(capsys.readouterr().out) == event


@pytest.mark.parametrize("token", [None, "", "   "])
def test_claude_preflight_requires_automation_token(tmp_path, monkeypatch, token):
    monkeypatch.setattr("search.auth.token_path", lambda: tmp_path / "missing-token")
    from search.providers import Agent, preflight

    executable = fake_cli(tmp_path, "claude")
    if token is None:
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", token)
    with pytest.raises(ValueError, match="claude setup-token.*export CLAUDE_CODE_OAUTH_TOKEN"):
        preflight([Agent("claude", "claude-opus-5-5", "high", str(executable))])
    assert not (tmp_path / "capture.json").exists()


@pytest.mark.parametrize("event", [
    {"type": "assistant", "error": "authentication_failed",
     "message": {"content": [{"type": "text", "text": "secret-token"}]}},
    {"type": "result", "subtype": "error_during_execution", "is_error": True,
     "errors": ["API Error: 401 Unauthorized; secret-token"]},
    {"type": "result", "subtype": "success", "is_error": True,
     "result": "Invalid API key · Please run /login; secret-token"},
    {"type": "result", "is_error": True,
     "error": {"type": "authentication_error", "message": "secret-token"}},
    {"type": "result", "is_error": True,
     "errors": ["OAuth token has expired. secret-token"]},
])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_claude_auth_failures_stop_slot_without_echoing_secrets(tmp_path, capsys, event, exit_code):
    from search.providers import run_provider

    executable = fake_cli(tmp_path, "claude", event)
    executable.write_text(executable.read_text() + f"sys.exit({exit_code})\n")
    assert run_provider([str(executable)], str(tmp_path), provider="claude") == 78
    output = capsys.readouterr()
    assert json.loads(output.out) == event
    assert "claude setup-token" in output.err
    assert "CLAUDE_CODE_OAUTH_TOKEN" in output.err
    assert "secret-token" not in output.err


@pytest.mark.parametrize("event,expected", [
    ({"type": "result", "is_error": True, "errors": ["API Error: 529 overloaded"]}, 1),
    ({"type": "result", "subtype": "error_during_execution", "errors": ["Connection timed out"]}, 1),
    ({"type": "assistant", "error": "rate_limit"}, 1),
    ({"type": "assistant", "message": {"content": [{"type": "text", "text": "Invalid API key"}]}}, 0),
    ({"type": "user", "error": "authentication_failed", "message": {"content": "Please run /login"}}, 0),
    ({"type": "tool_result", "error": "authentication_failed"}, 0),
    ({"type": "result", "subtype": "success", "is_error": False, "result": "Please run /login"}, 0),
    ({"type": "result", "is_error": True, "result": "File permission denied"}, 1),
])
def test_claude_classification_ignores_content_and_retries_transient_failures(tmp_path, capsys, event, expected):
    from search.providers import run_provider

    executable = fake_cli(tmp_path, "claude", event)
    assert run_provider([str(executable)], str(tmp_path), provider="claude") == expected
    assert capsys.readouterr().err == ""


def test_claude_native_auth_event_survives_terminal_and_usage_relay(tmp_path, capsys):
    from search.providers import run_provider

    events = [
        {"type": "assistant", "error": "authentication_failed"},
        {"type": "result", "subtype": "success", "is_error": True, "result": "Request failed"},
    ]
    executable = tmp_path / "claude"
    executable.write_text(f"#!{sys.executable}\n" + "\n".join(f"print({json.dumps(event)!r})" for event in events))
    executable.chmod(0o755)
    class Usage:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    usage = Usage()
    assert run_provider([str(executable)], str(tmp_path), usage, provider="claude") == 78
    assert usage.events == events
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == events
