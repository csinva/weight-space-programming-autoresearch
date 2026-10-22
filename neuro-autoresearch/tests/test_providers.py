"""Provider contract tests use fake CLIs: no model requests or account charges."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


def fake_cli(tmp_path, name, result=None):
    if result is None:
        result = {"event": "result", "result": {"status": "SUCCESS"}}
    executable = tmp_path / name
    help_text = "--model --sandbox --add-dir --cd --json --config --skip-git-repo-check --print --effort --mode --output-format --dangerously-bypass-approvals-and-sandbox --dangerously-skip-permissions --verbose"
    executable.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
if "--help" in sys.argv:
    print({help_text!r})
elif sys.argv[1:] == ["models"]:
    print("gemini-4-argon-high\\tGemini 4 Argon (High)")
else:
    print({json.dumps(result)!r})
    pathlib.Path("capture.json").write_text(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "python": os.environ.get("NEURO_PYTHON")}}))
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
    env = {**os.environ, f"NEURO_{provider.upper()}_BIN": str(executable), "CAPTURE": str(capture),
           f"NEURO_{provider.upper()}_MODEL": expected_model}
    result = subprocess.run([sys.executable, str(REPO / "search/providers.py"), "run",
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
    else:
        assert "--print" in record["argv"]
        assert "--dangerously-skip-permissions" in record["argv"]
        if provider == "claude":
            assert "--verbose" in record["argv"]
    assert "--sandbox" not in record["argv"]


@pytest.mark.parametrize("dataset", ["nsd", "tvsd"])
def test_launcher_rejects_final_targets_before_creating_run(dataset, tmp_path):
    # Use an unmistakable unused date and confirm no persistent launcher effects.
    date = "2999-01-01"
    target = REPO / "runs" / f"{dataset}__{date}"
    assert not target.exists()
    result = subprocess.run(["bash", str(REPO / "search/run.sh"), dataset, "--date", date, "--preflight"],
                            cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    assert "reserved for final encoding evaluation" in result.stderr
    assert not target.exists()


def test_proposer_worker_passes_rsa_protocol_to_provider(tmp_path, selected_python_env):
    provider, spec = "codex", "codex:astra"
    workspace = tmp_path / "proposers" / "p00"
    workspace.mkdir(parents=True)
    executable = fake_cli(tmp_path, provider)
    capture = workspace / "capture.json"
    env = {**selected_python_env, f"NEURO_{provider.upper()}_BIN": str(executable), "CAPTURE": str(capture),
           "LAB_RUN": str(tmp_path), "NEURO_DATASET": "laion", "NEURO_PROPOSER_MODEL": spec}
    result = subprocess.run(["bash", str(REPO / "search/proposer/worker.sh"), "p00"],
                            cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["cwd"] == str(workspace)
    prompt = record["argv"][-1]
    assert "Reward = RSA" in prompt
    assert "ALL subjects" in prompt
    assert "Final transfer evaluations" in prompt
    assert str(workspace) in record["argv"]
    assert record["python"] == sys.executable
    calls = [json.loads(line) for line in Path(env["TEST_PYTHON_CALLS"]).read_text().splitlines()]
    assert calls[0][:2] == [str(REPO / "search/providers.py"), "run"]


def test_repair_runs_in_candidate_directory(tmp_path, selected_python_env):
    workspace = tmp_path / "models"
    workspace.mkdir()
    model = workspace / "candidate.py"
    model.write_text("# fake failed candidate\n")
    log = tmp_path / "failure.log"
    log.write_text("NameError: missing_name\n")
    capture = workspace / "capture.json"
    executable = fake_cli(tmp_path, "codex")
    env = {**selected_python_env, "NEURO_CODEX_BIN": str(executable), "CAPTURE": str(capture)}
    result = subprocess.run(["bash", str(REPO / "search/evaluator/repair.sh"), str(model), str(log),
                             "codex:astra", "low", str(REPO)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["cwd"] == str(workspace)
    assert "--add-dir" not in record["argv"]
    assert "immutable" in record["argv"][-1]
    assert "NameError: missing_name" in record["argv"][-1]
    assert record["python"] == sys.executable
    calls = [json.loads(line) for line in Path(env["TEST_PYTHON_CALLS"]).read_text().splitlines()]
    assert calls[0][:2] == [str(REPO / "search/providers.py"), "run"]


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_argon_alias_tracks_effort_and_rejects_conflicting_override(monkeypatch, effort):
    from search.providers import resolve
    monkeypatch.delenv("NEURO_AGY_MODEL", raising=False)
    assert resolve("agy:argon4", effort).model == f"gemini-4-argon-{effort}"
    other = "low" if effort != "low" else "high"
    monkeypatch.setenv("NEURO_AGY_MODEL", f"gemini-4-argon-{other}")
    with pytest.raises(ValueError, match="conflicts"):
        resolve("agy:argon4", effort)


@pytest.mark.parametrize("result,expected", [
    ({"status": "SUCCESS", "denied_actions": [{"action": "command", "display_name": "RunCommand"}]}, 78),
    ({"status": "ERROR", "error": "invalid model selection"}, 78),
    ({"status": "ERROR", "error": "temporary service unavailable"}, 1),
    ({"status": "SUCCESS", "usage": {"total_tokens": 13}}, 0),
    ({"status": "INVALID"}, 78),
    ({}, 78),
])
def test_agy_native_terminal_outcome_controls_exit(tmp_path, monkeypatch, capsys, result, expected):
    from search.providers import run_agy
    event = {"event": "result", "result": result}
    executable = fake_cli(tmp_path, "agy", event)
    monkeypatch.setenv("CAPTURE", str(tmp_path / "capture.json"))
    assert run_agy([str(executable), "--print", "fake"], str(tmp_path)) == expected
    assert json.loads(capsys.readouterr().out) == event


def test_agy_wrapper_forwards_termination(tmp_path):
    import signal
    executable = tmp_path / "agy"
    executable.write_text(f"#!{sys.executable}\nimport time\nprint('ready', flush=True)\ntime.sleep(60)\n")
    executable.chmod(0o755)
    code = "from search.providers import run_agy; import sys; sys.exit(run_agy(sys.argv[1:2], sys.argv[2]))"
    child = subprocess.Popen([sys.executable, "-c", code, str(executable), str(tmp_path)],
                             cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGTERM)
        child.communicate(timeout=5)
        assert child.returncode == 128 + signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_provider_os_boundary_and_bounded_submission(tmp_path):
    import hub
    from harness.protocols import protocol_for
    from harness.isolation import check_isolation
    check_isolation()
    run = tmp_path / 'run'
    workspace = run / 'proposers/p00'
    workspace.mkdir(parents=True)
    sentinel = tmp_path / 'final-targets-secret'
    sentinel.write_text('must never be visible')
    protocol = protocol_for('laion')
    (run / 'env.json').write_text(json.dumps({
        'NEURO_DATASET': 'laion', 'NEURO_METRIC': 'rsa', 'NEURO_STAGE': 'search',
        'NEURO_SUBJECT': ','.join(protocol.search_subjects), 'protocol': protocol.to_dict()}))
    hub.init(str(run))
    executable = fake_cli(tmp_path, 'codex')
    text = executable.read_text()
    attack = f'''
    assert not pathlib.Path({str(sentinel)!r}).exists()
    assert not pathlib.Path({str(run / 'search.db')!r}).exists()
    assert not pathlib.Path('/proc/{os.getpid()}/root').exists()
    assert 'NEURO_NSD_ROOT' not in os.environ
    assert 'LAB_RUN' not in os.environ
    import importlib.util
    spec = importlib.util.spec_from_file_location('client', 'hub.py')
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
           'LAB_PROPOSER': 'p00', 'NEURO_DATASET': 'laion', 'NEURO_NSD_ROOT': str(sentinel)}
    result = subprocess.run([sys.executable, str(REPO / 'search/providers.py'), 'run',
                             '--role', 'proposer', '--agent', 'codex:astra', '--cwd', str(workspace)],
                            input='test', env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    mid = json.loads((workspace / 'submitted.json').read_text())
    with hub.connect(str(run)) as connection:
        row = dict(connection.execute('SELECT * FROM models WHERE id=?', (mid,)).fetchone())
    assert row['status'] == 'queued' and row['proposer'] == 'p00'
    assert row['test_corr'] is None
