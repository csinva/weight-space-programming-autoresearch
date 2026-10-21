"""Provider contract tests use fake CLIs: no model requests or account charges."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from search.providers import DEFAULT_ROSTER, preflight, resolve

REPO = Path(__file__).resolve().parents[1]


def fake_cli(tmp_path, name, *, incomplete=False):
    executable = tmp_path / name
    help_text = "--model --sandbox --add-dir --cd --json --config --skip-git-repo-check --print --effort --mode --output-format"
    if incomplete:
        help_text = "Open an editor window"
    executable.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
if "--help" in sys.argv:
    print({help_text!r})
elif sys.argv[1:] == ["models"]:
    print("gemini-4-argon-high\\tGemini 4 Argon (High)")
else:
    pathlib.Path(os.environ["CAPTURE"]).write_text(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}))
''')
    executable.chmod(0o755)
    return executable


def test_roster_resolves_requested_models(monkeypatch):
    monkeypatch.delenv("NEURO_CODEX_MODEL", raising=False)
    monkeypatch.delenv("NEURO_AGY_MODEL", raising=False)
    agents = [resolve(spec) for spec in DEFAULT_ROSTER.split()]
    assert [(agent.provider, agent.model) for agent in agents] == [
        ("codex", "gpt-6-astra"), ("agy", "gemini-4-argon-high")]
    assert resolve("agy:argon4:low", "high").effort == "low"
    assert resolve("codex:astra", "max").effort == "xhigh"
    monkeypatch.setenv("NEURO_AGY_MODEL", "gemini-4-argon-medium")
    assert resolve("agy:argon4").model == "gemini-4-argon-medium"


@pytest.mark.parametrize("spec", ["astra", "copilot:astra", "agy:", "codex:astra:invalid", "agy:argon4:low:extra"])
def test_invalid_agent_specs_fail(spec):
    with pytest.raises(ValueError):
        resolve(spec)


@pytest.mark.parametrize("provider,spec", [("codex", "codex:astra"), ("agy", "agy:argon4")])
def test_real_adapter_executes_one_prompt_without_shell(tmp_path, provider, spec):
    executable = fake_cli(tmp_path, provider)
    capture = tmp_path / "capture.json"
    prompt_file = tmp_path / "prompt.txt"
    prompt = 'Check literal $(touch NOT_ALLOWED) and `touch NOT_ALLOWED`\nKeep two lines.'
    prompt_file.write_text(prompt)
    env = {**os.environ, f"NEURO_{provider.upper()}_BIN": str(executable), "CAPTURE": str(capture)}
    result = subprocess.run([sys.executable, str(REPO / "search/providers.py"), "run",
                             "--agent", spec, "--cwd", str(tmp_path), "--write-dir", str(tmp_path),
                             "--prompt-file", str(prompt_file)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["cwd"] == str(tmp_path)
    assert record["argv"][-1] == prompt
    assert not (tmp_path / "NOT_ALLOWED").exists()
    if provider == "codex":
        assert record["argv"][0] == "exec"
        assert "approval_policy=\"never\"" in record["argv"]
        assert "workspace-write" in record["argv"]
    else:
        assert "--print" in record["argv"]
        assert "--mode" in record["argv"]


def test_editor_launcher_cannot_pass_headless_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURO_AGY_BIN", str(fake_cli(tmp_path, "agy", incomplete=True)))
    with pytest.raises(ValueError, match="lacks required headless options"):
        preflight([resolve("agy:argon4")])


def test_missing_argon_model_is_not_silently_substituted(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURO_AGY_BIN", str(fake_cli(tmp_path, "agy")))
    with pytest.raises(ValueError, match="not listed"):
        preflight([resolve("agy:unavailable-model")])


def test_dry_run_does_not_call_provider(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Task")
    env = {**os.environ, "NEURO_CODEX_BIN": "/missing/codex"}
    result = subprocess.run([sys.executable, str(REPO / "search/providers.py"), "run",
                             "--agent", "codex:astra", "--cwd", str(tmp_path),
                             "--prompt-file", str(prompt_file), "--dry-run"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["argv"][0] == "/missing/codex"


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


@pytest.mark.parametrize("provider,spec", [("codex", "codex:astra"), ("agy", "agy:argon4")])
def test_proposer_worker_passes_rsa_protocol_to_provider(tmp_path, provider, spec):
    workspace = tmp_path / "proposers" / "p00"
    workspace.mkdir(parents=True)
    executable = fake_cli(tmp_path, provider)
    capture = tmp_path / "capture.json"
    env = {**os.environ, f"NEURO_{provider.upper()}_BIN": str(executable), "CAPTURE": str(capture),
           "LAB_RUN": str(tmp_path), "NEURO_DATASET": "laion", "NEURO_PROPOSER_MODEL": spec,
           "PYTHON": sys.executable}
    result = subprocess.run(["bash", str(REPO / "search/proposer/worker.sh"), "p00"],
                            cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["cwd"] == str(workspace)
    prompt = record["argv"][-1]
    assert "Reward = RSA" in prompt
    assert "ALL subjects" in prompt
    assert "Final transfer evaluations" in prompt
    assert str(tmp_path) in record["argv"]


def test_launcher_rejects_incomplete_subject_cohort_before_mutation(tmp_path):
    date = "2999-01-02"
    target = REPO / "runs" / f"laion__{date}"
    assert not target.exists()
    env = {**os.environ, "NEURO_SUBJECT": "sub-01"}
    result = subprocess.run(["bash", str(REPO / "search/run.sh"), "laion", "--date", date, "--preflight"],
                            cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires all search subjects" in result.stderr
    assert not target.exists()


def test_missing_source_file_fails_readonly_preflight(tmp_path, monkeypatch):
    from harness.loaders.vision import looped
    from search.preflight import validate_sources
    monkeypatch.setattr(looped, "dataset_source_paths", lambda dataset: {"responses": str(tmp_path / "missing.pkl")})
    with pytest.raises(ValueError, match="missing, empty, or unreadable"):
        validate_sources("laion")
    assert not list(tmp_path.iterdir())


def test_repair_working_directory_does_not_grant_repository_writes(tmp_path):
    workspace = tmp_path / "models"
    workspace.mkdir()
    model = workspace / "candidate.py"
    model.write_text("# fake failed candidate\n")
    log = tmp_path / "failure.log"
    log.write_text("NameError: missing_name\n")
    capture = tmp_path / "capture.json"
    executable = fake_cli(tmp_path, "codex")
    env = {**os.environ, "NEURO_CODEX_BIN": str(executable), "CAPTURE": str(capture), "PYTHON": sys.executable}
    result = subprocess.run(["bash", str(REPO / "search/evaluator/repair.sh"), str(model), str(log),
                             "codex:astra", "low", str(REPO)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["cwd"] == str(workspace)
    assert record["argv"][record["argv"].index("--add-dir") + 1] == str(workspace)
    assert "immutable" in record["argv"][-1]
    assert "NameError: missing_name" in record["argv"][-1]


def test_conductor_uses_fixed_hub_from_run_workspace(tmp_path):
    import signal
    import time
    capture = tmp_path / "capture.json"
    executable = fake_cli(tmp_path, "codex")
    env = {**os.environ, "NEURO_CODEX_BIN": str(executable), "CAPTURE": str(capture),
           "LAB_RUN": str(tmp_path), "NEURO_DATASET": "laion", "NEURO_CONDUCTOR_MODEL": "codex:astra",
           "NEURO_EPOCH_MINS": "60", "PYTHON": sys.executable}
    process = subprocess.Popen(["bash", str(REPO / "search/conductor/loop.sh")],
                               env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not capture.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.02)
        assert capture.exists(), (tmp_path / "conductor.log").read_text()
        record = json.loads(capture.read_text())
        assert record["cwd"] == str(tmp_path)
        assert record["argv"][record["argv"].index("--add-dir") + 1] == str(tmp_path)
        assert f'python "{REPO}/hub.py"' in record["argv"][-1]
        assert "harness/, hub.py, search scripts, datasets, and run env.json are immutable" in record["argv"][-1]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)
