"""Pinned launcher roles remain reproducible despite caller environment overrides."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def test_fixed_launcher_pins_roles_and_forwards_run_options(tmp_path):
    search = tmp_path / "search"
    search.mkdir()
    for name in ("run_fixed.sh", "lib.sh", "providers.py"):
        shutil.copyfile(REPO / "search" / name, search / name)
    (search / "run.sh").write_text(f'''#!/bin/bash
exec {sys.executable} - "$@" <<'PY_CAPTURE'
import json, os, sys
print(json.dumps({{"args": sys.argv[1:], "roster": os.environ["NEURO_MODEL_ROSTER"],
                  "conductor": os.environ["NEURO_CONDUCTOR_MODEL"],
                  "conductor_effort": os.environ["NEURO_CONDUCTOR_EFFORT"],
                  "proposer_effort": os.environ["NEURO_EFFORT"], "agy_adc_auth": os.environ["AGY_ADC_AUTH"],
                  "claude_route_overrides": [name for name in (
                      "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
                      "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                      "ANTHROPIC_BASE_URL") if name in os.environ]}}))
PY_CAPTURE
''')
    env = {**os.environ, "NEURO_PYTHON": sys.executable, "NEURO_MODEL_ROSTER": "wrong:model", "NEURO_CONDUCTOR_MODEL": "wrong:model",
           "NEURO_CONDUCTOR_EFFORT": "low", "NEURO_EFFORT": "max", "AGY_ADC_AUTH": "false"}
    claude_overrides = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
                        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                        "ANTHROPIC_BASE_URL")
    env.update({name: "conflicting-test-route" for name in claude_overrides})
    command = ["bash", str(search / "run_fixed.sh"), "triplen", "--hours", "2", "--preflight"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    settings = json.loads(result.stdout.splitlines()[-1])
    assert settings == {
        "args": ["triplen", "--hours", "2", "--preflight", "--fleet", "3", "--effort", "medium"],
        "roster": "codex:gpt-6-astra:medium claude:claude-opus-5-5:medium agy:gemini-4-argon-medium:medium",
        "conductor": "codex:gpt-6-astra:xhigh", "conductor_effort": "xhigh", "proposer_effort": "medium", "agy_adc_auth": "true", "claude_route_overrides": [],
    }
    # The display mode exits before even invoking the delegated launcher.
    (search / "run.sh").write_text("exit 99\n")
    result = subprocess.run([*command, "--show-config"], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert "command:" in result.stdout
    assert "AGY_ADC_AUTH=true" in result.stdout
    for name in claude_overrides:
        assert f"-u {name}" in result.stdout
    assert "conflicting-test-route" not in result.stdout
    assert "--dangerously-bypass-approvals-and-sandbox" in result.stdout
    assert "--dangerously-skip-permissions" in result.stdout
    assert 'model_reasoning_effort="xhigh"' in result.stdout
    assert "--model claude-opus-5-5 --effort medium" in result.stdout
    result = subprocess.run([*command, "--effort=low"], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "fixed by this script" in result.stderr
