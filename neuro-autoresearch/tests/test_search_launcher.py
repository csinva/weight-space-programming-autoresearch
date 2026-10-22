"""The default Python launcher resolves the pinned three-provider roster."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from search import run


def test_defaults_and_machine_login(monkeypatch, capsys):
    for name in run.PROVIDER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'):
        monkeypatch.setenv(name, 'conflicting-route')
    assert run.main(['--show-config']) == 0
    out = capsys.readouterr().out
    assert 'p00:' in out and 'p01:' in out and 'p02:' in out and 'p03:' not in out
    assert '--model gpt-6-astra' in out
    assert '--model claude-opus-5-5 --effort medium' in out
    assert '--model gemini-4-argon-medium' in out
    assert 'model_reasoning_effort="xhigh"' in out
    assert '--dangerously-skip-permissions' in out
    assert '--dangerously-bypass-approvals-and-sandbox' in out
    import os
    assert 'ANTHROPIC_API_KEY' not in os.environ
    assert 'ANTHROPIC_AUTH_TOKEN' not in os.environ
    assert 'ANTHROPIC_BASE_URL' not in os.environ


@pytest.mark.parametrize('args', [['--hours', '0'], ['--hours', 'nan'], ['--epoch-mins', '-1'], ['--gpu', 'bad'], ['--date', 'bad']])
def test_bad_configuration_never_scaffolds(args, monkeypatch):
    monkeypatch.setattr(run, 'scaffold_run', lambda *a, **kw: pytest.fail('created run'))
    with pytest.raises(SystemExit) as result:
        run.main(args)
    assert result.value.code == 2


def test_preflight_has_no_persistent_side_effects(monkeypatch):
    from search import providers
    from neuro_eval import model
    monkeypatch.setenv('NEURO_SUBJECT', 'sub-01,sub-02,sub-03,sub-04,sub-05')
    monkeypatch.setattr(run, 'evaluation_environment', lambda d: {'NEURO_SUBJECT': 'sub-01,sub-02,sub-03,sub-04,sub-05'})
    monkeypatch.setattr(run, 'validate_sources', lambda d: {})
    monkeypatch.setattr(model, 'check_isolation', lambda: None)
    observed = []
    monkeypatch.setattr(providers, 'preflight', lambda agents: observed.extend(agents))
    monkeypatch.setattr(run, 'scaffold_run', lambda *a, **kw: pytest.fail('created run'))
    assert run.main(['laion', '--preflight']) == 0
    assert len(observed) == 5


def test_preflight_source_validation_reads_each_file(tmp_path, monkeypatch):
    from neuro_eval import datasets
    source = tmp_path / 'images.h5'
    source.write_bytes(b'data')
    monkeypatch.setattr(datasets, 'dataset_source_paths', lambda dataset: {'images': str(source)})
    assert run.validate_sources('laion') == {'images': str(source)}
    source.write_bytes(b'')
    with pytest.raises(ValueError, match='empty'):
        run.validate_sources('laion')


def test_module_launcher_uses_selected_interpreter(tmp_path):
    poison = tmp_path / 'python'
    poison.write_text('#!/bin/sh\nexit 99\n')
    poison.chmod(0o755)
    result = subprocess.run([sys.executable, '-m', 'search.run', '--show-config'],
                            env={**os.environ, 'PATH': str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'coordinator:' in result.stdout


def test_search_layout_has_no_shell_launchers():
    root = Path(__file__).resolve().parents[1]
    assert not list((root / 'search').rglob('*.sh'))
    assert not (root / 'hub.py').exists()
