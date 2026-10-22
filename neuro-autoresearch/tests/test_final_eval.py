"""Final artifacts are separate from search orchestration and search storage."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

import final_eval as final


def test_final_imports_no_search_orchestration():
    tree = ast.parse(Path(final.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name == "hub" or name.split(".")[0] == "search" for name in imports)


def test_default_output_uses_full_model_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "REPO", tmp_path)
    source = tmp_path / "candidate.py"
    source.write_text("MODEL = None\n")
    selection = final.manual_model(source, "triplen")

    def evaluate(selection, frozen):
        manifest = json.loads((frozen.parent / "selection.json").read_text())
        assert manifest["stage"] == "frozen_before_final"
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == manifest["model_sha256"]
        return {"status": "ok", "test_corr": 0.3}, {}

    monkeypatch.setattr(final, "evaluate_frozen", evaluate)
    output = final.run_final(selection)
    assert output == tmp_path / "final_results" / final.protocol_for("triplen").name / selection["sha256"] / "evaluation.json"
    assert json.loads(output.read_text())["dataset"] == "tvsd"
    with pytest.raises(FileExistsError):
        final.run_final(selection)


def test_manual_final_cannot_write_into_search_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "REPO", tmp_path)
    source = tmp_path / "model.py"
    source.write_text("MODEL = None\n")
    selection = final.manual_model(source, "laion")
    with pytest.raises(ValueError, match="outside the search run"):
        final.run_final(selection, tmp_path / "search_runs/laion__2026-10-22/final.json")
    assert not (tmp_path / "search_runs").exists()


def test_model_change_before_freezing_creates_no_final_artifact(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("MODEL = None\n")
    selection = final.manual_model(source, "laion")
    source.write_text("MODEL = 1\n")
    output = tmp_path / "final/evaluation.json"
    with pytest.raises(ValueError, match="between selection and freezing"):
        final.run_final(selection, output)
    assert not output.parent.exists()
