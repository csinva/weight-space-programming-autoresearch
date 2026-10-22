"""Freeze the RSA winner before final encoding, without feeding search."""
import hashlib
import json
from pathlib import Path

import pytest

import final_eval as final
from neuro_eval.benchmarks import SOURCE_REVISION, protocol_for
from search import queue


def test_champion_selection_checks_hash_and_search_metric(run, score_model):
    score_model(run, value=0.1)
    model_id = score_model(run)
    selected = final.select_model(run)
    assert selected["model_id"] == model_id
    assert selected["search_metric"] == "rsa"
    # A scored snapshot must retain its cohort and provenance through DB storage.
    with queue.connect(str(run)) as connection:
        scored = dict(connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone())
    assert scored["test_corr"] == 0.2
    assert json.loads(scored["subject_scores"]) == {
        subject: 0.2 for subject in protocol_for("laion").search_subjects}
    metadata = json.loads(scored["evaluation_metadata"])
    assert metadata["source_revision"] == SOURCE_REVISION
    assert metadata["metric"] == "rsa" and metadata["stage"] == "search"
    Path(selected["path"]).chmod(0o600)
    Path(selected["path"]).write_text("MODEL = None\n")
    with pytest.raises(ValueError, match="changed after RSA"):
        final.select_model(run)


def test_final_evaluation_freezes_model_and_does_not_update_queue(run, tmp_path, monkeypatch, score_model):
    from neuro_eval import datasets, encoding, model

    # Importing this candidate in the host would fail; evaluation must use isolation.
    (run / "proposers/p00/model.py").write_text("raise RuntimeError('host import')\n")
    score_model(run)
    selected = final.select_model(run)
    with queue.connect(str(run)) as connection:
        before = list(connection.iterdump())
    output = tmp_path / "final/evaluation.json"

    class FakeIsolatedModel:
        def __init__(self, path):
            assert path == output.parent / "model.py"
            self.source = path.read_bytes()

    def evaluate(candidate, dataset, stage):
        assert isinstance(candidate, FakeIsolatedModel)
        assert candidate.source == Path(selected["path"]).read_bytes()
        manifest = json.loads((output.parent / "selection.json").read_text())
        assert manifest["stage"] == "frozen_before_final"
        assert manifest["model_sha256"] == hashlib.sha256(candidate.source).hexdigest()
        assert manifest["selection"]["selection_rule"] == "highest_search_rsa_then_created_at_then_id"
        assert manifest["evaluation_sha256"]
        assert manifest["protocol"] == json.loads(json.dumps(protocol_for("laion").to_dict()))
        assert dataset.loader == "nsd" and dataset.metric == "encoding" and stage == "final"
        return {"status": "ok", "test_corr": 0.42}

    monkeypatch.setattr(model, "IsolatedModel", FakeIsolatedModel)
    monkeypatch.setattr(model, "scan_source", lambda path: None)
    monkeypatch.setattr(encoding, "evaluate_subjects", evaluate)
    monkeypatch.setattr(datasets, "dataset_provenance", lambda dataset: {})
    assert final.run_final(selected, output) == output
    document = json.loads(output.read_text())
    assert document["dataset"] == "nsd"
    assert document["encoding_score"] == 0.42
    assert document["selection"]["search_score"] == 0.2
    assert set(document["data_provenance"]) == set(protocol_for("laion").final_subjects)
    with queue.connect(str(run)) as connection:
        assert list(connection.iterdump()) == before
    with pytest.raises(FileExistsError):
        final.run_final(selected, output)
    with pytest.raises(ValueError, match="outside the search run"):
        final.run_final(selected, run / "final.json")


def test_failed_final_can_retry_only_the_same_frozen_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "REPO", tmp_path)
    source = tmp_path / "candidate.py"
    source.write_text("MODEL = None\n")
    selected = final.manual_model(source, "triplen")
    output = tmp_path / "final_results" / protocol_for("triplen").name / selected["sha256"] / "evaluation.json"

    def fail(selection, frozen):
        assert (frozen.parent / "selection.json").is_file()
        raise RuntimeError("synthetic final failure")

    monkeypatch.setattr(final, "evaluate_frozen", fail)
    with pytest.raises(RuntimeError, match="synthetic final failure"):
        final.run_final(selected)
    manifest = (output.parent / "selection.json").read_bytes()
    assert not output.exists()
    changed = {**selected, "selection_rule": "changed after seeing final feedback"}
    with pytest.raises(FileExistsError, match="different frozen selection"):
        final.run_final(changed)
    assert (output.parent / "selection.json").read_bytes() == manifest

    monkeypatch.setattr(final, "evaluate_frozen", lambda selection, frozen: (
        {"status": "ok", "test_corr": 0.3}, {}))
    assert final.run_final(selected) == output
    assert json.loads(output.read_text())["dataset"] == "tvsd"
    assert (output.parent / "selection.json").read_bytes() == manifest
    with pytest.raises(FileExistsError):
        final.run_final(selected)


@pytest.mark.parametrize("change", [
    {"NEURO_SUBJECT": "sub-01"},
    {"evaluation_sha256": "different evaluator"},
])
def test_final_rejects_changed_run_contract(run, change, monkeypatch, score_model):
    score_model(run)
    path = run / "env.json"
    environment = json.loads(path.read_text())
    environment.update(change)
    path.write_text(json.dumps(environment))
    # Reject the contract before even opening SQLite. Comparing database file
    # bytes is unreliable while prior queue connections can checkpoint the WAL.
    def unexpected_database_access(*args, **kwargs):
        pytest.fail("Invalid run contract must be rejected before database access")

    monkeypatch.setattr(final.sqlite3, "connect", unexpected_database_access)
    with pytest.raises(ValueError):
        final.select_model(run)


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
