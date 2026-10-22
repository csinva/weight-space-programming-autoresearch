"""Integration checks for reward provenance and the search/final boundary."""
import json
from pathlib import Path
import shutil

import pytest

from search import queue
import final_eval as final
from neuro_eval.benchmarks import protocol_for, SOURCE_REVISION
from search.run import EVAL_ENV_KEYS, PROVIDER_ENV_KEYS, scaffold_run

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    for key in (*EVAL_ENV_KEYS, *PROVIDER_ENV_KEYS, "NEURO_DATASET", "NEURO_METRIC", "NEURO_STAGE", "LAB_RUN", "LAB_PROPOSER"):
        monkeypatch.delenv(key, raising=False)
    from neuro_eval import datasets as looped
    responses, stimuli = tmp_path / "responses.pkl", tmp_path / "stimuli.h5"
    responses.write_bytes(b"fixture responses")
    stimuli.write_bytes(b"fixture images")
    monkeypatch.setattr(looped, "dataset_source_paths", lambda dataset: {
        "responses": str(responses), "stimuli": str(stimuli)})
    return responses


@pytest.fixture
def run(tmp_path, clean_environment):
    root = tmp_path / "repository"
    (root / "neuro_eval").mkdir(parents=True)
    (root / "search/proposer").mkdir(parents=True)
    (root / "search/proposer/prompt.md").write_text("Only edit model.py.\n")
    shutil.copyfile(REPO / "model.py", root / "model.py")
    return Path(scaffold_run("laion", "2026-10-21", str(root), 1))


def score_model(run, value=0.2):
    model_id = queue.propose(str(run / "proposers/p00/model.py"), run=str(run))
    result = {"status": "ok", "test_corr": value, "train_corr": value,
              "metric": "rsa", "stage": "search", "protocol_version": protocol_for("laion").version,
              "dataset": "laion", "source_revision": SOURCE_REVISION,
              "subjects": list(protocol_for("laion").search_subjects),
              "subject_scores": {s: value for s in protocol_for("laion").search_subjects}}
    assert queue.record(model_id, result, run=str(run)) == "ok"
    return model_id


def test_resume_rejects_scientific_changes_before_writing(run, monkeypatch):
    before = (run / "env.json").read_bytes()
    monkeypatch.setenv("NEURO_IMSIZE", "256")
    with pytest.raises(ValueError, match="changed evaluation contract"):
        scaffold_run("laion", "2026-10-21", str(run.parents[1]), 1)
    assert (run / "env.json").read_bytes() == before


def test_resume_detects_changed_source_data(run, clean_environment):
    clean_environment.write_bytes(b"different responses")
    with pytest.raises(ValueError, match="data_files"):
        scaffold_run("laion", "2026-10-21", str(run.parents[1]), 1)


def test_encoding_results_cannot_enter_search_feedback(run):
    model_id = queue.propose(str(run / "proposers/p00/model.py"), run=str(run))
    status = queue.record(model_id, {"status": "ok", "test_corr": 0.9,
                       "metric": "encoding", "stage": "final", "dataset": "laion",
                       "protocol_version": protocol_for("laion").version, "source_revision": SOURCE_REVISION,
                       "subjects": list(protocol_for("laion").search_subjects),
                       "subject_scores": {s: 0.9 for s in protocol_for("laion").search_subjects}}, run=str(run))
    assert status == "error"
    assert queue.champion(str(run)) is None
    with queue.connect(str(run)) as connection:
        row = connection.execute("SELECT test_corr FROM models WHERE id=?", (model_id,)).fetchone()
    assert row[0] is None


def test_champion_selection_checks_hash_and_search_metric(run):
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


def test_final_evaluation_freezes_model_and_does_not_update_queue(run, tmp_path, monkeypatch):
    model_id = score_model(run)
    selected = final.select_model(run)
    with queue.connect(str(run)) as connection:
        before = dict(connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone())

    def evaluate(selection, frozen):
        assert frozen.read_bytes() == Path(selection["path"]).read_bytes()
        manifest = json.loads((frozen.parent / "selection.json").read_text())
        assert manifest["model_sha256"] == selection["sha256"]
        assert manifest["selection"]["selection_rule"] == "highest_search_rsa_then_created_at_then_id"
        assert manifest["evaluation_sha256"]
        assert manifest["protocol"] == json.loads(json.dumps(protocol_for("laion").to_dict()))
        return {"status": "ok", "metric": "encoding", "stage": "final", "test_corr": 0.42}, {}

    monkeypatch.setattr(final, "evaluate_frozen", evaluate)
    output = final.run_final(selected, tmp_path / "final/evaluation.json")
    document = json.loads(output.read_text())
    assert document["dataset"] == "nsd"
    assert document["encoding_score"] == 0.42
    assert document["selection"]["search_score"] == 0.2
    assert Path(document["frozen_model"]).is_file()
    with queue.connect(str(run)) as connection:
        after = dict(connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone())
    assert after == before
    with pytest.raises(FileExistsError):
        final.run_final(selected, output)
    with pytest.raises(ValueError, match="outside the search run"):
        final.run_final(selected, run / "final.json")


def test_incomplete_cohort_is_not_a_reward(run):
    model_id = queue.propose(str(run / "proposers/p00/model.py"), run=str(run))
    result = {"status": "ok", "test_corr": 0.9, "metric": "rsa", "stage": "search",
              "protocol_version": protocol_for("laion").version, "source_revision": SOURCE_REVISION,
              "dataset": "laion", "subjects": ["sub-01"], "subject_scores": {"sub-01": 0.9}}
    assert queue.record(model_id, result, run=str(run)) == "error"
    assert queue.champion(str(run)) is None


def test_failed_final_keeps_selection_and_rejects_inconsistent_retry(tmp_path, monkeypatch):
    source = tmp_path / "candidate.py"
    source.write_text("MODEL = None\n")
    selected = final.manual_model(source, "laion")
    output = tmp_path / "final/evaluation.json"
    calls = []

    def fail(selection, frozen):
        calls.append(selection)
        assert (frozen.parent / "selection.json").is_file()
        raise RuntimeError("synthetic final failure")

    monkeypatch.setattr(final, "evaluate_frozen", fail)
    with pytest.raises(RuntimeError, match="synthetic final failure"):
        final.run_final(selected, output)
    manifest = (output.parent / "selection.json").read_bytes()
    with pytest.raises(RuntimeError, match="synthetic final failure"):
        final.run_final(selected, output)
    assert not output.exists()
    changed = {**selected, "selection_rule": "changed after seeing final feedback"}
    with pytest.raises(FileExistsError, match="different frozen selection"):
        final.run_final(changed, output)
    assert len(calls) == 2
    assert (output.parent / "selection.json").read_bytes() == manifest


def test_final_uses_isolated_model_without_importing_candidate(tmp_path, monkeypatch):
    import neuro_eval.model
    import neuro_eval.encoding
    from neuro_eval import datasets as looped

    source = tmp_path / "model.py"
    source.write_text("raise RuntimeError('candidate imported by scorer')\n")
    selection = final.manual_model(source, "laion")
    calls = []

    class FakeIsolatedModel:
        def __init__(self, path):
            self.source = path.read_bytes()
            calls.append(path)

    monkeypatch.setattr(neuro_eval.model, "IsolatedModel", FakeIsolatedModel)
    monkeypatch.setattr(neuro_eval.model, "scan_source", lambda path: None)
    monkeypatch.setattr(neuro_eval.encoding, "evaluate_subjects", lambda model, dataset, stage: {
        "status": "ok", "test_corr": 0.1})
    monkeypatch.setattr(looped, "dataset_provenance", lambda dataset: {})
    result, provenance = final.evaluate_frozen(selection, source)
    assert result["test_corr"] == 0.1
    assert calls == [source]
    assert set(provenance) == set(protocol_for("laion").final_subjects)


@pytest.mark.parametrize("change", [
    {"NEURO_SUBJECT": "sub-01"},
    {"NEURO_METRIC": "encoding"},
    {"NEURO_STAGE": "final"},
    {"source_revision": "different source"},
    {"evaluation_sha256": "different evaluator"},
])
def test_final_rejects_changed_run_contract(run, change):
    score_model(run)
    path = run / "env.json"
    environment = json.loads(path.read_text())
    environment.update(change)
    path.write_text(json.dumps(environment))
    database_before = (run / "search.db").read_bytes()
    with pytest.raises(ValueError):
        final.select_model(run)
    assert (run / "search.db").read_bytes() == database_before
