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
    with pytest.raises(ValueError, match="changed after search"):
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
        assert manifest["selection"]["selection_rule"] == "highest_search_score_then_created_at_then_id"
        assert manifest["evaluation_sha256"]
        assert manifest["protocol"] == json.loads(json.dumps(protocol_for("laion").to_dict()))
        assert dataset.search_region == "ventral" and dataset.search_metric == "rsa"
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
    selected = final.manual_model(source, "triplen", "ventral")
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
    {"NEURO_REGION": "early"},
    {"NEURO_METRIC": "encoding"},
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
    selection = final.manual_model(source, "laion", "ventral")
    with pytest.raises(ValueError, match="outside the search run"):
        final.run_final(selection, tmp_path / "search_runs/laion__2026-10-22/final.json")
    assert not (tmp_path / "search_runs").exists()


def test_model_change_before_freezing_creates_no_final_artifact(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("MODEL = None\n")
    selection = final.manual_model(source, "laion", "ventral")
    source.write_text("MODEL = 1\n")
    output = tmp_path / "final/evaluation.json"
    with pytest.raises(ValueError, match="between selection and freezing"):
        final.run_final(selection, output)
    assert not output.parent.exists()


@pytest.mark.parametrize('region', ['early', 'ventral'])
@pytest.mark.parametrize('metric', ['rsa', 'encoding'])
def test_final_preserves_selected_objective(run, tmp_path, monkeypatch, region, metric):
    from search.run import scaffold_run
    from neuro_eval import datasets, encoding, model

    configured = Path(scaffold_run('laion', '2026-10-22', str(run.parent.parent), 1,
                                   region=region, metric=metric))
    protocol = protocol_for('laion', region=region, metric=metric)
    monkeypatch.setenv('NEURO_REGION', region)
    monkeypatch.setenv('NEURO_METRIC', metric)
    mid = queue.propose(str(configured / 'proposers/p00/model.py'), run=str(configured))
    row = {'status': 'ok', 'test_corr': .25, 'metric': metric, 'stage': 'search',
           'dataset': 'laion', 'search_region': region, 'protocol_version': protocol.version,
           'source_revision': SOURCE_REVISION, 'aggregation': protocol.aggregation,
           'search_image_policy': protocol.image_policy,
           'subjects': list(protocol.search_subjects),
           'subject_scores': {s: .25 for s in protocol.search_subjects}}
    assert queue.record(mid, row, run=str(configured)) == 'ok'
    selected = final.select_model(configured)
    assert selected['search_region'] == region and selected['search_metric'] == metric

    class FakeModel:
        def __init__(self, path):
            self.source = path.read_bytes()

    def evaluate(candidate, dataset, stage):
        assert dataset.search_region == region and dataset.search_metric == metric
        assert dataset.reward_roi == f'{region} visual stream'
        assert dataset.loader == 'nsd' and dataset.metric == 'encoding' and stage == 'final'
        return {'status': 'ok', 'test_corr': .3}

    monkeypatch.setattr(model, 'IsolatedModel', FakeModel)
    monkeypatch.setattr(model, 'scan_source', lambda _: None)
    monkeypatch.setattr(encoding, 'evaluate_subjects', evaluate)
    monkeypatch.setattr(datasets, 'dataset_provenance', lambda _: {})
    output = final.run_final(selected, tmp_path / 'frozen/evaluation.json')
    artifact = json.loads(output.read_text())
    assert artifact['protocol'] == json.loads(json.dumps(protocol.to_dict()))
    assert artifact['selection']['search_region'] == region
    assert artifact['selection']['search_metric'] == metric
    assert set(artifact['data_provenance']) == set(protocol.final_subjects)
    scope = artifact['evaluation_scope']
    assert scope['final_neural_data_usage'] == 'final_only'
    assert scope['image_held_out_from_search'] is False
    assert 'NSD-source stimuli' in scope['limitation']
    assert 'must not be interpreted as image-held-out' in scope['limitation']
    manifest = json.loads((output.parent / 'selection.json').read_text())
    assert manifest['evaluation_scope'] == scope
