"""Integration checks for reward provenance and the search/final boundary."""
import json
from pathlib import Path
import shutil

import pytest
import torch

import hub
from experiments.evaluation import final
from harness.protocols import protocol_for, SOURCE_REVISION
from search.setup_run import EVAL_ENV_KEYS, PROVIDER_ENV_KEYS, scaffold_run

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    for key in (*EVAL_ENV_KEYS, *PROVIDER_ENV_KEYS, "NEURO_DATASET", "NEURO_METRIC", "NEURO_STAGE", "LAB_RUN", "LAB_PROPOSER"):
        monkeypatch.delenv(key, raising=False)
    from harness.loaders.vision import looped
    responses, stimuli = tmp_path / "responses.pkl", tmp_path / "stimuli.h5"
    responses.write_bytes(b"fixture responses")
    stimuli.write_bytes(b"fixture images")
    monkeypatch.setattr(looped, "dataset_source_paths", lambda dataset: {
        "responses": str(responses), "stimuli": str(stimuli)})
    return responses


@pytest.fixture
def run(tmp_path, clean_environment):
    root = tmp_path / "repository"
    (root / "harness").mkdir(parents=True)
    (root / "hub.py").write_text("")
    (root / "search/proposer").mkdir(parents=True)
    (root / "search/proposer/prompt.md").write_text("Only edit model.py.\n")
    shutil.copyfile(REPO / "model.py", root / "model.py")
    return Path(scaffold_run("laion", "2026-10-21", str(root), 1))


def score_model(run, value=0.2):
    model_id = hub.propose(str(run / "proposers/p00/model.py"), run=str(run))
    result = {"status": "ok", "test_corr": value, "train_corr": value,
              "metric": "rsa", "stage": "search", "protocol_version": protocol_for("laion").version,
              "dataset": "laion", "source_revision": SOURCE_REVISION,
              "subjects": list(protocol_for("laion").search_subjects),
              "subject_scores": {s: value for s in protocol_for("laion").search_subjects}}
    assert hub.record(model_id, result, run=str(run)) == "ok"
    return model_id


def test_scaffold_freezes_protocol_and_agent_instructions(run):
    environment = json.loads((run / "env.json").read_text())
    assert environment["NEURO_METRIC"] == "rsa"
    assert environment["NEURO_SUBJECT"] == "sub-01,sub-03,sub-05,sub-06,sub-07"
    assert environment["protocol"]["final_dataset"] == "nsd"
    assert environment["data_files"]["responses"]["size_bytes"] > 0
    for name in ("AGENTS.md", "GEMINI.md"):
        assert (run / "proposers/p00" / name).is_symlink()


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


def test_final_datasets_cannot_be_scaffolded_for_search(tmp_path, clean_environment):
    for dataset in ("nsd", "tvsd"):
        with pytest.raises(ValueError, match="reserved for final"):
            scaffold_run(dataset, "2026-10-21", str(tmp_path), 1)
    assert not (tmp_path / "runs").exists()


def test_encoding_results_cannot_enter_search_feedback(run):
    model_id = hub.propose(str(run / "proposers/p00/model.py"), run=str(run))
    status = hub.record(model_id, {"status": "ok", "test_corr": 0.9,
                       "metric": "encoding", "stage": "final"}, run=str(run))
    assert status == "error"
    assert hub.champion(str(run)) is None
    with hub.connect(str(run)) as connection:
        row = connection.execute("SELECT test_corr FROM models WHERE id=?", (model_id,)).fetchone()
    assert row[0] is None


def test_champion_selection_checks_hash_and_search_metric(run):
    model_id = score_model(run)
    selected = final.select_model(run)
    assert selected["model_id"] == model_id
    assert selected["search_metric"] == "rsa"
    Path(selected["path"]).write_text("MODEL = None\n")
    with pytest.raises(ValueError, match="changed after RSA"):
        final.select_model(run)


def test_final_evaluation_freezes_model_and_does_not_update_queue(run, tmp_path, monkeypatch):
    model_id = score_model(run)
    selected = final.select_model(run)
    with hub.connect(str(run)) as connection:
        before = dict(connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone())

    def evaluate(selection, frozen):
        assert frozen.read_bytes() == Path(selection["path"]).read_bytes()
        return {"status": "ok", "metric": "encoding", "stage": "final", "test_corr": 0.42}, {}

    monkeypatch.setattr(final, "evaluate_frozen", evaluate)
    output = final.run_final(selected, tmp_path / "final/evaluation.json")
    document = json.loads(output.read_text())
    assert document["dataset"] == "nsd"
    assert document["encoding_score"] == 0.42
    assert document["selection"]["search_score"] == 0.2
    assert Path(document["frozen_model"]).is_file()
    with hub.connect(str(run)) as connection:
        after = dict(connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone())
    assert after == before
    with pytest.raises(FileExistsError):
        final.run_final(selected, output)
    with pytest.raises(ValueError, match="outside the search run"):
        final.run_final(selected, run / "final.json")


def test_final_environment_restores_search_state(monkeypatch):
    monkeypatch.setenv("NEURO_DATASET", "triplen")
    monkeypatch.setenv("NEURO_SUBJECT", "M1,M2,M3,M4,M5")
    with final.final_environment({"search_dataset": "triplen", "environment": {}}):
        import os
        assert os.environ["NEURO_DATASET"] == "tvsd"
        assert os.environ["NEURO_SUBJECT"] == "F,N"
        assert os.environ["NEURO_METRIC"] == "encoding"
    assert os.environ["NEURO_DATASET"] == "triplen"
    assert os.environ["NEURO_SUBJECT"] == "M1,M2,M3,M4,M5"


def test_final_environment_preserves_target_storage_override(monkeypatch):
    import os
    monkeypatch.setenv("NSD_DATA_DIR", "/custom/final/nsd")
    with final.final_environment({"search_dataset": "laion", "environment": {"NEURO_IMSIZE": "128"}}):
        assert os.environ["NSD_DATA_DIR"] == "/custom/final/nsd"


def test_wrong_dataset_or_incomplete_cohort_is_not_a_reward(run):
    model_id = hub.propose(str(run / "proposers/p00/model.py"), run=str(run))
    result = {"status": "ok", "test_corr": 0.9, "metric": "rsa", "stage": "search",
              "protocol_version": protocol_for("laion").version, "source_revision": SOURCE_REVISION,
              "dataset": "triplen", "subjects": ["M1"], "subject_scores": {"M1": 0.9}}
    assert hub.record(model_id, result, run=str(run)) == "error"
    assert hub.champion(str(run)) is None


def test_final_selection_rejects_changed_harness(run, monkeypatch):
    score_model(run)
    monkeypatch.setattr("search.setup_run.harness_sha256", lambda: "different")
    with pytest.raises(ValueError, match="different scientific code"):
        final.select_model(run)


def test_dotenv_is_loaded_before_contract_snapshot(monkeypatch, clean_environment):
    import os
    from search.setup_run import evaluation_environment
    def load(*args, **kwargs):
        os.environ.setdefault("LAION_DATA_DIR", "/configured/laion")
    monkeypatch.setattr("dotenv.load_dotenv", load)
    first = evaluation_environment("laion")
    assert first["LAION_DATA_DIR"] == "/configured/laion"
    assert evaluation_environment("laion") == first


def test_seed_model_meets_cap_and_is_batch_independent():
    from model import Model
    from harness.reward import robust_n_params
    model = Model()
    X = torch.rand(3, 128 * 128 * 3)
    whole = model(X)
    assert torch.equal(whole, torch.cat([model(X[i:i+1]) for i in range(len(X))]))
    assert robust_n_params(model) <= 100_000
    assert torch.equal(whole, Model()(X))
