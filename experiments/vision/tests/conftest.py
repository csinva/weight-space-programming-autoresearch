"""Synthetic search runs shared by search and final-evaluation workflows."""
from pathlib import Path
import shutil

import pytest

from neuro_eval.benchmarks import SOURCE_REVISION, protocol_for
from search import queue
from search.run import EVAL_ENV_KEYS, PROVIDER_ENV_KEYS, scaffold_run

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def legacy_contract_environment(monkeypatch):
    """Existing fixtures exercise the ventral/RSA contract explicitly.

    Production defaults (pooled early + ventral encoding) are covered by tests
    that clear these variables themselves.
    """
    monkeypatch.setenv("NEURO_REGION", "ventral")
    monkeypatch.setenv("NEURO_METRIC", "rsa")


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    for key in (*EVAL_ENV_KEYS, *PROVIDER_ENV_KEYS, "NEURO_DATASET", "NEURO_REGION", "NEURO_METRIC", "NEURO_STAGE", "LAB_RUN", "LAB_PROPOSER", "VANILLA_AUTORESEARCH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEURO_REGION", "ventral")
    monkeypatch.setenv("NEURO_METRIC", "rsa")
    from neuro_eval import datasets
    responses, stimuli = tmp_path / "responses.pkl", tmp_path / "stimuli.h5"
    responses.write_bytes(b"fixture responses")
    stimuli.write_bytes(b"fixture images")
    monkeypatch.setattr(datasets, "dataset_source_paths", lambda dataset: {
        "responses": str(responses), "stimuli": str(stimuli)})
    return responses


@pytest.fixture
def run(tmp_path, clean_environment):
    root = tmp_path / "repository"
    root.mkdir(parents=True)
    shutil.copyfile(REPO / "model.py", root / "model.py")
    return Path(scaffold_run("laion", "2026-10-21", str(root), 2))


@pytest.fixture
def search_result():
    protocol = protocol_for("laion")
    return {"status": "ok", "test_corr": 0.2, "metric": "rsa", "stage": "search",
            "protocol_version": protocol.version, "dataset": "laion",
            "search_region": "ventral", "aggregation": protocol.aggregation,
            "search_image_policy": protocol.image_policy,
            "source_revision": SOURCE_REVISION, "subjects": list(protocol.search_subjects),
            "subject_scores": {subject: 0.2 for subject in protocol.search_subjects},
            "rsa_diagnostics": {"schema_version": 1, "primary_rsa": 0.2}}


@pytest.fixture
def score_model(search_result):
    def score(run, value=0.2):
        model_id = queue.propose(str(run / "proposers/p00/model.py"), run=str(run))
        result = {**search_result, "test_corr": value,
                  "subject_scores": {subject: value for subject in search_result["subjects"]}}
        assert queue.record(model_id, result, run=str(run)) == "ok"
        return model_id
    return score
