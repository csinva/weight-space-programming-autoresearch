"""Standalone comparisons retain the fixed cohorts and restore caller settings."""
import os

import pytest

from neuro_eval import comparison, datasets, encoding
from neuro_eval.benchmarks import default_search_region, protocol_for


def score_row(subject="subject"):
    return {"status": "ok", "test_corr": 0.25, "train_corr": 0.3,
            "roi_corr": {"V1": {"test": 0.2, "train": 0.3}},
            "subject_scores": {subject: 0.25}, "n_params": 20,
            "n_targets": 2, "seconds": 1.0}


@pytest.mark.parametrize("dataset", ["laion", "nsd", "triplen"])
@pytest.mark.parametrize("final", [False, True])
def test_compare_routes_protocol_and_returns_artifact_rows(monkeypatch, dataset, final):
    monkeypatch.setenv("NEURO_SUBJECT", "caller-subject")
    monkeypatch.setenv("NEURO_STAGE", "caller-stage")
    monkeypatch.setenv("NEURO_MAX_PARAMS", "123")
    before = dict(os.environ)
    region = default_search_region(dataset)
    protocol = protocol_for(dataset, region=region, metric="encoding")
    calls = []
    models = {"reference": object(), "generated": object()}

    def check(model, data):
        assert "NEURO_SUBJECT" not in os.environ
        assert os.environ["NEURO_STAGE"] == ("final" if final else "search")
        assert os.environ["NEURO_DATASET"] == (protocol.final_dataset if final else dataset)
        assert data.metric == "encoding"
        assert data.search_region == region
        assert data.cohort == ("heldout" if final else "search")
        assert data.images == ("heldout" if final else "search")
        expected_cap = "1000000000" if model is models["reference"] else "123"
        assert os.environ["NEURO_MAX_PARAMS"] == expected_cap
        calls.append((model, data.subject))
        return score_row(data.subject)

    def search_score(model, data, *, stage):
        assert stage == "search"
        return check(model, data)

    monkeypatch.setattr(encoding, "evaluate_subjects", search_score)
    monkeypatch.setattr(datasets, "load_split", lambda data: object())
    monkeypatch.setattr(comparison, "with_areas", lambda split, data: split)
    monkeypatch.setattr(encoding, "evaluate_on_split", lambda model, split, data: check(model, data))
    monkeypatch.setattr(comparison, "forward_seconds", lambda model: 0.12345)
    rows = comparison.compare_models(models, dataset=dataset, final=final, trusted=("reference",))
    assert dict(os.environ) == before
    assert rows.keys() == models.keys()
    for row in rows.values():
        assert row.keys() == {"status", "test_corr", "areas", "subject_scores", "n_params",
                              "eval_seconds", "forward_seconds_1k"}
        assert row["status"] == "ok"
        assert row["test_corr"] == 0.25
        assert row["areas"] == {"V1": 0.2}
        assert row["forward_seconds_1k"] == 0.123
        if final:
            assert set(row["subject_scores"]) == set(protocol.final_subjects)
    if final:
        assert calls == [(model, subject) for subject in protocol.final_subjects for model in models.values()]
    else:
        assert len(calls) == 2


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("cap", [None, "321"])
def test_compare_restores_environment_after_failure(monkeypatch, final, cap):
    if cap is None:
        monkeypatch.delenv("NEURO_MAX_PARAMS", raising=False)
    else:
        monkeypatch.setenv("NEURO_MAX_PARAMS", cap)
    monkeypatch.setenv("NEURO_SUBJECT", "original")
    before = dict(os.environ)

    def fail(*args, **kwargs):
        assert os.environ["NEURO_MAX_PARAMS"] == "1000000000"
        raise RuntimeError("failed model")

    monkeypatch.setattr(encoding, "evaluate_subjects", fail)
    monkeypatch.setattr(encoding, "evaluate_on_split", fail)
    monkeypatch.setattr(datasets, "load_split", lambda data: object())
    with pytest.raises(RuntimeError, match="failed model"):
        comparison.compare_models({"reference": object()}, final=final, trusted=("reference",))
    assert dict(os.environ) == before


def test_model_loading_retains_static_scan(tmp_path):
    from neuro_eval.model import load_model
    path = tmp_path / "model.py"
    path.write_text("MODEL = lambda images: images\n")
    assert load_model(path)("images") == "images"
    path.write_text("import requests\nMODEL = None\n")
    with pytest.raises(ValueError, match="rejected"):
        load_model(path)
