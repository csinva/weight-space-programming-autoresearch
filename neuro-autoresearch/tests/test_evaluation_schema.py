from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "evaluation"))

from common import (  # noqa: E402
    cohort_roi_scores,
    sha256_file,
    validate_evaluation,
)
import legacy_run as evaluation_run  # noqa: E402


def test_cohort_roi_scores_preserves_subjects_and_equal_weight_means():
    result = cohort_roi_scores(
        {
            "1": {"early": 0.3, "ventral": 0.5},
            "2": {"early": 0.1, "ventral": 0.4},
        },
        ["1", "2"],
    )

    assert result == {
        "subjects": {
            "1": {"early": 0.3, "ventral": 0.5},
            "2": {"early": 0.1, "ventral": 0.4},
        },
        "mean": {"early": 0.2, "ventral": 0.45},
    }


def _document(model_path):
    cohort = {
        "subjects": {"1": 0.4, "2": 0.2},
        "mean": 0.3,
    }
    roi_cohort = {
        "subjects": {
            "1": {"early": 0.5, "ventral": 0.3},
            "2": {"early": 0.3, "ventral": 0.1},
        },
        "mean": {"early": 0.4, "ventral": 0.2},
    }
    return {
        "schema_version": 2,
        "dataset": {
            "id": "nsd",
            "domain": "vision",
            "metric": "noise_normalized_r2",
            "rois": ["early", "ventral"],
            "required_rois": ["early", "ventral"],
        },
        "cohorts": {"search": ["1", "2"], "heldout": ["1", "2"]},
        "models": {
            "champion": {
                "model_sha256": sha256_file(model_path),
                "scores": {"search": deepcopy(cohort), "heldout": deepcopy(cohort)},
                "roi_scores": {
                    "search": deepcopy(roi_cohort),
                    "heldout": deepcopy(roi_cohort),
                },
            }
        },
    }


def test_schema_v2_validates_roi_subjects_and_means(tmp_path):
    model_path = tmp_path / "model.py"
    model_path.write_text("MODEL = None\n")
    document = _document(str(model_path))

    validate_evaluation(document, ["champion"], str(model_path))

    document["models"]["champion"]["roi_scores"]["search"]["mean"]["early"] = 0.9
    with pytest.raises(ValueError, match="roi_scores.search.mean"):
        validate_evaluation(document, ["champion"], str(model_path))


def test_schema_v2_rejects_missing_required_roi(tmp_path):
    model_path = tmp_path / "model.py"
    model_path.write_text("MODEL = None\n")
    document = _document(str(model_path))
    del document["models"]["champion"]["roi_scores"]["heldout"]["subjects"]["2"]["ventral"]

    with pytest.raises(ValueError, match="missing required ROIs"):
        validate_evaluation(document, ["champion"], str(model_path))


def test_runner_writes_per_subject_roi_scores(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    champion_path = repo / "runs" / "best" / "nsd" / "model.py"
    champion_path.parent.mkdir(parents=True)
    champion_path.write_text("MODEL = None\n")
    baseline_path = repo / "model.py"
    baseline_path.write_text("MODEL = None\n")
    output = tmp_path / "evaluation.json"

    config = {
        "domain": "vision",
        "metric": "noise_normalized_r2",
        "search": ["1", "2"],
        "heldout": ["5", "7"],
        "rois": ["early", "ventral"],
    }

    def fake_score_subject(dataset, subject, champ_path, lm_cache):
        value = float(subject) / 10
        early, ventral = {
            "1": (0.2, 0.3), "2": (0.3, 0.4), "5": (0.6, 0.7), "7": (0.8, 0.9)
        }[str(subject)]
        return {
            "champion": {
                "test_corr": value,
                "roi_corr": {"early": early, "ventral": ventral},
            },
            "random_projection": {
                "test_corr": value / 2,
                "roi_corr": {"early": value / 3, "ventral": value / 4},
            },
        }

    monkeypatch.setattr(evaluation_run, "REPO", str(repo))
    monkeypatch.setattr(evaluation_run, "BASELINE_MODEL", str(baseline_path))
    monkeypatch.setattr(evaluation_run, "VISION_BASELINES", [])
    monkeypatch.setattr(evaluation_run, "evaluation_for", lambda dataset: config)
    monkeypatch.setattr(evaluation_run, "score_subject", fake_score_subject)
    monkeypatch.setattr(evaluation_run, "load_model", lambda path: type("M", (), {"n_params": 1})())
    monkeypatch.setattr(evaluation_run, "git_commit", lambda: "abc123")
    monkeypatch.setattr(evaluation_run, "model_provenance", lambda path: {
        "dataset": "nsd",
        "subjects": "1,2",
        "test_corr": 0.15,
        "n_params": 1,
        "source_run": "nsd__test",
        "model_id": "model1",
        "idea": "test",
    })

    document = evaluation_run.main(["--dataset", "nsd", "--output", str(output)])

    assert document["schema_version"] == 2
    assert document["dataset"]["required_rois"] == ["early", "ventral"]
    assert document["models"]["champion"]["roi_scores"]["search"] == {
        "subjects": {
            "1": {"early": 0.2, "ventral": 0.3},
            "2": {"early": 0.3, "ventral": 0.4},
        },
        "mean": {"early": 0.25, "ventral": 0.35},
    }
    assert output.exists()
