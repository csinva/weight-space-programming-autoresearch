"""Source agreement, split isolation, and complete-cohort reward contracts."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import rankdata, spearmanr

from neuro_eval.datasets import get_dataset
from neuro_eval.encoding import encoding_scores
from neuro_eval.rsa import rsa_scores
from neuro_eval.benchmarks import protocol_for
from neuro_eval.rsa import evaluate_on_split_safe, evaluate_subjects
from neuro_eval import encoding, rsa


@pytest.fixture(autouse=True)
def clean_evaluation_env(monkeypatch):
    for key in ("NEURO_SUBJECT", "NEURO_STAGE", "NEURO_METRIC", "NEURO_MAX_PARAMS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("neuro_eval.rsa._readout_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("neuro_eval.encoding._readout_device", lambda: torch.device("cpu"))


class Identity:
    name = "identity"
    n_params = 0

    def __call__(self, X):
        return X


def search_split(seed=0):
    rng = np.random.RandomState(seed)
    X = torch.tensor(rng.normal(size=(9, 5)), dtype=torch.float32)
    Y = torch.tensor(rng.normal(size=(9, 7)), dtype=torch.float32)
    return X[:0], X, Y[:0], Y, {"early": np.arange(7) < 2, "ventral": np.arange(7) >= 2}, None


def encoding_fixture(seed=40):
    rng = np.random.RandomState(seed)
    X = torch.tensor(rng.normal(size=(30, 6)), dtype=torch.float32)
    Y = X @ torch.tensor(rng.normal(size=(6, 7)), dtype=torch.float32)
    Y += torch.tensor(rng.normal(size=Y.shape), dtype=torch.float32) * 0.3
    masks = {"early": np.arange(7) < 2, "ventral": np.arange(7) >= 2}
    return X, Y, masks


def test_rsa_matches_numpy_pearson_and_scipy_with_equal_region_weight():
    tied_rdm = torch.tensor([[0., 1., 1., 2.], [1., 0., 2., 3.],
                             [1., 2., 0., 3.], [2., 3., 3., 0.]])
    upper = tied_rdm.numpy()[np.triu_indices(4, 1)]
    np.testing.assert_array_equal(rsa.triu_ranks(tied_rdm), rankdata(upper, method="average"))
    _, X, _, Y, _, _ = search_split()
    masks = {"early": np.arange(7) < 3, "ventral": np.arange(7) >= 3}
    row = rsa_scores(X, Y, masks, torch.device("cpu"))
    model_rdm = 1 - np.corrcoef(X.numpy())
    np.testing.assert_allclose(rsa.rdm_chunked(X, torch.device("cpu")), model_rdm, atol=1e-6)
    upper = np.triu_indices(len(X), 1)
    expected = {}
    for name, mask in masks.items():
        neural_rdm = 1 - np.corrcoef(Y[:, mask].numpy())
        np.testing.assert_allclose(rsa.compute_rdm(Y[:, mask]), neural_rdm, atol=1e-6)
        expected[name] = spearmanr(model_rdm[upper], neural_rdm[upper]).statistic
        assert row["roi_corr"][name]["test"] == pytest.approx(expected[name], abs=1e-12)
    assert row["test_corr"] == pytest.approx(np.mean(list(expected.values())), abs=1e-12)


def test_encoding_matches_frozen_source_and_ignores_heldout_data_during_fit():
    X, Y, masks = encoding_fixture()
    tr, te, dev = np.arange(22), np.arange(22, 30), torch.device("cpu")
    design = encoding.design(X, tr, dev, features="center")
    # Captured from the untouched source kernel at revision
    # 3ee575baeceb1ad784102162a051b7444182fdaa before consolidation.
    direct = np.array([0.9962937831878662, 0.9902385473251343, 0.990955650806427,
                       0.9958555698394775, 0.9480392932891846, 0.9931279420852661,
                       0.9898065328598022])
    actual, alphas, _ = encoding.unit_scores(design, Y, tr, te)
    np.testing.assert_allclose(actual, direct, atol=1e-6)
    row = encoding_scores(X[tr], X[te], Y[tr], Y[te], masks, dev)
    for name, mask in masks.items():
        assert row["roi_corr"][name]["test"] == pytest.approx(direct[mask].mean(), abs=1e-6)
    assert row["test_corr"] == pytest.approx(np.mean([direct[m].mean() for m in masks.values()]), abs=1e-6)
    changed_X, changed_Y = X.clone(), Y.clone()
    changed_X[te] = 1_000 + changed_X[te] * 20
    changed_Y[te] *= -100
    changed_design = encoding.design(changed_X, tr, dev, features="center")
    torch.testing.assert_close(design[tr], changed_design[tr], rtol=0, atol=0)
    _, changed_alphas, _ = encoding.unit_scores(changed_design, changed_Y, tr, te)
    np.testing.assert_array_equal(alphas, changed_alphas)
    changed = encoding_scores(X[tr], changed_X[te], Y[tr], changed_Y[te], masks, dev)
    assert row["train_corr"] == pytest.approx(changed["train_corr"], abs=1e-7)


@pytest.mark.parametrize("failure", ["nan", "constant"])
def test_invalid_model_never_gets_search_reward(failure):
    class Bad(Identity):
        def __call__(self, X):
            if failure == "nan":
                return X * float("nan")
            return torch.ones_like(X)

    row = evaluate_on_split_safe(Bad(), search_split(), get_dataset("laion"))
    assert row["status"].startswith("error:")
    assert np.isnan(row["test_corr"])
    assert row["reward"] is None


@pytest.mark.parametrize("overflow_target", ["model", "neural"])
def test_rsa_rejects_nonfinite_geometry_from_finite_inputs(overflow_target):
    _, X, _, Y, masks, _ = search_split()
    if overflow_target == "model":
        X = X * 1e20
    else:
        Y = Y * 1e20
    assert torch.isfinite(X).all() and torch.isfinite(Y).all()
    with pytest.raises(ValueError, match=f"Nonfinite {overflow_target}.*RDM"):
        rsa_scores(X, Y, masks, torch.device("cpu"))


def test_missing_subject_invalidates_entire_cohort(monkeypatch):
    def fake_load(dataset):
        if dataset.subject == "M3":
            raise FileNotFoundError("missing M3 responses")
        return search_split()

    monkeypatch.setattr("neuro_eval.datasets.load_split", fake_load)
    row = evaluate_subjects(Identity(), get_dataset("triplen"))
    assert row["status"].startswith("error:")
    assert row["failed_subject"] == "M3"
    assert set(row["subject_results"]) == {"M1", "M2"}
    assert np.isnan(row["test_corr"])
    assert row["reward"] is None


def test_partial_cohort_and_search_on_final_data_fail_before_loading(monkeypatch):
    def unexpected_load(dataset):
        pytest.fail("Invalid protocol attempted to open neural data")

    monkeypatch.setattr("neuro_eval.datasets.load_split", unexpected_load)
    monkeypatch.setenv("NEURO_SUBJECT", "M1")
    partial = evaluate_subjects(Identity(), get_dataset("triplen"))
    assert "complete cohort" in partial["status"]
    monkeypatch.delenv("NEURO_SUBJECT")
    wrong_stage = evaluate_subjects(Identity(), get_dataset("nsd"), stage="search")
    assert "fixed 'final'" in wrong_stage["status"]


def test_source_group_rsa_is_aligned_and_does_not_change_primary_reward():
    from neuro_eval.datasets import VisionSplit

    split = search_split()
    baseline = evaluate_on_split_safe(Identity(), split, get_dataset("laion"))
    groups = {"laion": [0, 2, 4, 6], "thingsplus": [1, 3, 5, 7], "ood": [8], "nsd": [0, 1, 2]}
    enriched = VisionSplit(split, source_groups=groups)
    row = evaluate_on_split_safe(Identity(), enriched, get_dataset("laion"))
    assert row["status"] == "ok"
    assert row["reward"] == baseline["reward"]
    assert row["roi_corr"] == baseline["roi_corr"]
    feedback = row["rsa_diagnostics"]
    assert feedback["primary_rsa"] == row["reward"]
    assert set(feedback["source_groups"]) == {"laion", "thingsplus", "ood"}
    for name in ("laion", "thingsplus"):
        idx = groups[name]
        expected = rsa_scores(split[1][idx], split[3][idx], split[4], torch.device("cpu"))
        actual = feedback["source_groups"][name]
        assert actual["rsa"] == expected["test_corr"]
        assert actual["n_stimuli"] == len(idx)
        assert actual["regions"] == {r: v["test"] for r, v in expected["roi_corr"].items()}
    assert feedback["source_groups"]["ood"]["status"] == "unavailable"
    assert feedback["source_groups"]["ood"]["rsa"] is None
    assert feedback["regions"]["early"]["n_targets"] == 2


def test_cohort_diagnostics_retain_nonshared_regions_and_subject_coverage(monkeypatch):
    def fake_load(dataset):
        split = list(search_split(int(dataset.subject[1:])))
        if dataset.subject not in {"M1", "M4"}:
            split[4] = {"ventral": split[4]["ventral"]}
        return tuple(split)

    monkeypatch.setattr("neuro_eval.datasets.load_split", fake_load)
    row = evaluate_subjects(Identity(), get_dataset("triplen"))
    assert row["status"] == "ok"
    assert set(row["roi_corr"]) == {"ventral"}  # compatibility field stays unchanged
    feedback = row["rsa_diagnostics"]
    assert feedback["primary_rsa"] == row["reward"]
    assert feedback["regions"]["early"]["subjects"] == ["M1", "M4"]
    assert feedback["regions"]["early"]["n_targets_by_subject"] == {"M1": 2, "M4": 2}
    assert feedback["regions"]["ventral"]["n_subjects"] == 5
    assert feedback["source_groups"] == {}
    assert row["subjects"] == ["M1", "M2", "M3", "M4", "M5"]
    assert row["protocol_version"] == protocol_for("triplen").version
    assert row["reward"] == np.mean(list(row["subject_scores"].values()))


def test_source_group_cohort_requires_complete_coverage(monkeypatch):
    from neuro_eval.datasets import VisionSplit

    def fake_load(dataset):
        return VisionSplit(search_split(int(dataset.subject[-2:])), source_groups={
            "laion": [0] if dataset.subject == "sub-03" else [0, 1, 2, 3],
            "ood": list(range(9))})

    monkeypatch.setattr("neuro_eval.datasets.load_split", fake_load)
    row = evaluate_subjects(Identity(), get_dataset("laion"))
    assert row["status"] == "ok"
    assert row["subjects"] == ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
    assert row["reward"] == pytest.approx(np.mean(list(row["subject_scores"].values())))
    complete = row["rsa_diagnostics"]["source_groups"]["ood"]
    assert complete["status"] == "ok" and complete["n_subjects"] == 5
    assert complete["rsa"] == pytest.approx(np.mean([
        entry["rsa"] for entry in complete["subject_results"].values()]))
    group = row["rsa_diagnostics"]["source_groups"]["laion"]
    assert group["status"] == "unavailable" and group["rsa"] is None
    assert group["n_subjects"] == 4
    assert group["subject_results"]["sub-03"]["status"] == "unavailable"
    assert all(region["rsa"] is None and region["status"] == "unavailable"
               for region in group["regions"].values())


@pytest.mark.parametrize("phase,dataset_name", [("rsa", "tvsd"), ("encoding", "laion")])
def test_phase_entry_points_reject_other_phase_before_loading(monkeypatch, phase, dataset_name):
    import importlib

    def unexpected_load(dataset):
        pytest.fail("Wrong-phase evaluator opened a dataset")

    monkeypatch.setattr("neuro_eval.datasets.load_split", unexpected_load)
    evaluator = importlib.import_module(f"neuro_eval.{phase}")
    row = evaluator.evaluate_subjects(Identity(), get_dataset(dataset_name))
    assert row["status"].startswith("error:")
    assert row["reward"] is None
    assert np.isnan(row["score"])


@pytest.mark.parametrize("name,subjects", [
    ("nsd", ["01", "02", "03", "04", "05", "06", "07", "08"]),
    ("tvsd", ["F", "N"]),
])
def test_final_encoding_keeps_complete_cohort_and_equal_subject_mean(monkeypatch, name, subjects):
    from neuro_eval.encoding import evaluate_subjects as evaluate_final

    seen = []

    def load(dataset):
        seen.append(dataset.subject)
        X, Y, masks = encoding_fixture(seed=len(seen))
        return X[:22], X[22:], Y[:22], Y[22:], masks, None

    monkeypatch.setattr("neuro_eval.datasets.load_split", load)
    row = evaluate_final(Identity(), get_dataset(name))
    assert row["status"] == "ok"
    assert seen == subjects
    assert row["score"] == pytest.approx(np.mean(list(row["subject_scores"].values())))
    assert row["reward"] is None


# Frozen CPU outputs from looped-networks revision
# 3ee575baeceb1ad784102162a051b7444182fdaa, captured before consolidation.
# Scores check every held-out row; one prediction per target also pins the scale.
SOURCE_BASELINES = {
    "wide": {
        "alphas": [3.359818286283774, 3.359818286283774, 3.359818286283774],
        "scores": [-0.09804017096757889, -0.2485073059797287, -0.4164963364601135],
        "first_prediction": [-0.00874994695186615, -0.0838826447725296, -0.2854132354259491],
    },
    "rank_deficient": {
        "alphas": [37.92690190732238, 6951927.961775591, 10000000000.0],
        "scores": [-0.23567646741867065, -0.4419967532157898, -0.3658258616924286],
        "first_prediction": [0.02152606099843979, -1.4222945310393698e-07, -1.3299694678892138e-10],
    },
}

@pytest.mark.parametrize("case,width", [("wide", 25), ("rank_deficient", 4)])
def test_encoding_frozen_source_regression(case, width):
    rng = np.random.RandomState(173)
    X = torch.tensor(rng.normal(size=(23, width)), dtype=torch.float32)
    Y = torch.tensor(rng.normal(size=(23, 3)), dtype=torch.float32)
    if case == "rank_deficient":
        X = torch.cat([X, X[:, :2], torch.zeros((23, 1))], 1)
    tr, te = np.arange(16), np.arange(16, 23)
    Z = encoding.design(X, tr, "cpu", features="center")
    alphas, actual, prediction = encoding._fit_predict(Z, Y, tr, te)
    baseline = SOURCE_BASELINES[case]
    np.testing.assert_allclose(alphas, baseline["alphas"], rtol=1e-14)
    np.testing.assert_allclose(prediction[0], baseline["first_prediction"], rtol=2e-5, atol=1e-14)
    np.testing.assert_allclose(encoding.pearson(actual, prediction), baseline["scores"],
                               rtol=2e-5, atol=1e-6)


def test_materialized_parameter_cap_is_enforced(monkeypatch):
    class HiddenWeights(Identity):
        def __call__(self, X):
            self.fixed = torch.ones(20)
            return X

    monkeypatch.setattr("neuro_eval.model.MAX_PARAMS", 10)
    row = evaluate_on_split_safe(HiddenWeights(), search_split(), get_dataset("laion"))
    assert "exceeds the 10 cap" in row["status"]
    assert row["reward"] is None


def test_batched_forward_accepts_isolated_numpy_features():
    from neuro_eval.model import _forward, FORWARD_BATCH

    images = torch.arange((FORWARD_BATCH + 3) * 5, dtype=torch.float32).reshape(-1, 5)
    actual = _forward(lambda batch: batch.numpy().copy(), images)
    torch.testing.assert_close(actual, images, rtol=0, atol=0)
