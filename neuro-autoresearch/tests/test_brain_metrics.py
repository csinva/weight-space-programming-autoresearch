"""Source agreement, split isolation, and complete-cohort reward contracts."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import rankdata, spearmanr

from harness.datasets import get_dataset
from harness.metrics import encoding_scores, rsa_scores
from harness.protocols import protocol_for
from harness.reward import evaluate_on_split_safe, evaluate_subjects
from harness.vendor.looped.metrics import encoding, rsa


@pytest.fixture(autouse=True)
def clean_evaluation_env(monkeypatch):
    for key in ("NEURO_SUBJECT", "NEURO_STAGE", "NEURO_METRIC", "NEURO_MAX_PARAMS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("harness.reward._readout_device", lambda: torch.device("cpu"))


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


def encoding_fixture():
    rng = np.random.RandomState(40)
    X = torch.tensor(rng.normal(size=(30, 6)), dtype=torch.float32)
    Y = X @ torch.tensor(rng.normal(size=(6, 7)), dtype=torch.float32)
    Y += torch.tensor(rng.normal(size=Y.shape), dtype=torch.float32) * 0.3
    masks = {"early": np.arange(7) < 2, "ventral": np.arange(7) >= 2}
    return X, Y, masks


def test_average_ranks_match_scipy_for_ties():
    rdm = torch.tensor([[0., 1., 1., 2.], [1., 0., 2., 3.], [1., 2., 0., 3.], [2., 3., 3., 0.]])
    upper = rdm.numpy()[np.triu_indices(4, 1)]
    np.testing.assert_array_equal(rsa.triu_ranks(rdm).numpy(), rankdata(upper, method="average"))


def test_rsa_matches_numpy_pearson_and_scipy_with_equal_region_weight():
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


def test_encoding_matches_source_and_ignores_heldout_data_during_fit():
    X, Y, masks = encoding_fixture()
    tr, te, dev = np.arange(22), np.arange(22, 30), torch.device("cpu")
    design = encoding.design(X, tr, dev, features="center")
    direct, alphas, _ = encoding.unit_scores(design, Y, tr, te)
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


@pytest.mark.parametrize("failure", ["nan", "rows", "width", "constant"])
def test_invalid_model_never_gets_search_reward(failure):
    class Bad(Identity):
        def __call__(self, X):
            if failure == "nan":
                return X * float("nan")
            if failure == "rows":
                return X[:-1]
            if failure == "width":
                return X.reshape(-1)
            return torch.ones_like(X)

    row = evaluate_on_split_safe(Bad(), search_split(), get_dataset("laion"))
    assert row["status"].startswith("error:")
    assert np.isnan(row["test_corr"])
    assert row["reward"] is None


def test_materialized_parameter_cap_is_enforced(monkeypatch):
    class HiddenWeights(Identity):
        def __call__(self, X):
            self.fixed = torch.ones(20)
            return X

    monkeypatch.setattr("harness.reward.MAX_PARAMS", 10)
    row = evaluate_on_split_safe(HiddenWeights(), search_split(), get_dataset("laion"))
    assert "exceeds the 10 cap" in row["status"]
    assert row["reward"] is None


def test_default_search_loads_every_subject_and_never_opens_ood(monkeypatch):
    seen = []

    def fake_load(dataset):
        seen.append((dataset.loader, dataset.subject))
        return search_split(len(seen))

    monkeypatch.setattr("harness.loaders.load_split", fake_load)
    row = evaluate_subjects(Identity(), get_dataset("laion"))
    protocol = protocol_for("laion")
    assert seen == [("laion", subject) for subject in protocol.search_subjects]
    assert row["subjects"] == list(protocol.search_subjects)
    assert row["test_corr"] == pytest.approx(np.mean(list(row["subject_scores"].values())))
    assert row["reward"] == row["test_corr"]
    assert row["metric"] == "rsa" and row["stage"] == "search"
    assert row["protocol_version"] == protocol.version


def test_missing_subject_invalidates_entire_cohort(monkeypatch):
    def fake_load(dataset):
        if dataset.subject == "M3":
            raise FileNotFoundError("missing M3 responses")
        return search_split()

    monkeypatch.setattr("harness.loaders.load_split", fake_load)
    row = evaluate_subjects(Identity(), get_dataset("triplen"))
    assert row["status"].startswith("error:")
    assert row["failed_subject"] == "M3"
    assert set(row["subject_results"]) == {"M1", "M2"}
    assert np.isnan(row["test_corr"])


def test_partial_cohort_and_search_on_final_data_fail_before_loading(monkeypatch):
    def unexpected_load(dataset):
        pytest.fail("Invalid protocol attempted to open neural data")

    monkeypatch.setattr("harness.loaders.load_split", unexpected_load)
    monkeypatch.setenv("NEURO_SUBJECT", "M1")
    partial = evaluate_subjects(Identity(), get_dataset("triplen"))
    assert "complete cohort" in partial["status"]
    monkeypatch.delenv("NEURO_SUBJECT")
    wrong_stage = evaluate_subjects(Identity(), get_dataset("nsd"), stage="search")
    assert "fixed 'final'" in wrong_stage["status"]


def test_final_encoding_is_not_exposed_as_search_reward():
    X, Y, masks = encoding_fixture()
    split = X[:22], X[22:], Y[:22], Y[22:], masks, np.full(7, 0.01)
    row = evaluate_on_split_safe(Identity(), split, get_dataset("tvsd"))
    assert row["status"] == "ok"
    assert row["metric"] == "encoding" and row["stage"] == "final"
    assert row["reward"] is None
    assert -1 <= row["score"] <= 1
