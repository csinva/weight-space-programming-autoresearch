"""Candidate-local RSA reuse must preserve every subject's scientific result."""

import numpy as np
import pytest
import torch

from neuro_eval import encoding, rsa
from neuro_eval.benchmarks import protocol_for
from neuro_eval.datasets import VisionSplit, get_dataset, with_subject


@pytest.fixture(autouse=True)
def evaluation_environment(monkeypatch):
    for key in ("NEURO_SUBJECT", "NEURO_STAGE", "NEURO_METRIC", "NEURO_MAX_PARAMS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rsa, "_readout_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(encoding, "_readout_device", lambda: torch.device("cpu"))
    # Exercise a multi-batch input without constructing a large synthetic RDM.
    monkeypatch.setattr("neuro_eval.model.FORWARD_BATCH", 4)


class RecordingModel:
    n_params = 0

    def __init__(self, transform=False):
        self.batch_sizes = []
        self.transform = transform

    def __call__(self, images):
        self.batch_sizes.append(len(images))
        features = torch.as_tensor(images).clone()
        return features.square() if self.transform else features


def cohort(name, *, variant=None, numpy_inputs=False):
    subjects = protocol_for(name).search_subjects
    generator = torch.Generator().manual_seed(53)
    images = torch.randn(12, 5, generator=generator)
    result = {}
    for index, subject in enumerate(subjects):
        inputs = images.clone()  # Equal content in independent loader allocations.
        if variant == "changed" and index == 2:
            inputs[0, 0] += 3
        elif variant == "reordered" and index == 2:
            inputs = inputs.flip(0)
        responses = torch.randn(12, 7, generator=generator)
        masks = {"ventral": np.arange(7) >= 2}
        if name == "laion" or subject in {"M1", "M4"}:
            masks["early"] = np.arange(7) < 2
        if numpy_inputs:
            inputs = inputs.numpy().copy()
        # Triple-N must concatenate both folds before forwarding and ranking.
        boundary = 5 if name == "triplen" else 0
        groups = {"laion": [0, 2, 4, 6], "thingsplus": [1, 3, 5, 7],
                  "ood": [8, 9, 10, 11]} if name == "laion" else {}
        result[subject] = VisionSplit((inputs[:boundary], inputs[boundary:],
            responses[:boundary], responses[boundary:], masks, None), source_groups=groups)
    return result


def assert_subject_equivalence(actual, expected):
    assert actual["status"] == "ok"
    assert actual["subjects"] == list(expected)
    for subject, baseline in expected.items():
        row = actual["subject_results"][subject]
        for key in ("test_corr", "train_corr", "roi_corr", "rsa_diagnostics",
                    "n_stimuli", "n_targets", "n_train", "n_test", "protocol_version"):
            assert row[key] == baseline[key], (subject, key)
    assert actual["score"] == np.mean([row["score"] for row in expected.values()])


@pytest.mark.parametrize("name", ["triplen", "laion"])
@pytest.mark.parametrize("numpy_inputs", [False, True])
def test_identical_images_forward_and_rank_once_with_all_subject_results(
        monkeypatch, name, numpy_inputs):
    splits = cohort(name, numpy_inputs=numpy_inputs)
    dataset = get_dataset(name)
    expected = {subject: rsa.evaluate_on_split(RecordingModel(), split, with_subject(dataset, subject))
                for subject, split in splits.items()}
    monkeypatch.setattr("neuro_eval.datasets.load_split", lambda dataset: splits[dataset.subject])
    ranked_sizes = []
    original = rsa.rdm_chunked

    def record_geometry(features, *args, **kwargs):
        ranked_sizes.append(len(features))
        return original(features, *args, **kwargs)

    monkeypatch.setattr(rsa, "rdm_chunked", record_geometry)
    model = RecordingModel()
    actual = rsa.evaluate_subjects(model, dataset)
    assert_subject_equivalence(actual, expected)
    assert model.batch_sizes == [12]
    assert ranked_sizes.count(12) == 1
    assert ranked_sizes.count(4) == (3 if name == "laion" else 0)
    if name == "triplen":
        assert "early" not in actual["rsa_diagnostics"]["regions"]


@pytest.mark.parametrize("variant", ["changed", "reordered"])
@pytest.mark.parametrize("numpy_inputs", [False, True])
def test_different_ordered_images_require_new_features_and_geometry(monkeypatch, variant, numpy_inputs):
    splits = cohort("triplen", variant=variant, numpy_inputs=numpy_inputs)
    dataset = get_dataset("triplen")
    expected = {subject: rsa.evaluate_on_split(RecordingModel(), split, with_subject(dataset, subject))
                for subject, split in splits.items()}
    monkeypatch.setattr("neuro_eval.datasets.load_split", lambda dataset: splits[dataset.subject])
    model = RecordingModel()
    actual = rsa.evaluate_subjects(model, dataset)
    assert_subject_equivalence(actual, expected)
    assert model.batch_sizes == [12, 12]


def test_model_geometry_never_survives_candidate_evaluation(monkeypatch):
    splits = cohort("triplen")
    dataset = get_dataset("triplen")
    monkeypatch.setattr("neuro_eval.datasets.load_split", lambda dataset: splits[dataset.subject])
    model = RecordingModel()
    first = rsa.evaluate_subjects(model, dataset)
    model.transform = True  # Even the same callable object is a fresh evaluation.
    second = rsa.evaluate_subjects(model, dataset)
    other = RecordingModel(transform=True)
    third = rsa.evaluate_subjects(other, dataset)
    assert first["status"] == second["status"] == third["status"] == "ok"
    assert model.batch_sizes == [12, 12]
    assert other.batch_sizes == [12]
    assert first["subject_scores"] != second["subject_scores"]
    assert second["subject_scores"] == third["subject_scores"]


def test_subgroup_cache_respects_each_subjects_ordered_indices(monkeypatch):
    splits = cohort("laion")
    subjects = list(splits)
    for subject, indices in zip(subjects[1:3], ([6, 4, 2, 0], [0, 1, 2, 3])):
        original = splits[subject]
        splits[subject] = VisionSplit(original, source_groups={
            **original.source_groups, "laion": indices})
    dataset = get_dataset("laion")
    expected = {subject: rsa.evaluate_on_split(RecordingModel(), split, with_subject(dataset, subject))
                for subject, split in splits.items()}
    monkeypatch.setattr("neuro_eval.datasets.load_split", lambda dataset: splits[dataset.subject])
    actual = rsa.evaluate_subjects(RecordingModel(), dataset)
    assert_subject_equivalence(actual, expected)


def test_neural_subgroup_cache_reuses_fixed_responses_and_invalidates_mutation(monkeypatch):
    split = next(iter(cohort("laion").values()))
    dataset = get_dataset("laion")
    rsa._NEURAL_RANKS.clear()
    calls = []
    original = rsa.compute_rdm

    def record_neural(responses, *args, **kwargs):
        calls.append(len(responses))
        return original(responses, *args, **kwargs)

    monkeypatch.setattr(rsa, "compute_rdm", record_neural)
    first = rsa.evaluate_on_split(RecordingModel(), split, dataset)
    assert len(calls) == 4  # Selected pool, full set and three source groups.
    second = rsa.evaluate_on_split(RecordingModel(), split, dataset)
    assert len(calls) == 4
    assert first["rsa_diagnostics"] == second["rsa_diagnostics"]
    split[3][0, 2] += 7
    changed = rsa.evaluate_on_split(RecordingModel(), split, dataset)
    assert len(calls) == 8
    rsa._NEURAL_RANKS.clear()
    fresh = rsa.evaluate_on_split(RecordingModel(), split, dataset)
    assert changed["rsa_diagnostics"] == fresh["rsa_diagnostics"]
    assert first["rsa_diagnostics"] != changed["rsa_diagnostics"]


def test_final_encoding_still_batches_train_and_test_separately():
    generator = torch.Generator().manual_seed(12)
    images = torch.randn(20, 5, generator=generator)
    targets = images @ torch.randn(5, 6, generator=generator)
    split = (images[:13], images[13:], targets[:13], targets[13:],
             {"ventral": np.ones(6, dtype=bool)}, None)
    model = RecordingModel()
    result = encoding.evaluate_on_split(model, split, get_dataset("tvsd"))
    assert result["status"] == "ok"
    assert result["reward"] is None
    assert model.batch_sizes == [4, 4, 4, 1, 4, 3]
