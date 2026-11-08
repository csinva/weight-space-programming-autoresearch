"""Search feature reuse preserves full-cohort ridge scores and model contracts."""

import numpy as np
import pytest
import torch

from neuro_eval import encoding
from neuro_eval.benchmarks import protocol_for
from neuro_eval.datasets import VisionSplit, get_dataset, with_subject
from neuro_eval.rsa import _image_key


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for key in ('NEURO_SUBJECT', 'NEURO_STAGE', 'NEURO_METRIC', 'NEURO_REGION', 'NEURO_MAX_PARAMS'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(encoding, '_readout_device', lambda: torch.device('cpu'))


class RecordingModel:
    n_params = 0

    def __init__(self):
        self.calls = []
        self.square = False

    def __call__(self, images):
        self.calls.append(len(images))
        result = torch.as_tensor(images).float().clone()
        return result.square() if self.square else result


def cohort(*, numpy_inputs=False, variant=None):
    generator = torch.Generator().manual_seed(912)
    shared = torch.randn(8, 4, generator=generator)
    result = {}
    for index, subject in enumerate(protocol_for('laion', 'early', 'encoding').search_subjects):
        train = torch.randn(15, 4, generator=generator)
        test = shared.clone()
        if index == 1:
            if variant == 'changed':
                test[0, 0] += 3
            elif variant == 'reordered':
                test = test.flip(0)
            elif variant == 'dtype':
                test = test.double()
        targets = torch.randn(23, 5, generator=generator)
        masks = {'early visual stream': np.ones(5, dtype=bool),
                 'V1': np.array([True, True, False, False, False]),
                 'V2': np.array([False, True, True, True, False]),
                 'V3': np.array([False, False, False, True, True])}
        if numpy_inputs:
            train, test = train.numpy().copy(), test.numpy().copy()
        result[subject] = VisionSplit((train, test, targets[:15], targets[15:], masks, None))
    return result


def bind(monkeypatch, splits):
    monkeypatch.setattr('neuro_eval.datasets.load_split', lambda d: splits[d.subject])
    return get_dataset('laion', region='early', metric='encoding')


def compare(actual, expected):
    assert actual['status'] == 'ok'
    assert actual['subjects'] == list(expected)
    for subject, baseline in expected.items():
        row = actual['subject_results'][subject]
        for key in ('test_corr', 'train_corr', 'roi_corr', 'stream_diagnostics',
                    'n_train', 'n_test', 'n_targets', 'n_params', 'encoding_features',
                    'encoding_cv_folds', 'encoding_alpha_min', 'encoding_alpha_max'):
            assert row[key] == baseline[key], (subject, key)
    assert actual['score'] == np.mean([r['score'] for r in expected.values()])


@pytest.mark.parametrize('numpy_inputs', [False, True])
@pytest.mark.parametrize('variant,test_calls', [(None, 1), ('changed', 3), ('reordered', 3), ('dtype', 3)])
def test_full_cohort_matches_uncached_ridge_with_exact_image_identity(monkeypatch, numpy_inputs, variant, test_calls):
    splits = cohort(numpy_inputs=numpy_inputs, variant=variant)
    dataset = bind(monkeypatch, splits)
    expected = {subject: encoding.evaluate_on_split(RecordingModel(), split, with_subject(dataset, subject))
                for subject, split in splits.items()}
    model = RecordingModel()
    actual = encoding.evaluate_subjects(model, dataset)
    compare(actual, expected)
    assert model.calls.count(15) == len(splits)  # Private training is never cached.
    assert model.calls.count(8) == test_calls


def test_mutated_same_allocation_cannot_hit_stale_features(monkeypatch):
    splits = cohort()
    dataset = get_dataset('laion', region='early', metric='encoding')
    shared = next(iter(splits.values()))[1]
    loaded = []

    def load(d):
        if len(loaded) == 2:
            shared[0, 0] += 4
        loaded.append(d.subject)
        split = splits[d.subject]
        return VisionSplit((split[0], shared, *split[2:]))

    monkeypatch.setattr('neuro_eval.datasets.load_split', load)
    model = RecordingModel()
    assert encoding.evaluate_subjects(model, dataset)['status'] == 'ok'
    assert model.calls.count(8) == 2


def test_cache_storage_is_bounded_detached_and_not_exposed(monkeypatch):
    cache = encoding._TestFeatureCache()
    features = torch.randn(8, 4, requires_grad=True)
    saved = features.detach().clone()
    cache.put('first', features)
    assert cache.features.device.type == 'cpu'
    assert not cache.features.requires_grad
    with torch.no_grad():
        features.zero_()
    retrieved = cache.get('first')
    assert torch.equal(retrieved, saved)
    retrieved.zero_()
    assert torch.equal(cache.get('first'), saved)
    cache.put('second', saved + 1)
    assert cache.get('first') is None
    monkeypatch.setattr(encoding, '_TEST_FEATURE_CACHE_BYTES', 1)
    cache.put('too_large', saved)
    assert cache.features is None
    assert cache.get('second') is None


def test_oversize_cache_skips_reuse_without_changing_scores(monkeypatch):
    splits = cohort()
    dataset = bind(monkeypatch, splits)
    baseline = encoding.evaluate_subjects(RecordingModel(), dataset)
    monkeypatch.setattr(encoding, '_TEST_FEATURE_CACHE_BYTES', 1)
    model = RecordingModel()
    actual = encoding.evaluate_subjects(model, dataset)
    compare(actual, baseline['subject_results'])
    assert model.calls.count(8) == len(splits)


def test_candidate_cache_cleared_on_success_failure_and_exception(monkeypatch):
    splits = cohort()
    dataset = bind(monkeypatch, splits)
    instances = []
    original = encoding._TestFeatureCache

    class ObservedCache(original):
        def __init__(self):
            super().__init__()
            instances.append(self)

    monkeypatch.setattr(encoding, '_TestFeatureCache', ObservedCache)
    model = RecordingModel()
    assert encoding.evaluate_subjects(model, dataset)['status'] == 'ok'
    model.square = True
    assert encoding.evaluate_subjects(model, dataset)['status'] == 'ok'
    assert model.calls.count(8) == 2
    assert all(c.features is None for c in instances)

    original_scorer = encoding.encoding_scores
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError('deliberate readout failure')
        return original_scorer(*args, **kwargs)

    monkeypatch.setattr(encoding, 'encoding_scores', fail_second)
    failed = encoding.evaluate_subjects(RecordingModel(), dataset)
    assert failed['status'] != 'ok'
    assert failed['reward'] is None
    assert instances[-1].features is None

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(encoding.common, 'evaluate_subjects', interrupted)
    with pytest.raises(KeyboardInterrupt):
        encoding.evaluate_subjects(RecordingModel(), dataset)
    assert instances[-1].features is None


def test_cache_hit_rechecks_current_model_cap(monkeypatch):
    split = next(iter(cohort().values()))
    dataset = get_dataset('laion', region='early', metric='encoding')
    cache = encoding._TestFeatureCache()
    cache.put(_image_key(split[1]), split[1])
    model = RecordingModel()
    model.n_params = 100_001
    # Bypass extraction's earlier guard to isolate the cache-hit contract.
    monkeypatch.setattr(encoding.common, 'encoding_forward', lambda *args, **kwargs: split[0].clone())
    with pytest.raises(ValueError, match='cap'):
        encoding.evaluate_on_split(model, split, dataset, _test_cache=cache)


def test_cache_hit_training_width_change_is_error(monkeypatch):
    splits = cohort()
    dataset = bind(monkeypatch, splits)

    class ChangingWidth(RecordingModel):
        def __call__(self, images):
            result = super().__call__(images)
            return result[:, :3] if self.calls.count(15) > 1 else result

    result = encoding.evaluate_subjects(ChangingWidth(), dataset)
    assert result['status'] != 'ok'
    assert 'width differs' in result['status']
    assert result['reward'] is None


def test_final_evaluation_does_not_use_search_cache_or_forward(monkeypatch):
    split = next(iter(cohort().values()))
    dataset = get_dataset('nsd', region='early', search_metric='encoding')
    monkeypatch.setattr('neuro_eval.datasets.load_split', lambda d: split)

    def prohibited(*args, **kwargs):
        pytest.fail('Final evaluation must retain the original forwarding path')

    monkeypatch.setattr(encoding.common, 'encoding_forward', prohibited)
    monkeypatch.setattr(encoding, '_TestFeatureCache', prohibited)
    model = RecordingModel()
    result = encoding.evaluate_subjects(model, dataset)
    assert result['status'] == 'ok'
    assert result['subjects'] == list(protocol_for('nsd').final_subjects)
    assert model.calls.count(15) == model.calls.count(8) == 4


def test_scoring_cannot_mutate_the_cached_raw_representation(monkeypatch):
    splits = cohort()
    dataset = bind(monkeypatch, splits)
    baseline = encoding.evaluate_subjects(RecordingModel(), dataset)
    original = encoding.encoding_scores

    def mutating_score(train, test, *args, **kwargs):
        result = original(train, test, *args, **kwargs)
        test.zero_()  # A future scorer must not gain access to cache storage.
        return result

    monkeypatch.setattr(encoding, 'encoding_scores', mutating_score)
    actual = encoding.evaluate_subjects(RecordingModel(), dataset)
    compare(actual, baseline['subject_results'])


def test_real_isolated_worker_reused_across_entire_cohort(monkeypatch, tmp_path):
    import subprocess
    from neuro_eval.model import IsolatedModel, check_isolation

    try:
        check_isolation()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    splits = cohort()
    dataset = bind(monkeypatch, splits)
    # Force several forwards per split, as well as several subject splits.
    monkeypatch.setattr(encoding.common, 'ENCODING_FORWARD_BATCH', 6)
    baseline = encoding.evaluate_subjects(RecordingModel(), dataset)
    path = tmp_path / 'model.py'
    path.write_text('class Model:\n def __call__(self, X): return X\nMODEL = Model()\n')
    model = IsolatedModel(path)
    original = subprocess.Popen
    workers = []

    def track(argv, *args, **kwargs):
        process = original(argv, *args, **kwargs)
        if '/worker.py' in argv:
            workers.append(process)
        return process

    monkeypatch.setattr(subprocess, 'Popen', track)
    actual = encoding.evaluate_subjects(model, dataset)
    compare(actual, baseline['subject_results'])
    assert len(workers) == 1
    assert workers[0].poll() is not None
