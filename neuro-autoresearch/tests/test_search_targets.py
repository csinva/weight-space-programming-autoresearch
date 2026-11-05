"""Scientific configuration, selected-region reward, and search split firewall."""
import numpy as np
import pytest
import torch

from neuro_eval.benchmarks import protocol_for, validate_search_dataset
from neuro_eval.datasets import _selected_ids, get_dataset
from neuro_eval import rsa, encoding


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ('NEURO_REGION', 'NEURO_METRIC', 'NEURO_STAGE', 'NEURO_SUBJECT'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rsa, '_readout_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(encoding, '_readout_device', lambda: torch.device('cpu'))


@pytest.mark.parametrize('metric', ['rsa', 'encoding'])
def test_config_contract_and_unsupported_cohort(metric):
    early = protocol_for('laion', region='early', metric=metric)
    ventral = protocol_for('laion', region='ventral', metric=metric)
    assert early.name != ventral.name
    assert early.search_subjects == ventral.search_subjects
    assert early.search_metric == metric
    with pytest.raises(ValueError, match='M2, M3, M5'):
        protocol_for('triplen', region='early', metric=metric)
    for name in ('nsd', 'tvsd'):
        with pytest.raises(ValueError, match='reserved'):
            validate_search_dataset(name)


def native_split():
    return {'regions': ['ventral'], 'neural': {'ventral': {0: {
        'train': {'1': None, '2': None, '3': None},
        'test': {'4': None, '5': None, '6': None}}}},
        'shared_test_ids': ['4', '5', '6'], 'test_subsets': {'nsd': ['5']},
        'nsd_source_ids': ['2', '5']}


def test_laion_keeps_all_shared_test_and_native_training_images():
    native = native_split()
    assert _selected_ids(get_dataset('laion', metric='rsa'), native, 0) == ([], ['4', '5', '6'])
    assert _selected_ids(get_dataset('laion', metric='encoding'), native, 0) == (['1', '2', '3'], ['4', '5', '6'])
    del native['nsd_source_ids']
    assert _selected_ids(get_dataset('laion', metric='encoding'), native, 0) == (['1', '2', '3'], ['4', '5', '6'])


def test_triplen_rsa_keeps_all_images_encoding_retains_source_fold():
    native = native_split()
    assert _selected_ids(get_dataset('triplen'), native, 0) == ([], ['1', '2', '3', '4', '5', '6'])
    assert _selected_ids(get_dataset('triplen', metric='encoding'), native, 0) == (['1', '2', '3'], ['4', '5', '6'])


@pytest.mark.parametrize('metric', ['rsa', 'encoding'])
@pytest.mark.parametrize('region', ['early', 'ventral'])
def test_reward_is_selected_region_and_preserves_diagnostics(metric, region):
    gen = torch.Generator().manual_seed(17)
    x = torch.randn(24, 6, generator=gen)
    y = torch.cat([x[:, :3], torch.randn(24, 3, generator=gen)], 1)
    masks = {'early visual stream': np.arange(6) < 3,
             'ventral visual stream': np.arange(6) >= 3}
    boundary = 16 if metric == 'encoding' else 0
    split = (x[:boundary], x[boundary:], y[:boundary], y[boundary:], masks, None)
    class Model:
        n_params = 0
        def __call__(self, x):
            return x
    evaluator = rsa if metric == 'rsa' else encoding
    d = get_dataset('laion', region=region, metric=metric)
    row = evaluator.evaluate_on_split(Model(), split, d)
    assert row['reward'] == row['roi_corr'][f'{region} visual stream']['test']
    assert set(row['roi_corr']) == {d.reward_roi}
    assert row['search_region'] == region
    assert row['metric'] == metric


def test_scientific_configuration_separates_dataset_cache_identity():
    variants = {get_dataset('laion', region=region, metric=metric)
                for region in ('early', 'ventral') for metric in ('rsa', 'encoding')}
    assert len(variants) == 4


def test_final_rows_preserve_originating_search_metric():
    from neuro_eval._evaluation import _row_metadata, _validate_contract
    dataset = get_dataset('nsd', region='early', search_metric='encoding')
    _validate_contract(dataset, 'final')
    metadata = _row_metadata(dataset)
    assert dataset.metric == 'encoding'
    assert dataset.reward_roi == 'early visual stream'
    assert metadata['protocol'] == protocol_for('laion', region='early', metric='encoding').name
    assert metadata['search_image_policy'] == protocol_for('laion', region='early', metric='encoding').image_policy


@pytest.mark.parametrize('region', ['early', 'ventral'])
def test_rsa_source_groups_use_selected_region(region):
    from neuro_eval.datasets import VisionSplit
    generator = torch.Generator().manual_seed(92)
    images = torch.randn(12, 5, generator=generator)
    responses = torch.randn(12, 7, generator=generator)
    masks = {'early visual stream': np.arange(7) < 3,
             'ventral visual stream': np.arange(7) >= 3}
    split = VisionSplit((images[:0], images, responses[:0], responses, masks, None),
                        source_groups={'laion': [0, 2, 4, 6, 8], 'nsd': [3, 5, 7, 9, 11], 'ood': [1]})
    class Model:
        n_params = 0
        def __call__(self, x):
            return x
    dataset = get_dataset('laion', region=region)
    row = rsa.evaluate_on_split(Model(), split, dataset)
    for source in ('laion', 'nsd'):
        subgroup = row['rsa_diagnostics']['source_groups'][source]
        assert subgroup['status'] == 'ok'
        assert subgroup['rsa'] == subgroup['regions'][dataset.reward_roi]
        assert set(subgroup['regions']) == {dataset.reward_roi}
    assert row['rsa_diagnostics']['source_groups']['ood']['status'] == 'unavailable'
    assert row['reward'] == row['roi_corr'][dataset.reward_roi]['test']


def test_rsa_unavailable_other_region_does_not_invalidate_selected_reward():
    from neuro_eval.datasets import VisionSplit
    generator = torch.Generator().manual_seed(192)
    images = torch.randn(12, 5, generator=generator)
    responses = torch.randn(12, 7, generator=generator)
    responses[:, :3] = 1  # Undefined early geometry, valid ventral geometry.
    masks = {'early visual stream': np.arange(7) < 3,
             'ventral visual stream': np.arange(7) >= 3}
    split = VisionSplit((images[:0], images, responses[:0], responses, masks, None),
                        source_groups={'laion': [0, 2, 4, 6, 8]})
    class Model:
        n_params = 0
        def __call__(self, x):
            return x
    valid = rsa.evaluate_on_split_safe(Model(), split, get_dataset('laion', region='ventral'))
    assert valid['status'] == 'ok'
    assert np.isfinite(valid['reward'])
    assert 'early visual stream' not in valid['rsa_diagnostics']['regions']
    group = valid['rsa_diagnostics']['source_groups']['laion']
    assert group['status'] == 'ok'
    assert group['rsa'] == group['regions']['ventral visual stream']
    invalid = rsa.evaluate_on_split_safe(Model(), split, get_dataset('laion', region='early'))
    assert invalid['status'].startswith('error:')
    assert invalid['reward'] is None


def test_stream_masks_intersect_unit_identity_without_duplicate_targets():
    from neuro_eval.datasets import _stream_masks
    d = get_dataset('laion', region='ventral')
    pooled = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    native = {'units': {d.reward_roi: {0: pooled}}, 'diagnostic_units': {
        'hV4': pooled[[0, 1]], 'FFA': np.array([[99, 0, 0]]),
        'PPA': np.array([[2, 0, 0], [3, 0, 0], [98, 0, 0]])}}
    masks, missing = _stream_masks(d, native, 0, {d.reward_roi: np.ones(4, dtype=bool)})
    assert masks['hV4'].tolist() == [True, True, False, False]
    assert masks['PPA'].tolist() == [False, True, True, False]
    assert set(masks) == {d.reward_roi, 'hV4', 'PPA'}
    assert 'FFA' in missing
    assert all(len(mask) == 4 for mask in masks.values())


@pytest.mark.parametrize('metric', ['rsa', 'encoding'])
def test_subregions_preserve_pooled_reward_and_model_calls(metric, monkeypatch):
    from neuro_eval.datasets import VisionSplit
    generator = torch.Generator().manual_seed(121)
    x = torch.randn(28, 7, generator=generator)
    y = torch.randn(28, 9, generator=generator)
    d = get_dataset('laion', region='ventral', metric=metric)
    boundary = 20 if metric == 'encoding' else 0
    base = {d.reward_roi: np.ones(9, dtype=bool)}
    extended = {**base, 'hV4': np.arange(9) < 4, 'PPA': (np.arange(9) >= 2) & (np.arange(9) < 6)}
    class Model:
        n_params = 0
        calls = 0
        def __call__(self, x):
            self.calls += 1
            return x
    evaluator = rsa if metric == 'rsa' else encoding
    results, calls, fits = [], [], []
    original_fit = encoding.unit_scores
    def record_fit(*args, **kwargs):
        fits.append(1)
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(encoding, 'unit_scores', record_fit)
    for masks in [base, extended]:
        split = VisionSplit((x[:boundary], x[boundary:], y[:boundary], y[boundary:], masks, None))
        model = Model()
        results.append(evaluator.evaluate_on_split(model, split, d))
        calls.append(model.calls)
    assert results[0]['reward'] == results[1]['reward']
    assert calls[0] == calls[1]
    assert len(fits) == (2 if metric == 'encoding' else 0)
    assert results[1]['stream_diagnostics']['subregions']['FFA']['status'] == 'unavailable'
    assert results[1]['stream_diagnostics']['subregions']['hV4']['status'] == 'ok'


def test_subregion_cohort_never_reports_partial_mean():
    from neuro_eval._evaluation import _cohort_stream_diagnostics
    rows = []
    for i in range(5):
        entry = {'status': 'ok' if i < 4 else 'unavailable', 'score': .4 if i < 4 else None, 'n_targets': 3}
        rows.append({'subject': f's{i}', 'stream_diagnostics': {'subregions': {'FFA': entry}}})
    diagnostic = _cohort_stream_diagnostics(rows, .7)
    assert diagnostic['primary_score'] == .7
    assert diagnostic['subregions']['FFA']['score'] is None
    assert diagnostic['subregions']['FFA']['n_subjects'] == 4
    assert diagnostic['subregions']['FFA']['status'] == 'unavailable'


def test_identical_it_pool_masks_reuse_rsa_score(monkeypatch):
    generator = torch.Generator().manual_seed(124)
    x = torch.randn(12, 5, generator=generator)
    y = torch.randn(12, 8, generator=generator)
    mask = np.ones(8, dtype=bool)
    calls = []
    original = rsa.spearman_from_ranks
    def record(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(rsa, 'spearman_from_ranks', record)
    scores = rsa.rsa_scores(x, y, {'ventral visual stream': mask, 'IT': mask.copy()}, torch.device('cpu'))
    assert len(calls) == 1
    assert scores['roi_corr']['IT'] == scores['roi_corr']['ventral visual stream']


def test_undefined_subregion_rsa_preserves_valid_pool():
    from neuro_eval.datasets import VisionSplit
    generator = torch.Generator().manual_seed(125)
    x = torch.randn(12, 5, generator=generator)
    y = torch.randn(12, 8, generator=generator)
    d = get_dataset('laion', region='ventral')
    masks = {d.reward_roi: np.ones(8, dtype=bool), 'hV4': np.arange(8) == 0}
    split = VisionSplit((x[:0], x, y[:0], y, masks, None))
    class Model:
        n_params = 0
        def __call__(self, x):
            return x
    result = rsa.evaluate_on_split_safe(Model(), split, d)
    assert result['status'] == 'ok'
    assert np.isfinite(result['reward'])
    assert result['stream_diagnostics']['subregions']['hV4']['status'] == 'unavailable'
    assert result['stream_diagnostics']['subregions']['hV4']['score'] is None
