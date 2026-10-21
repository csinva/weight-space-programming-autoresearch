"""Validated adapters around the pinned, unmodified looped-networks metrics.

RSA is Pearson dissimilarity followed by average-rank Spearman, independently
for every supplied anatomical region. Encoding uses the source's default
``features='center'`` and train-only per-unit ridge cross-validation. Regions
receive equal weight; callers average the resulting subject scores equally.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import weakref

import numpy as np
import torch

from harness.vendor.looped.metrics import encoding, rsa

SOURCE_COMMIT = "3ee575baeceb1ad784102162a051b7444182fdaa"
ENCODING_FEATURES = "center"
_NEURAL_RANKS_LIMIT = 16  # enough for both search cohorts; float64 ranks stay on CPU
_NEURAL_RANKS: OrderedDict = OrderedDict()


def matrix(value, name: str, *, rows: int | None = None) -> torch.Tensor:
    """Reject invalid representations/responses before they can produce reward."""
    result = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty-width 2D matrix; got {tuple(result.shape)}")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{name} has {result.shape[0]} rows; expected {rows}")
    if result.is_complex() or not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values")
    result = result.float()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} overflows the source metric's float32 representation")
    return result


def region_masks(masks: dict, n_targets: int) -> dict[str, np.ndarray]:
    if not masks:
        raise ValueError("Vision scoring requires named anatomical region masks")
    result = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_targets,) or not mask.any():
            raise ValueError(f"Region {name!r} has an empty or misaligned target mask")
        result[name] = mask
    return result


def _rsa_ranks(rdm: torch.Tensor, label: str) -> torch.Tensor:
    ranks = rsa.triu_ranks(rdm)
    if not torch.isfinite(ranks).all() or not ranks.numel() or ranks.max() == ranks.min():
        raise ValueError(f"Degenerate {label} RDM: RSA is undefined")
    return ranks


def _neural_ranks(targets, mask, device, region, cache_sources):
    """Reuse fixed neural geometry, never model geometry.

    Keys retain weak references to the original immutable loader tensors, not
    temporary concatenations or memory addresses alone. Tensor version counters
    invalidate ordinary in-place edits; identity checks prevent pointer reuse.
    Arrays/inference tensors without mutation tracking deliberately bypass cache.
    """
    key = None
    if cache_sources and all(torch.is_tensor(value) for value in cache_sources):
        try:
            key = (tuple((id(value), value._version, tuple(value.shape), value.dtype, str(value.device))
                         for value in cache_sources), hashlib.sha256(mask.tobytes()).digest())
        except RuntimeError:
            pass
    if key in _NEURAL_RANKS:
        refs, ranks = _NEURAL_RANKS[key]
        if all(ref() is value for ref, value in zip(refs, cache_sources)):
            _NEURAL_RANKS.move_to_end(key)
            return ranks.to(device)
        del _NEURAL_RANKS[key]
    response = targets[:, torch.as_tensor(mask, device=targets.device)].to(device)
    if response.shape[1] < 2 or torch.any(response.var(dim=1, unbiased=False) <= 0):
        raise ValueError(f"Degenerate neural responses in region {region!r}")
    ranks = _rsa_ranks(rsa.compute_rdm(response), f"neural {region}")
    if key is not None:
        _NEURAL_RANKS[key] = tuple(weakref.ref(value) for value in cache_sources), ranks.cpu()
        while len(_NEURAL_RANKS) > _NEURAL_RANKS_LIMIT:
            _NEURAL_RANKS.popitem(last=False)
    return ranks


def rsa_scores(features, targets, masks, device, *, cache_sources=None) -> dict:
    """Score one fixed search stimulus set; no readout is fitted."""
    features = matrix(features, "model features")
    targets = matrix(targets, "neural responses", rows=len(features))
    if len(features) < 3 or features.shape[1] < 2:
        raise ValueError("RSA requires at least three stimuli and two model features")
    # A constant feature vector has no Pearson geometry, even though numerical
    # epsilon guards in the source RDM builder can make its RDM appear finite.
    if torch.any(features.var(dim=1, unbiased=False) <= 0):
        raise ValueError("Degenerate model features: an RSA stimulus row is constant")
    model_ranks = _rsa_ranks(rsa.rdm_chunked(features, device, on_device=True), "model")
    scores = {}
    for region, mask in region_masks(masks, targets.shape[1]).items():
        neural_ranks = _neural_ranks(targets, mask, device, region, cache_sources)
        score = rsa.spearman_from_ranks(model_ranks, neural_ranks)
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite RSA score for region {region!r}")
        scores[region] = {"train": score, "test": score}
    mean = float(np.mean([row["test"] for row in scores.values()]))
    return {"train_corr": mean, "test_corr": mean, "roi_corr": scores,
            "metric_detail": "pearson_rdm_average_rank_spearman",
            "n_stimuli": len(features), "encoding_features": None}


def encoding_scores(train_features, test_features, train_targets, test_targets, masks, device) -> dict:
    """One source ridge fit; training predictions are diagnostics, never CV data."""
    train_features = matrix(train_features, "training model features")
    test_features = matrix(test_features, "test model features")
    train_targets = matrix(train_targets, "training neural responses", rows=len(train_features))
    test_targets = matrix(test_targets, "test neural responses", rows=len(test_features))
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Training and test feature widths differ")
    if train_targets.shape[1] != test_targets.shape[1]:
        raise ValueError("Training and test neural target widths differ")
    n_train, n_test = len(train_features), len(test_features)
    if n_train < encoding.N_FOLDS or n_test < 2:
        raise ValueError("Encoding requires at least five training stimuli and two test stimuli")
    masks = region_masks(masks, train_targets.shape[1])
    features = torch.cat([train_features.to(device), test_features.to(device)])
    targets = torch.cat([train_targets.to(device), test_targets.to(device)])
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    design = encoding.design(features, tr, device, features=ENCODING_FEATURES)
    # Query both splits in one fitted model. Only tr enters the solver/CV and
    # target scaling; source subsets reuse predictions without a second fit.
    _, alphas, scores = encoding.unit_scores(
        design, targets, tr, np.arange(len(features)), subsets={"train": tr, "test": te})
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Encoding returned nonfinite unit scores")
    regions = {name: {split: float(values[mask].mean()) for split, values in scores.items()}
               for name, mask in masks.items()}
    return {"train_corr": float(np.mean([row["train"] for row in regions.values()])),
            "test_corr": float(np.mean([row["test"] for row in regions.values()])),
            "roi_corr": regions, "metric_detail": "ridge_5fold_per_unit_pearson_r",
            "encoding_features": ENCODING_FEATURES, "encoding_cv_folds": encoding.N_FOLDS,
            "encoding_alpha_min": float(alphas.min()), "encoding_alpha_max": float(alphas.max())}
