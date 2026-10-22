"""RSA scoring for the complete fixed search cohort."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import time
import weakref

import numpy as np
import torch

from neuro_eval import _evaluation as common
from neuro_eval._evaluation import matrix, region_masks
from neuro_eval.model import _validate_model_size, _readout_device
from neuro_eval.reference.metrics import rsa

_NEURAL_RANKS_LIMIT = 16
_NEURAL_RANKS = OrderedDict()


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


def rsa_diagnostics(features, targets, masks, device, scored, source_groups=None) -> dict:
    """Descriptive search-only feedback; subgroup failures never alter reward.

    Subgroups recompute both RDMs and ranks on their own images. They are not
    averaged into the primary score. Neural rank caching is deliberately omitted
    here because the full-cohort cache keys do not identify stimulus subsets.
    """
    masks = region_masks(masks, targets.shape[1])
    groups = {}
    for name, indices in (source_groups or {}).items():
        if name not in {"laion", "thingsplus", "ood"}:
            continue
        indices = np.asarray(indices, dtype=np.int64)
        entry = {"n_stimuli": len(indices), "status": "unavailable", "rsa": None, "regions": {}}
        try:
            if (indices.ndim != 1 or len(set(indices.tolist())) != len(indices)
                    or np.any(indices < 0) or np.any(indices >= len(features))):
                raise ValueError("Invalid source-group stimulus indices")
            subgroup = rsa_scores(features[indices], targets[indices], masks, device)
        except ValueError as exc:
            entry["reason"] = str(exc)
        else:
            entry.update(status="ok", rsa=subgroup["test_corr"],
                         regions={r: values["test"] for r, values in subgroup["roi_corr"].items()})
        groups[name] = entry
    return {"schema_version": 1, "metric": "rsa", "stage": "search",
            "primary_rsa": scored["test_corr"], "n_stimuli": len(features),
            "regions": {name: {"rsa": scored["roi_corr"][name]["test"],
                                "n_targets": int(mask.sum())} for name, mask in masks.items()},
            "source_groups": groups}


def evaluate_on_split(model, split, dataset):
    common._validate_contract(dataset, "search")
    started = time.time()
    X_train, X_test, Y_train, Y_test, masks, _ = split
    device = _readout_device()
    if not len(X_train):
        X, Y = X_test, Y_test
    else:
        X = torch.cat([X_train, X_test]) if torch.is_tensor(X_train) else np.concatenate([X_train, X_test])
        Y = torch.cat([Y_train, Y_test]) if torch.is_tensor(Y_train) else np.concatenate([Y_train, Y_test])
    features = common.forward(model, X, device, "model features")
    n_params = _validate_model_size(model, features.shape[1])
    scored = rsa_scores(features, Y, masks, device, cache_sources=(Y_train, Y_test))
    scored["rsa_diagnostics"] = rsa_diagnostics(features, Y, masks, device, scored,
        getattr(split, "source_groups", {}) if dataset.loader == "laion" else {})
    return common.result_row(model, split, dataset, scored, n_params, started)


def evaluate_on_split_safe(model, split, dataset):
    return common.evaluate_on_split_safe(model, split, dataset, evaluate_on_split)


def evaluate_subjects(model, dataset, *, stage=None):
    if stage is not None and stage != "search":
        return common._error_row(model, None, dataset, ValueError("This evaluator requires search"), 0)
    return common.evaluate_subjects(model, dataset, evaluate_on_split, stage="search")
