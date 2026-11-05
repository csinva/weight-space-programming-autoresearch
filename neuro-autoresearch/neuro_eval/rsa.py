"""RSA search scoring for the complete fixed cohort.

Numerical kernels copied unchanged from looped-networks, revision
3ee575baeceb1ad784102162a051b7444182fdaa, looped/metrics/rsa.py.
Unused source experiment helpers are omitted. Cohort policies live in benchmarks.
"""

from __future__ import annotations

from collections import OrderedDict
import logging
import hashlib
import time
import weakref

import numpy as np
import torch

from neuro_eval import _evaluation as common
from neuro_eval._evaluation import matrix, region_masks
from neuro_eval.model import _validate_model_size, _readout_device

logger = logging.getLogger(__name__)


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Row-wise ordinal ranks via double argsort (tied values get distinct consecutive ranks)."""
    return torch.argsort(torch.argsort(x, dim=1), dim=1).float()


def compute_rdm(
    representations: torch.Tensor, *, correlation: str = "Pearson", correction: float = 1e-12
) -> torch.Tensor:
    """(n_samples x n_samples) RDM (1 - correlation), Pearson or Spearman.

    Diagonal is 0 (zero self-dissimilarity); off-diagonal ranges 0 (identical)
    to 2 (perfectly anti-correlated).
    """
    corr = correlation.lower()
    if corr not in {"pearson", "spearman"}:
        raise ValueError("correlation must be 'Pearson' or 'Spearman'")

    x = representations.float().clone()
    if corr == "spearman":
        x = _rank(x)

    x -= x.mean(dim=1, keepdim=True)
    std = x.pow(2).mean(dim=1).add(correction).sqrt()

    zero_mask = std < correction * 10
    if zero_mask.any():
        logger.warning("%d/%d rows have ~zero variance for %s RDM", zero_mask.sum(), len(std), correlation)
        std[zero_mask] = 1.0

    cov = (x @ x.T) / x.size(1)
    corr_mat = cov / (std[:, None] * std[None, :] + correction)
    corr_mat.clamp_(-1, 1).fill_diagonal_(1.0)
    return 1.0 - corr_mat


def triu_ranks(rdm: torch.Tensor) -> torch.Tensor:
    """Average ranks (as scipy.stats.rankdata) of the upper triangle of rdm, float64 on its device."""
    n = rdm.size(0)
    i = torch.triu_indices(n, n, offset=1, device=rdm.device)
    v, order = rdm[i[0], i[1]].double().sort()
    _, inv, counts = torch.unique_consecutive(v, return_inverse=True, return_counts=True)
    avg = counts.cumsum(0).double() - (counts.double() - 1) / 2  # 1-based mean rank of each tie block
    ranks = torch.empty_like(v)
    ranks[order] = avg[inv]
    return ranks


def spearman_from_ranks(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman of two RDMs from their triu_ranks: Pearson of the ranks (= scipy.stats.spearmanr); NaN if constant."""
    a, b = a - a.mean(), b.to(a.device) - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def rdm_chunked(X, device, chunk=200_000, zscore_units=False, on_device=False):
    """1 - Pearson correlation RDM of X (n, D), on the CPU or already on `device` (no host copy then), accumulated
    over feature chunks. Returns float32 on the CPU, or on `device` if on_device.

    zscore_units: first standardise every unit (column) over the n images, so that no unit dominates the
    correlation by its scale or its mean (units constant over the images become 0).
    """
    n, D = X.shape

    def cols(j):
        x = X[:, j:j + chunk].to(device).double()
        if zscore_units:
            sd = x.std(0)
            x = torch.where(sd > 0, (x - x.mean(0)) / sd.clamp_min(1e-12), torch.zeros_like(x))
        return x

    mean = torch.zeros(n, dtype=torch.float64, device=device)
    for j in range(0, D, chunk):
        mean += cols(j).sum(1)
    mean /= D
    G = torch.zeros(n, n, dtype=torch.float64, device=device)
    for j in range(0, D, chunk):
        xc = (cols(j) - mean[:, None]).float()
        G += (xc @ xc.T).double()
        del xc
    d = torch.sqrt(torch.diag(G)).clamp_min(1e-12)
    rdm = 1.0 - G / (d[:, None] * d[None, :])
    rdm.fill_diagonal_(0.0)
    return rdm.float() if on_device else rdm.float().cpu()


_NEURAL_RANKS_LIMIT = 64
_NEURAL_RANKS = OrderedDict()


def _rsa_ranks(rdm: torch.Tensor, label: str) -> torch.Tensor:
    # Ranking NaNs can turn invalid geometry into finite, nonconstant ranks.
    if not torch.isfinite(rdm).all():
        raise ValueError(f"Nonfinite {label} RDM: RSA is undefined")
    ranks = triu_ranks(rdm)
    if not torch.isfinite(ranks).all() or not ranks.numel() or ranks.max() == ranks.min():
        raise ValueError(f"Degenerate {label} RDM: RSA is undefined")
    return ranks


def _neural_ranks(targets, mask, device, region, cache_sources, cache_indices=None):
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
                         for value in cache_sources), hashlib.sha256(mask.tobytes()).digest(),
                   cache_indices)
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
    ranks = _rsa_ranks(compute_rdm(response), f"neural {region}")
    if key is not None:
        _NEURAL_RANKS[key] = tuple(weakref.ref(value) for value in cache_sources), ranks.cpu()
        while len(_NEURAL_RANKS) > _NEURAL_RANKS_LIMIT:
            _NEURAL_RANKS.popitem(last=False)
    return ranks


def _model_ranks(features, device):
    """Validate and prepare the model geometry using the source numerical kernel."""
    if len(features) < 3 or features.shape[1] < 2:
        raise ValueError("RSA requires at least three stimuli and two model features")
    # Epsilon guards in the RDM builder must not hide constant stimulus rows.
    if torch.any(features.var(dim=1, unbiased=False) <= 0):
        raise ValueError("Degenerate model features: an RSA stimulus row is constant")
    return _rsa_ranks(rdm_chunked(features, device, on_device=True), "model")


def rsa_scores(features, targets, masks, device, *, cache_sources=None, cache_indices=None,
               model_ranks=None, required_region=None) -> dict:
    """Score one fixed search stimulus set; no readout is fitted."""
    features = matrix(features, "model features")
    targets = matrix(targets, "neural responses", rows=len(features))
    if model_ranks is None:
        model_ranks = _model_ranks(features, device)
    scores, unavailable, identical_masks = {}, {}, {}
    for region, mask in region_masks(masks, targets.shape[1]).items():
        mask_key = mask.tobytes()
        if mask_key in identical_masks:
            scores[region] = dict(identical_masks[mask_key])
            continue
        try:
            neural_ranks = _neural_ranks(targets, mask, device, region, cache_sources, cache_indices)
            score = spearman_from_ranks(model_ranks, neural_ranks)
            if not np.isfinite(score):
                raise ValueError(f"Nonfinite RSA score for region {region!r}")
        except ValueError as exc:
            if required_region is None or region == required_region:
                raise
            unavailable[region] = str(exc)
            continue
        scores[region] = {"train": score, "test": score}
        identical_masks[mask_key] = scores[region]
    if required_region is not None and required_region not in scores:
        raise ValueError(f"Selected region {required_region!r} has no valid RSA score")
    mean = float(np.mean([row["test"] for row in scores.values()]))
    return {"train_corr": mean, "test_corr": mean, "roi_corr": scores,
            "metric_detail": "pearson_rdm_average_rank_spearman",
            "n_stimuli": len(features), "encoding_features": None, "unavailable_regions": unavailable}


def rsa_diagnostics(features, targets, masks, device, scored, source_groups=None,
                    *, model_rank_cache=None, cache_sources=None, dataset=None) -> dict:
    """Descriptive search-only feedback; subgroup failures never alter reward.

    Subgroups compute both RDMs and ranks on their own images. They are not
    averaged into the primary score. Cached subgroup geometry identifies the
    ordered indices as well as the original full stimulus/response tensors.
    """
    masks = region_masks(masks, targets.shape[1])
    groups = {}
    for name, indices in (source_groups or {}).items():
        if name not in {"laion", "thingsplus", "ood", "nsd"}:
            continue
        indices = np.asarray(indices, dtype=np.int64)
        entry = {"n_stimuli": len(indices), "status": "unavailable", "rsa": None, "regions": {}}
        try:
            if (indices.ndim != 1 or len(set(indices.tolist())) != len(indices)
                    or np.any(indices < 0) or np.any(indices >= len(features))):
                raise ValueError("Invalid source-group stimulus indices")
            subgroup_features = features[indices]
            key = tuple(indices.tolist())
            ranks = model_rank_cache.get(key) if model_rank_cache is not None else None
            if ranks is None:
                ranks = _model_ranks(subgroup_features, device)
                if model_rank_cache is not None:
                    model_rank_cache[key] = ranks
            subgroup = rsa_scores(subgroup_features, targets[indices], masks, device,
                                  model_ranks=ranks, cache_sources=cache_sources,
                                  cache_indices=key, required_region=(
                                      dataset.reward_roi if dataset.reward_roi in masks else dataset.search_region
                                      ) if dataset is not None else None)
            if dataset is not None:
                common.select_reward(subgroup, dataset)
        except ValueError as exc:
            entry["reason"] = str(exc)
        else:
            entry.update(status="ok", rsa=subgroup["test_corr"],
                         unavailable_regions=subgroup.get("unavailable_regions", {}),
                         regions={r: values["test"] for r, values in subgroup["roi_corr"].items()})
        groups[name] = entry
    return {"schema_version": 1, "metric": "rsa", "stage": "search",
            "primary_rsa": scored["test_corr"], "n_stimuli": len(features),
            "regions": {name: {"rsa": scored["roi_corr"][name]["test"],
                                "n_targets": int(mask.sum())} for name, mask in masks.items() if name in scored["roi_corr"]},
            "unavailable_regions": scored.get("unavailable_regions", {}),
            "source_groups": groups,
            "search_region": dataset.search_region if dataset is not None else None,
            "aggregation": ("selected_region" if dataset is not None
                            else "equal_available_region_mean")}


def _image_key(images):
    """Content identity includes every input byte in stimulus order, shape and dtype.

    Loader subjects can use separately allocated tensors or different images of
    the same shape. Neither subject labels nor allocation identity prove reuse.
    Hash the current contents on every lookup so mutations cannot hit stale data.
    """
    if torch.is_tensor(images):
        array = images.detach().cpu().contiguous()
        dtype, shape = str(array.dtype), tuple(array.shape)
        raw = array.view(torch.uint8).numpy()
    else:
        array = np.ascontiguousarray(images)
        dtype, shape = array.dtype.str, array.shape
        raw = array.view(np.uint8)
    return dtype, shape, hashlib.sha256(memoryview(raw)).digest()


def evaluate_on_split(model, split, dataset, *, _model_cache=None):
    common._validate_contract(dataset, "search")
    if dataset.metric != "rsa":
        raise ValueError("RSA evaluator requires metric=rsa")
    started = time.time()
    X_train, X_test, Y_train, Y_test, masks, _ = split
    masks = common.selected_masks(masks, dataset)
    device = _readout_device()
    if not len(X_train):
        X, Y = X_test, Y_test
    else:
        X = torch.cat([X_train, X_test]) if torch.is_tensor(X_train) else np.concatenate([X_train, X_test])
        Y = torch.cat([Y_train, Y_test]) if torch.is_tensor(Y_train) else np.concatenate([Y_train, Y_test])
    key = _image_key(X) if _model_cache is not None else None
    prepared = _model_cache.get(key) if _model_cache is not None else None
    if prepared is None:
        features = common.forward(model, X, device, "model features", full_batch=True)
    else:
        features = prepared[0]
    n_params = _validate_model_size(model, features.shape[1])
    if prepared is None:
        prepared = features, _model_ranks(features, device), {}
        if _model_cache is not None:
            _model_cache[key] = prepared
    scored = rsa_scores(features, Y, masks, device, cache_sources=(Y_train, Y_test),
                        model_ranks=prepared[1], required_region=(dataset.reward_roi
                            if dataset.reward_roi in masks else dataset.search_region))
    common.select_reward(scored, dataset)
    scored["unavailable_regions"].update(getattr(split, "unavailable_regions", {}))
    scored["rsa_diagnostics"] = rsa_diagnostics(features, Y, masks, device, scored,
        getattr(split, "source_groups", {}) if dataset.loader == "laion" else {},
        model_rank_cache=prepared[2], cache_sources=(Y_train, Y_test), dataset=dataset)
    return common.result_row(model, split, dataset, scored, n_params, started)


def evaluate_on_split_safe(model, split, dataset):
    return common.evaluate_on_split_safe(model, split, dataset, evaluate_on_split)


def evaluate_subjects(model, dataset, *, stage=None):
    if stage is not None and stage != "search":
        return common._error_row(model, None, dataset, ValueError("This evaluator requires search"), 0)
    # Lifetime is exactly one candidate cohort evaluation. No model outputs or
    # geometry survive into another candidate or another evaluation of this one.
    model_cache = {}

    def scorer(model, split, dataset):
        return evaluate_on_split(model, split, dataset, _model_cache=model_cache)

    return common.evaluate_subjects(model, dataset, scorer, stage="search")
