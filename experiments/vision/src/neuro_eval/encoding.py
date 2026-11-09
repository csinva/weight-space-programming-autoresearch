"""Encoding scoring for complete search or final fixed cohorts.

Numerical kernels copied unchanged from source revision
3ee575baeceb1ad784102162a051b7444182fdaa, metrics/encoding.py.
Unused source experiment helpers are omitted. Cohort policies live in benchmarks.
Train statistics center/scale features and z-score targets; five unshuffled folds
select each target's alpha by mean negative MSE with the source log-alpha tie
preference. The smaller Gram is eigensolved in float64, avoiding structural null
space noise from a float32 sample-space solve when features are fewer than images.
"""

from __future__ import annotations

from contextlib import nullcontext
import time

import numpy as np
import torch

from neuro_eval import _evaluation as common
from neuro_eval._evaluation import matrix, region_masks
from neuro_eval.model import IsolatedModel, _validate_model_size, _readout_device
from neuro_eval.rsa import _image_key

ALPHAS = np.logspace(-10, 10, 20)
N_FOLDS = 5
N_TARGETS_BATCH = 1000


def design(X, train_idx, device, *, features):
    """(n, D) float32 design Z on device, scaled so that Z @ Z.T == gram(X, train_idx, device, features=features).

    features="zscore": every feature z-scored, then divided by sqrt(D) (equal weight per block).
    features="center": features only centred (as in the Gallant lab voxelwise tutorials, StandardScaler(with_std=False)),
        so units keep their relative scales; Z is divided by the root mean squared train norm (the Gram's mean train
        diagonal). That is one global scale factor, the same as rescaling the alpha grid, so the per-voxel CV over
        alphas works on a comparable range for every model.
    """
    Z = X.to(device).float()
    tr = torch.as_tensor(train_idx, device=Z.device)
    Z = Z - Z[tr].mean(0)
    if features == "zscore":
        return Z / (Z[tr].std(0) + 1e-8) / Z.shape[1] ** 0.5
    if features == "center":
        return Z / Z[tr].pow(2).sum(1).mean().sqrt().clamp_min(1e-6)
    return Z / Z.shape[1] ** 0.5


def zscore_targets(Y, tr, te):
    """Y[tr] and Y[te], each z-scored per voxel with the train mean and std."""
    mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-8
    return (Y[tr] - mu) / sd, (Y[te] - mu) / sd


def pearson(a, b):
    """Per-column Pearson r of two (n, t) tensors."""
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)


class _Ridge:
    """Ridge of Y on Z for every alpha at once: coef(a, cols) such that the prediction is (Zq @ V) @ coef,
    i.e. Zq (Z'Z + a I)^-1 Z'Y.

    Only the eigendecomposition of the smaller Gram is done in float64: D <= n, Z'Z (D x D; prediction
    Zq V diag(1/(lam + a)) V'Z'Y); D > n, ZZ' (n x n, full rank; prediction Zq Z'U diag(1/(lam + a)) U'Y). Neither
    has structural zero eigenvalues, and float64 keeps the small ones far below any alpha even at condition numbers
    ~1e17. The projections of Y stay in the input dtype: their rounding error only enters along directions whose
    prediction weight is equally small."""

    def __init__(self, Z, Y):
        self.dtype = Z.dtype
        Zd = Z.double()
        if Z.shape[1] <= Z.shape[0]:
            lam, V = torch.linalg.eigh(Zd.T @ Zd)
            self.V = V.to(Z.dtype)
            self.PtY = (Z @ self.V).T @ Y  # V'Z'Y
        else:
            lam, U = torch.linalg.eigh(Zd @ Zd.T)
            U = U.to(Z.dtype)
            self.V, self.PtY = Z.T @ U, U.T @ Y  # Z'U, U'Y
        self.lam, self.PtY = lam.clamp_min(0), self.PtY.double()

    def coef(self, alpha, cols=slice(None)):
        return (self.PtY[:, cols] / (self.lam + alpha)[:, None]).to(self.dtype)

    def predict(self, Zq, alpha, cols=slice(None)):
        return (Zq @ self.V) @ self.coef(alpha, cols)


def _fit_predict(Z, Y, tr, te):
    """Per-voxel ridge fit on tr with alpha by 5-fold CV; returns (chosen alphas, z-scored Y[te], its prediction)."""
    dev = Z.device
    tr_t, te_t = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
    Ytr, Yte = zscore_targets(Y.to(dev), tr_t, te_t)
    Ztr, t = Z[tr_t], Ytr.shape[1]
    cv =torch.zeros(len(ALPHAS), t, dtype=torch.float64, device=dev)
    n = len(tr)
    for val in np.array_split(np.arange(n), N_FOLDS):  # = sklearn KFold(N_FOLDS), as himalaya's cv=5
        fit = torch.as_tensor(np.setdiff1d(np.arange(n), val), device=dev)
        val = torch.as_tensor(val, device=dev)
        r = _Ridge(Ztr[fit], Ytr[fit])
        Zv = Ztr[val] @ r.V
        for b in range(0, t, N_TARGETS_BATCH):
            cols = slice(b, b + N_TARGETS_BATCH)
            for i, a in enumerate(ALPHAS):
                pred = Zv @ r.coef(float(a), cols)
                cv[i, cols] -= ((Ytr[val, cols] - pred) ** 2).mean(0).double()
    cv = cv / N_FOLDS + torch.log(torch.as_tensor(ALPHAS, device=dev))[:, None] * 1e-10  # ties -> larger alpha
    best = cv.argmax(0)
    r = _Ridge(Ztr, Ytr)
    Zte = Z[te_t] @ r.V
    pred = torch.empty_like(Yte)
    for i in best.unique().tolist():
        cols = (best == i).nonzero().squeeze(1)
        pred[:, cols] = Zte @ r.coef(float(ALPHAS[i]), cols)
    return ALPHAS[best.cpu().numpy()], Yte, pred


def unit_scores(Z, Y, tr, te, subsets=None):
    """One ridge fit on rows tr of the design Z (see design): ((n_voxels,) Pearson r on te, chosen alphas,
    {name: (n_voxels,) r on each subset of te (positions into te)}), float64 numpy. Voxels are fit independently,
    so any set of them can share the fit."""
    a, Yte, pred = _fit_predict(Z, Y, tr, te)
    r = lambda x, y: pearson(x, y).double().cpu().numpy()
    pick = lambda x, v: x[torch.as_tensor(v, device=x.device)]
    return r(Yte, pred), a, {k: r(pick(Yte, v), pick(pred, v)) for k, v in (subsets or {}).items()}


ENCODING_FEATURES = "center"


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
    if n_train < N_FOLDS or n_test < 2:
        raise ValueError("Encoding requires at least five training stimuli and two test stimuli")
    masks = region_masks(masks, train_targets.shape[1])
    features = torch.cat([train_features.to(device), test_features.to(device)])
    targets = torch.cat([train_targets.to(device), test_targets.to(device)])
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    Z = design(features, tr, device, features=ENCODING_FEATURES)
    # Query both splits in one fitted model. Only tr enters the solver/CV and
    # target scaling; source subsets reuse predictions without a second fit.
    _, alphas, scores = unit_scores(
        Z, targets, tr, np.arange(len(features)), subsets={"train": tr, "test": te})
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Encoding returned nonfinite unit scores")
    regions = {name: {split: float(values[mask].mean()) for split, values in scores.items()}
               for name, mask in masks.items()}
    return {"train_corr": float(np.mean([row["train"] for row in regions.values()])),
            "test_corr": float(np.mean([row["test"] for row in regions.values()])),
            "roi_corr": regions, "metric_detail": "ridge_5fold_per_unit_pearson_r",
            "encoding_features": ENCODING_FEATURES, "encoding_cv_folds": N_FOLDS,
            "encoding_alpha_min": float(alphas.min()), "encoding_alpha_max": float(alphas.max())}


# One raw test representation at most; no training features or fitted readouts.
# A maximal search representation for 1,492 images occupies less than 60 MiB.
_TEST_FEATURE_CACHE_BYTES = 128 * 1024 * 1024


class _TestFeatureCache:
    def __init__(self):
        self.clear()

    def clear(self):
        self.key = None
        self.features = None

    def get(self, key):
        if key == self.key and self.features is not None:
            # Callers/scorers never receive the cache's own mutable storage.
            return self.features.clone()
        return None

    def put(self, key, features):
        self.clear()
        if features.numel() * features.element_size() <= _TEST_FEATURE_CACHE_BYTES:
            self.features = features.detach().cpu().clone()
            self.key = key


def evaluate_on_split(model, split, dataset, *, _test_cache=None):
    common._validate_contract(dataset)
    if dataset.metric != "encoding":
        raise ValueError("Encoding evaluator requires metric=encoding")
    started = time.time()
    X_train, X_test, Y_train, Y_test, masks, _ = split
    masks = common.selected_masks(masks, dataset)
    device = _readout_device()
    feature_forward = common.encoding_forward if dataset.stage == "search" else common.forward
    A_train = feature_forward(model, X_train, device, "training model features")
    key = _image_key(X_test) if _test_cache is not None and dataset.stage == "search" else None
    A_test = _test_cache.get(key) if key is not None else None
    cache_hit = A_test is not None
    if not cache_hit:
        A_test = feature_forward(model, X_test, device, "test model features")
    # A model can materialize more parameters during another subject's training
    # forward; reuse must never bypass the current model's limits.
    if cache_hit:
        _validate_model_size(model, A_test.shape[1])
    if A_train.shape[1] != A_test.shape[1]:
        raise ValueError("Model representation width differs between training and test stimuli")
    n_params = _validate_model_size(model, A_train.shape[1])
    if key is not None and not cache_hit:
        _test_cache.put(key, A_test)
    scored = encoding_scores(A_train, A_test, Y_train, Y_test, masks, device)
    common.select_reward(scored, dataset)
    return common.result_row(model, split, dataset, scored, n_params, started)


def evaluate_on_split_safe(model, split, dataset):
    return common.evaluate_on_split_safe(model, split, dataset, evaluate_on_split)


def evaluate_subjects(model, dataset, *, stage=None):
    if dataset.metric != "encoding":
        return common._error_row(model, None, dataset, ValueError("Encoding evaluator requires metric=encoding"), 0)
    if stage is not None and stage != dataset.stage:
        return common._error_row(model, None, dataset, ValueError("Evaluator stage conflicts with dataset"), 0)
    if dataset.stage != "search":
        return common.evaluate_subjects(model, dataset, evaluate_on_split, stage=dataset.stage)
    cache = _TestFeatureCache()

    def scorer(model, split, subject_dataset):
        return evaluate_on_split(model, split, subject_dataset, _test_cache=cache)

    try:
        session = model.encoding_session(_readout_device()) if isinstance(model, IsolatedModel) else nullcontext()
        with session:
            return common.evaluate_subjects(model, dataset, scorer, stage=dataset.stage)
    finally:
        # No representation survives success, an error row, or an exception.
        cache.clear()
