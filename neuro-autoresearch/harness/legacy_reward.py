"""Historical language/encoding readout, separate from the fixed vision protocols.

Kept for optional language experiments and historical analysis. Current vision
RSA and final encoding use harness.metrics and never noise-normalize scores.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from harness.datasets import Dataset
from harness.ridge import ridge_predict, _readout_device, _to_np
from harness.roi import roi_breakdown, select_targets, reward_region_mask


def _corr_per_column(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-target Pearson r, with zero for constant columns."""
    a = y_hat - y_hat.mean(0)
    b = y - y.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = np.divide((a * b).sum(0), den, out=np.zeros_like(den), where=den > 0)
    return r


def _noise_normalize(r: np.ndarray, nc) -> np.ndarray:
    """Historical sign-preserving r²/noise-ceiling normalization; raw r without a ceiling."""
    if nc is None:
        return r
    nc = np.maximum(np.asarray(nc, dtype=float), 1e-12)
    return np.sign(r) * r ** 2 / nc


def predictivity(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean per-column Pearson correlation."""
    return float(_corr_per_column(np.asarray(y_hat), np.asarray(y)).mean())


def _cv_column_r(A, Y, k_folds: int, device, nc):
    """Average per-target scores over contiguous temporal K-fold splits.

    Every fold trains on all other rows, including later rows; this is block
    cross-validation, not causal forward chaining."""
    A, Y = A.to(device), Y.to(device)        # A is on the GPU, Y may be on the CPU -- align both
    n = A.shape[0]
    train_np = test_np = None
    for k in range(k_folds):
        lo, hi = round(k * n / k_folds), round((k + 1) * n / k_folds)
        keep = torch.ones(n, dtype=torch.bool, device=device)
        keep[lo:hi] = False
        pred_tr, pred_te = ridge_predict(A[keep], A[lo:hi], Y[keep], device=device)
        te = _noise_normalize(_corr_per_column(pred_te, _to_np(Y[lo:hi])), nc)
        tr = _noise_normalize(_corr_per_column(pred_tr, _to_np(Y[keep])), nc)
        test_np = te if test_np is None else test_np + te
        train_np = tr if train_np is None else train_np + tr
    return train_np / k_folds, test_np / k_folds


def evaluate_on_split(model, split, dataset: Dataset, slices=()) -> dict:
    """Historical language evaluation: optional feature pipeline, ridge, and noise normalization."""
    from harness.reward import MAX_DOUT, _experiment_notes, _forward, _parameter_limit, robust_n_params

    X_train, X_test, Y_train, Y_test, roi_masks, nc = split
    device = _readout_device()
    t0 = time.time()
    # Move the stimulus to the readout device so the model's forward pass runs on the GPU too.
    # .to(cuda) copies; .to(cpu) is a no-op, so clone on CPU to keep the shared memoized split
    # safe from an in-place-mutating model.
    if torch.is_tensor(X_train):
        X_train, X_test = X_train.to(device), X_test.to(device)
        if device.type == "cpu":
            X_train, X_test = X_train.clone(), X_test.clone()
    A_train, A_test = _forward(model, X_train), _forward(model, X_test)

    d_out = int(A_train.shape[1])
    if d_out >= MAX_DOUT:
        raise ValueError(f"representation D_out={d_out} >= {MAX_DOUT} cap "
                         f"(keep the model's output dimension under {MAX_DOUT})")

    # The ONE parameter-cap check: robust (every fixed tensor the model materialised, now that the
    # forward has built any lazy banks) and fail-fast (before the expensive ridge). Over the cap is a
    # recorded error, not a crash (evaluate_on_split_safe catches it). This is the sole enforcement --
    # no separate pre-forward check, which would only under-count the not-yet-built weights anyway.
    n_params = robust_n_params(model)
    limit = _parameter_limit()
    if n_params > limit:
        raise ValueError(f"n_params {n_params:,} exceeds the {limit:,} cap "
                         f"(counts ALL fixed weights the model materialises, not just declared)")

    # Optional per-dataset feature pipeline between the model and the fixed ridge. For the Huth
    # fMRI-language dataset ("lanczos_fir") the per-word model embeddings are Lanczos-downsampled to
    # the fMRI TR grid, z-scored, and FIR-delayed (the reference src/features.py pipeline), turning
    # per-word activations into the TR-rate design matrix aligned with the TR responses Y. A pure
    # no-op for every other dataset (empty feature_pipeline), so the reward is unchanged elsewhere.
    if getattr(dataset, "feature_pipeline", ""):
        from harness.loaders import apply_feature_pipeline
        A_train, A_test = (apply_feature_pipeline(dataset, A_train, "train"),
                           apply_feature_pipeline(dataset, A_test, "test"))

    # The reward narrows the targets to the dataset's region; the readout fits exactly those targets.
    # Every loader bakes its region into Y at load time and may narrow to one named sub-ROI of its
    # roi_masks (a None mask -> fit every loaded target).
    mask = reward_region_mask(roi_masks, dataset.reward_roi)
    if mask is not None:
        # Re-align the advisory breakdown + the per-voxel ceiling to the narrowed targets scored.
        roi_masks = {k: np.asarray(v, dtype=bool)[mask] for k, v in roi_masks.items()}
        if nc is not None:
            nc = nc[mask]
        if torch.is_tensor(Y_train):
            mask = torch.as_tensor(mask, device=Y_train.device)
    Yv_train, Yv_test = select_targets(Y_train, mask), select_targets(Y_test, mask)

    # Per-column held-out predictivity. cv_folds>1 (iEEG) uses rotating temporal K-fold CV so there
    # is no single fixed hold-out window a proposer can bake features against; every other dataset
    # keeps its single train/test split. Both then noise-normalize (r^2/NC_r2, or raw r when nc is
    # None) and average identically downstream -- the scalar reward AND the per-ROI breakdown.
    if getattr(dataset, "cv_folds", 0) > 1:
        A_full, Y_full = torch.cat([A_train, A_test], 0), torch.cat([Yv_train, Yv_test], 0)
        train_np, test_np = _cv_column_r(A_full, Y_full, dataset.cv_folds, device, nc)
        n_test = round(int(Y_full.shape[0]) / dataset.cv_folds)   # CV: one representative fold's split
        n_train = int(Y_full.shape[0]) - n_test
    else:
        pred_train, pred_test = ridge_predict(A_train, A_test, Yv_train, device=device)
        test_np = _noise_normalize(_corr_per_column(pred_test, _to_np(Yv_test)), nc)
        train_np = _noise_normalize(_corr_per_column(pred_train, _to_np(Yv_train)), nc)
        n_train, n_test = int(Yv_train.shape[0]), int(Yv_test.shape[0])
    # Round the scalar reward to 4 decimals at the SOURCE so every downstream surface -- the stored
    # DB value, champion ranking, and the proposer's reward feedback -- sees the same 4-dp number,
    # for every dataset. (roi_breakdown rounds its per-ROI means likewise.)
    test_corr, train_corr = round(float(test_np.mean()), 4), round(float(train_np.mean()), 4)

    return {
        "study": dataset.study,
        "model_name": model.name,
        "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
        "experiment_notes": _experiment_notes(model),
        "n_params": n_params,   # robust: declared vs. actually-materialised tensors, whichever larger
        "n_train": n_train,
        "n_test": n_test,
        "n_targets": int(Yv_train.shape[1]),
        "test_corr": test_corr,
        "train_corr": train_corr,
        "roi_corr": roi_breakdown(roi_masks, train_np, test_np),
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
        "reward_roi": dataset.reward_roi,
    }
