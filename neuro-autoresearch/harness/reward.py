"""Fixed reward. The agent should NOT edit this file.

Reward = held-out predictivity of a ridge readout from model activations to recorded neural
activity, on the dataset's reward region (NSD: the ventral visual stream). Only the ridge readout
is fit; the model is never trained. The readout itself (both implementations + the ``NEURO_READOUT``
switch) lives in ``harness.ridge``; this file does the surrounding eval: pick the region's target
columns, run the readout, score the mean per-target Pearson r, and assemble the result row.
The caller persists the row (the search loop records it into a run's ``search.db``; the manual
``harness.run`` path just prints it).
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import torch

from harness.datasets import Dataset
from harness.ridge import ridge_predict, _readout_device, _to_np
from harness.roi import (
    roi_breakdown, _select_targets, _study_for, _slice_target_mask, _standalone_target_mask,
)

MAX_DOUT = 10_000             # hard cap on a model's representation width D_out (ridge feasibility)


def _corr_per_column(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = y_hat - y_hat.mean(0)
    b = y - y.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = np.divide((a * b).sum(0), den, out=np.zeros_like(den), where=den > 0)
    return r


def predictivity(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean per-column Pearson correlation."""
    return float(_corr_per_column(np.asarray(y_hat), np.asarray(y)).mean())


def evaluate_on_split(model, split, dataset: Dataset, slices=()) -> dict:
    """Evaluate ``model`` on an already-loaded ``split`` (the reusable core).

    Runs the model forward, then fits the ``harness.ridge`` readout from the activations to the
    dataset's REWARD-REGION neural targets only (NSD: the ventral stream), and returns the mean
    per-target held-out Pearson r as ``test_corr``. Times only the model + readout (not the one-off
    data load). ``slices`` is accepted for back-compat and ignored. A representation wider than
    ``MAX_DOUT`` is rejected (raised -> recorded as a ``status='error'`` row by the safe wrapper).
    """
    X_train, X_test, Y_train, Y_test, roi_masks = split
    device = _readout_device()
    t0 = time.time()
    # Move the stimulus to the readout device so the model's forward pass runs on the GPU too.
    if torch.is_tensor(X_train):
        X_train, X_test = X_train.to(device), X_test.to(device)
    A_train, A_test = model(X_train), model(X_test)

    d_out = int(A_train.shape[1])
    if d_out >= MAX_DOUT:
        raise ValueError(f"representation D_out={d_out} >= {MAX_DOUT} cap "
                         f"(keep the model's output dimension under {MAX_DOUT})")

    # The reward narrows the targets to the dataset's region (NSD: ventral); the readout fits
    # exactly those targets. NeuralSet studies resolve the region via an ns.Study ROI resolver;
    # the standalone loaders bake the region into Y at load time and may narrow to one named
    # sub-ROI of their roi_masks (a None mask -> fit every loaded target).
    study = _study_for(dataset)
    if study is None:
        mask = _standalone_target_mask(roi_masks, dataset.reward_roi)
    else:
        mask = _slice_target_mask(study, dataset.reward_roi, int(Y_train.shape[1]))
    if mask is not None:
        # Re-align the advisory breakdown to the narrowed targets the readout actually scores.
        roi_masks = {k: np.asarray(v, dtype=bool)[mask] for k, v in roi_masks.items()}
        if torch.is_tensor(Y_train):
            mask = torch.as_tensor(mask, device=Y_train.device)
    Yv_train, Yv_test = _select_targets(Y_train, mask), _select_targets(Y_test, mask)

    pred_train, pred_test = ridge_predict(A_train, A_test, Yv_train, device=device)
    test_np = _corr_per_column(pred_test, _to_np(Yv_test))
    train_np = _corr_per_column(pred_train, _to_np(Yv_train))
    test_corr, train_corr = float(test_np.mean()), float(train_np.mean())

    return {
        "study": dataset.study,
        "model_name": model.name,
        "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
        "summary": getattr(model, "summary", ""),
        "n_params": model.n_params,
        "n_train": int(Yv_train.shape[0]),
        "n_test": int(Yv_test.shape[0]),
        "n_targets": int(Yv_train.shape[1]),
        "test_corr": test_corr,
        "train_corr": train_corr,
        "gap": train_corr - test_corr,
        "roi_corr": roi_breakdown(roi_masks, train_np, test_np),
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
        "reward_roi": dataset.reward_roi,
    }


def _error_status(exc: BaseException) -> str:
    """A compact, single-line ``status`` value for a failed evaluation."""
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())
    return ("error: " + msg)[:300]


def _error_row(model, split, dataset: Dataset, exc: BaseException,
               seconds: float, slices=()) -> dict:
    """A durable ``status='error'`` row for a model/readout fault (NaN scores)."""
    _, _, Y_train, Y_test, _ = split
    nan = float("nan")
    return {
        "study": dataset.study,
        "model_name": getattr(model, "name", "?"),
        "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
        "summary": getattr(model, "summary", ""),
        "n_params": getattr(model, "n_params", nan),
        "n_train": int(Y_train.shape[0]),
        "n_test": int(Y_test.shape[0]),
        "n_targets": int(Y_test.shape[1]) if getattr(Y_test, "ndim", 1) > 1 else 0,
        "test_corr": nan,
        "train_corr": nan,
        "gap": nan,
        "roi_corr": {},
        "seconds": round(seconds, 2),
        "status": _error_status(exc),
        "reward_roi": dataset.reward_roi,
    }


def evaluate_on_split_safe(model, split, dataset: Dataset, slices=()) -> dict:
    """:func:`evaluate_on_split` that never raises on a model/readout fault.

    The unattended autoresearch loop must record a bad proposal rather than crash, so any exception
    (a wrong output shape, a too-wide representation, NaNs, ...) is caught: the traceback is printed
    for the agent to read and a ``status='error'`` row (NaN scores) is returned instead.
    """
    t0 = time.time()
    try:
        return evaluate_on_split(model, split, dataset, slices)
    except Exception as exc:
        traceback.print_exc()
        row = _error_row(model, split, dataset, exc, time.time() - t0, slices)
        _free_gpu(exc)
        return row


def _free_gpu(exc: BaseException | None = None) -> None:
    """Release GPU memory after a failed eval so the next iteration starts clean."""
    import gc
    if exc is not None:
        exc.__traceback__ = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(model, dataset: Dataset, slices=()) -> dict:
    """Load the dataset split and evaluate ``model`` on it.

    ``seconds`` covers the whole from-scratch evaluation (data load + readout); use
    :func:`evaluate_on_split` to reuse a pre-loaded split and time only the readout.
    """
    t0 = time.time()
    from harness.loaders import load_split  # lazy: keeps score-only callers off the data/study stack
    row = evaluate_on_split(model, load_split(dataset), dataset, slices)
    row["seconds"] = round(time.time() - t0, 2)
    return row
