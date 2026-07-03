"""Fixed reward. The agent should NOT edit this file.

Reward = held-out predictivity of a ridge readout from model activations to recorded neural
activity, on the dataset's reward region (e.g. NSD's ventral visual stream). Only the ridge readout
is fit; the model is never trained. The readout itself (both implementations + the ``NEURO_READOUT``
switch) lives in ``harness.ridge``; this file does the surrounding eval: pick the region's target
columns, run the readout, score the reward (noise-normalized mean predictivity), and assemble the row.
The caller persists the row (the search loop records it into a run's ``search.db``; the manual
``harness.run`` path just prints it).
"""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import torch

from harness.datasets import Dataset
from harness.ridge import ridge_predict, _readout_device, _to_np
from harness.roi import roi_breakdown, select_targets, reward_region_mask

MAX_DOUT = 10_000             # hard cap on a model's representation width D_out (ridge feasibility)
FORWARD_BATCH = 512           # chunk the forward pass over samples (per-sample invariant => identical output)


def _forward(model, X):
    """Run ``model`` over ``X`` in ``FORWARD_BATCH``-sized sample chunks, concatenating the per-chunk
    activations. Identical to one ``model(X)`` call given the per-sample invariant (no statistic
    across the batch), but keeps a heavy model's peak memory bounded by the chunk, not all N."""
    if not torch.is_tensor(X) or X.shape[0] <= FORWARD_BATCH:
        return model(X)
    return torch.cat([model(X[i:i + FORWARD_BATCH]) for i in range(0, X.shape[0], FORWARD_BATCH)], 0)


def _experiment_notes(model) -> str:
    """The proposer-authored notes captured from the model's class attributes, as a compact JSON
    blob ``{"description", "insight"}``: a plain-language account of the model and the key takeaway
    it carries. This is what cross-island ``peek-island`` surfaces (idea + notes, no source), so it
    must be self-contained -- a re-implementable description, not hyperparameter shorthand. Each
    field is kept to a few information-dense sentences."""
    return json.dumps({
        "description": (getattr(model, "description", "") or "")[:600],
        "insight": (getattr(model, "insight", "") or "")[:600],
    })


def _corr_per_column(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = y_hat - y_hat.mean(0)
    b = y - y.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = np.divide((a * b).sum(0), den, out=np.zeros_like(den), where=den > 0)
    return r


def _noise_normalize(r: np.ndarray, nc) -> np.ndarray:
    """Per-voxel noise-normalized score: Algonauts r^2 / NC_r2 (fraction of EXPLAINABLE variance the
    model captured) where a per-voxel ceiling ``nc`` exists, else raw Pearson r (datasets without
    repeat-based ceilings keep the unnormalized reward).

    Sign-preserving (``sign(r) * r^2 / NC_r2``) so an anti-correlated voxel scores NEGATIVE rather
    than being rewarded like a well-predicted one. The denominator is floored at a tiny epsilon as a
    pure divide-by-zero guard; the real ceiling floor lives at voxel-selection time
    (``_ncsnr_select``) so near-zero-ceiling voxels never reach this average regardless of the
    ``NSD_NCSNR_MIN`` reliability cut."""
    if nc is None:
        return r
    nc = np.maximum(np.asarray(nc, dtype=float), 1e-12)
    return np.sign(r) * r ** 2 / nc


def predictivity(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean per-column Pearson correlation."""
    return float(_corr_per_column(np.asarray(y_hat), np.asarray(y)).mean())


def _cv_column_r(A, Y, k_folds: int, device, nc):
    """Per-column predictivity under contiguous temporal K-fold CV, averaged over the folds.

    A single fixed hold-out window is trivially overfit by a search that optimizes it directly (a
    proposer bakes features tuned to that one window). Rotating the held-out fold over the whole
    recording removes that fixed target -- a model must generalize across every fold, so only
    genuinely predictive features raise the reward. Rows are already in trigger-onset order (train
    block first, test tail last), so contiguous index folds are temporal. Returns (train_col_r,
    test_col_r) aligned to Y's columns."""
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
    """Evaluate ``model`` on an already-loaded ``split`` (the reusable core).

    Runs the model forward, then fits the ``harness.ridge`` readout from the activations to the
    dataset's REWARD-REGION neural targets only (e.g. the ventral stream on NSD), and returns the
    noise-normalized mean over reward-region voxels as ``test_corr`` (Algonauts r^2/NC_r2 where the
    loader supplies a per-voxel ceiling, else raw mean Pearson r). Times only the model + readout (not the one-off
    data load). ``slices`` is accepted for back-compat and ignored. A representation wider than
    ``MAX_DOUT`` is rejected (raised -> recorded as a ``status='error'`` row by the safe wrapper).
    """
    X_train, X_test, Y_train, Y_test, roi_masks, nc = split
    device = _readout_device()
    t0 = time.time()
    # Move the stimulus to the readout device so the model's forward pass runs on the GPU too.
    if torch.is_tensor(X_train):
        X_train, X_test = X_train.to(device), X_test.to(device)
    A_train, A_test = _forward(model, X_train), _forward(model, X_test)

    d_out = int(A_train.shape[1])
    if d_out >= MAX_DOUT:
        raise ValueError(f"representation D_out={d_out} >= {MAX_DOUT} cap "
                         f"(keep the model's output dimension under {MAX_DOUT})")

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
    else:
        pred_train, pred_test = ridge_predict(A_train, A_test, Yv_train, device=device)
        test_np = _noise_normalize(_corr_per_column(pred_test, _to_np(Yv_test)), nc)
        train_np = _noise_normalize(_corr_per_column(pred_train, _to_np(Yv_train)), nc)
    # Round the scalar reward to 4 decimals at the SOURCE so every downstream surface -- the stored
    # DB value, champion ranking, and the proposer's reward feedback -- sees the same 4-dp number,
    # for every dataset. (roi_breakdown rounds its per-ROI means likewise.)
    test_corr, train_corr = round(float(test_np.mean()), 4), round(float(train_np.mean()), 4)

    return {
        "study": dataset.study,
        "model_name": model.name,
        "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
        "experiment_notes": _experiment_notes(model),
        "n_params": model.n_params,
        "n_train": int(Yv_train.shape[0]),
        "n_test": int(Yv_test.shape[0]),
        "n_targets": int(Yv_train.shape[1]),
        "test_corr": test_corr,
        "train_corr": train_corr,
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
    _, _, Y_train, Y_test, _, _ = split
    nan = float("nan")
    return {
        "study": dataset.study,
        "model_name": getattr(model, "name", "?"),
        "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
        "experiment_notes": _experiment_notes(model),
        "n_params": getattr(model, "n_params", nan),
        "n_train": int(Y_train.shape[0]),
        "n_test": int(Y_test.shape[0]),
        "n_targets": int(Y_test.shape[1]) if getattr(Y_test, "ndim", 1) > 1 else 0,
        "test_corr": nan,
        "train_corr": nan,
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


def _mean_rows(rows: list[dict]) -> dict:
    """Average ok per-subject rows into one row: scalar test/train_corr + per-ROI roi_corr (equal
    weight per subject -- each is already noise-ceiling-normalized to its own ceiling). Metadata
    (model_name/idea/n_params) is taken from the first row; ``seconds`` is summed. Scalars and
    per-ROI means are rounded to 4 dp to match the single-subject ``evaluate_on_split`` reward, so
    both paths feed the DB / ranking / proposer feedback the same precision. Only ROIs present for
    EVERY subject are averaged -- a reliability cut can leave an ROI empty (and thus absent from
    ``roi_breakdown``) on some subjects but not others, and a partial mean would be misleading."""
    f = rows[0]
    common = set.intersection(*(set(row.get("roi_corr") or {}) for row in rows))
    roi_avg = {r: {k: round(float(np.mean([row["roi_corr"][r][k] for row in rows])), 4)
                   for k in (f["roi_corr"][r])}
               for r in (f.get("roi_corr") or {}) if r in common}
    return {**f, "test_corr": round(float(np.mean([row["test_corr"] for row in rows])), 4),
            "train_corr": round(float(np.mean([row["train_corr"] for row in rows])), 4),
            "roi_corr": roi_avg, "seconds": round(sum(row.get("seconds", 0) for row in rows), 2)}


def evaluate_subjects(model, dataset: Dataset, slices=()) -> dict:
    """The search reward. ``NEURO_SUBJECT`` selects the subject(s): a single id (e.g. ``"1"``) scores
    that one; a comma list (e.g. ``"1,2"``) scores EACH and averages the rows into one reward
    (proposers see only the mean). Unset -> the dataset's default subject. A fault on ANY subject is
    the row's status, so a recorded model must work on all listed subjects."""
    from harness.datasets import with_subject
    from harness.loaders import load_split

    subs = os.environ.get("NEURO_SUBJECT")
    if not subs:
        return evaluate_on_split_safe(model, load_split(dataset), dataset, slices)
    rows = []
    for s in (x.strip() for x in subs.split(",") if x.strip()):
        d = with_subject(dataset, s)
        row = evaluate_on_split_safe(model, load_split(d), d, slices)
        row["subject"] = d.subject
        if not str(row.get("status", "")).startswith("ok"):
            return row                       # propagate the first subject's failure as the reward
        rows.append(row)
    return _mean_rows(rows)
