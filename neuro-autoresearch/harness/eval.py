"""Fixed reward + result saving. The agent should NOT edit this file.

Reward = held-out predictivity of a ridge readout from model activations to
recorded neural activity. Only the ridge readout is fit; the model is never trained.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from config import Dataset
from data import load_split

# Ridge penalties searched by the readout. The readout picks the best alpha PER
# TARGET via efficient leave-one-out GCV (one SVD reused across all alphas), so
# searching 10 alphas costs ~one fit, not 10x. A fixed alpha=1.0 massively
# overfit small samples (train_corr >> test_corr); per-target CV closes that gap.
_ALPHAS = np.logspace(-3, 6, 10)


def _corr_per_column(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = y_hat - y_hat.mean(0)
    b = y - y.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = np.divide((a * b).sum(0), den, out=np.zeros_like(den), where=den > 0)
    return r


def predictivity(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean per-column Pearson correlation."""
    return float(_corr_per_column(np.asarray(y_hat), np.asarray(y)).mean())


def _readout_device() -> torch.device:
    """The A100 when present, else CPU. The readout is identical on either."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_f64(a, device: torch.device) -> torch.Tensor:
    """Model activations / targets -> a contiguous float64 tensor on ``device``."""
    if not torch.is_tensor(a):
        a = torch.as_tensor(np.asarray(a))
    return a.to(device=device, dtype=torch.float64)


def _corr_per_column_t(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-column Pearson r on the GPU (torch mirror of ``_corr_per_column``)."""
    a = y_hat - y_hat.mean(0)
    b = y - y.mean(0)
    den = torch.sqrt((a * a).sum(0) * (b * b).sum(0))
    return torch.where(den > 0, (a * b).sum(0) / den, torch.zeros_like(den))


def _ridge_gcv_coef(Xc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
    """Per-target ridge coefficients with leave-one-out GCV alpha selection.

    Reproduces ``RidgeCV(alphas=_ALPHAS, alpha_per_target=True)`` (fit_intercept):
    one reduced SVD of the centered design ``Xc`` is reused across all alphas to
    score each target's leave-one-out error in closed form, and each target keeps
    the coefficients of its own best alpha. ``Xc``/``yc`` are already centered, so
    an unpenalised intercept is emulated by cancelling the all-ones direction.
    """
    n, p = Xc.shape
    U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V, s2 = Vh.transpose(-2, -1), s * s
    UT_y = U.t() @ yc
    UT_1 = U.t() @ torch.ones(n, dtype=Xc.dtype, device=Xc.device)
    wide = n <= p

    best_score = best_coef = None
    for alpha in (float(a) for a in _ALPHAS):
        if wide:
            d = alpha / (s2 + alpha)
            alpha_c = U @ (d[:, None] * UT_y)
            alpha_d = (d * U ** 2).sum(1)
            g = U @ (d * UT_1)
        else:  # n > p: reduced SVD misses (n-p) zero-eigenvalue directions
            m = alpha / (s2 + alpha) - 1.0
            alpha_c = U @ (m[:, None] * UT_y) + yc
            alpha_d = (m * U ** 2).sum(1) + 1.0
            g = U @ (m * UT_1) + 1.0
        alpha_d = alpha_d - g * (1.0 / n)             # cancel the intercept direction
        looe = alpha_c / alpha_d[:, None]
        score = -(looe * looe).mean(0)                # higher = lower LOO error
        coef = V @ ((s / (s2 + alpha))[:, None] * UT_y)
        if best_score is None:
            best_score, best_coef = score, coef
        else:
            upd = score > best_score
            best_coef[:, upd] = coef[:, upd]
            best_score[upd] = score[upd]
    return best_coef


def evaluate_on_split(model, split, dataset: Dataset) -> dict:
    """Evaluate ``model`` on an already-loaded ``split`` (the reusable core).

    Times only the model + ridge readout (not the one-off data load), so callers
    that load a split once can evaluate many proposals without re-loading. The
    ridge readout (per-target GCV fit + Pearson scoring) runs on the GPU.
    """
    X_train, X_test, Y_train, Y_test = split
    device = _readout_device()
    t0 = time.time()
    # Move the stimulus to the readout device first so the model's forward pass
    # (the stimulus -> activation matmul) also runs on the GPU, not just the readout.
    if torch.is_tensor(X_train):
        X_train, X_test = X_train.to(device), X_test.to(device)
    A_train = _as_f64(model(X_train), device)
    A_test = _as_f64(model(X_test), device)
    Y_train, Y_test = _as_f64(Y_train, device), _as_f64(Y_test, device)

    X_offset, y_offset = A_train.mean(0), Y_train.mean(0)
    coef = _ridge_gcv_coef(A_train - X_offset, Y_train - y_offset)
    pred_train = (A_train - X_offset) @ coef + y_offset
    pred_test = (A_test - X_offset) @ coef + y_offset

    test_corr = _corr_per_column_t(pred_test, Y_test)
    train_corr = _corr_per_column_t(pred_train, Y_train).mean()
    return {
        "dataset": dataset.study,
        "model_name": model.name,
        "description": model.description,
        "n_params": model.n_params,
        "test_corr": float(test_corr.mean()),
        "train_corr": float(train_corr),
        "frac_targets_above_0.1": float((test_corr > 0.1).double().mean()),
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
    }


def evaluate(model, dataset: Dataset) -> dict:
    """Load the dataset split and evaluate ``model`` on it.

    ``seconds`` covers the whole from-scratch evaluation (data load + readout);
    use :func:`evaluate_on_split` to reuse a pre-loaded split and time only the
    readout.
    """
    t0 = time.time()
    row = evaluate_on_split(model, load_split(dataset), dataset)
    row["seconds"] = round(time.time() - t0, 2)
    return row


def save_results(row: dict, results_dir: str = "runs") -> None:
    """Upsert ``row`` into overall_results.csv keyed by (dataset, model_name) and plot."""
    os.makedirs(results_dir, exist_ok=True)
    csv = os.path.join(results_dir, "overall_results.csv")
    df = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    key = (df["dataset"] == row["dataset"]) & (df["model_name"] == row["model_name"]) \
        if not df.empty else pd.Series([], dtype=bool)
    if not df.empty and key.any():
        df = df[~key]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = df["dataset"] + " / " + df["model_name"]
    plt.figure(figsize=(max(4, len(df)), 4))
    plt.bar(labels, df["test_corr"])
    plt.ylabel("test_corr")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_corr.png"))
    plt.close()
