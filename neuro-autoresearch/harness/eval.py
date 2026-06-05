"""Fixed reward + result saving. The agent should NOT edit this file.

Reward = held-out predictivity of a ridge readout from model activations to
recorded neural activity. Only the ridge readout is fit; the model is never trained.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import Dataset
from data import load_split


def _corr_per_column(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = y_hat - y_hat.mean(0)
    b = y - y.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = np.divide((a * b).sum(0), den, out=np.zeros_like(den), where=den > 0)
    return r


def predictivity(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean per-column Pearson correlation."""
    return float(_corr_per_column(np.asarray(y_hat), np.asarray(y)).mean())


def evaluate(model, dataset: Dataset) -> dict:
    t0 = time.time()
    X_train, X_test, Y_train, Y_test = load_split(dataset)
    A_train = np.asarray(model(X_train), dtype="float64")
    A_test = np.asarray(model(X_test), dtype="float64")
    Y_train, Y_test = Y_train.numpy(), Y_test.numpy()

    ridge = Ridge(alpha=1.0).fit(A_train, Y_train)
    test_corr = _corr_per_column(ridge.predict(A_test), Y_test)
    return {
        "dataset": dataset.study,
        "model_name": model.name,
        "description": model.description,
        "n_params": model.n_params,
        "test_corr": float(test_corr.mean()),
        "train_corr": predictivity(ridge.predict(A_train), Y_train),
        "frac_targets_above_0.1": float((test_corr > 0.1).mean()),
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
    }


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
