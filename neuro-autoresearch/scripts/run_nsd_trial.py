"""Reproducible NSD autoresearch-trial runner.

Drives the fixed harness end-to-end on the active dataset (NSD fMRI), loading the
split ONCE and then, for each in-code model proposal, separately timing the model
forward pass, the ridge readout fit, and the ridge predict. The one-off data load
is the slow step (~9s); the linear (ridge) prediction is reported on its own so
its cost is visible apart from the load.

Run from the repo root:
    /usr/bin/python3.12 scripts/run_nsd_trial.py --iterations 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.linear_model import Ridge

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "harness"))
sys.path.insert(0, os.path.join(_ROOT, "evolve"))

from config import DATASET          # noqa: E402
from data import load_split         # noqa: E402
from eval import _corr_per_column, predictivity  # noqa: E402  (reuse harness logic)
from model import MODELS            # noqa: E402


def evaluate_model(model, split) -> dict:
    """Evaluate one model on an already-loaded split, timing each linear step."""
    X_train, X_test, Y_train, Y_test = split

    t = time.perf_counter()
    A_train = np.asarray(model(X_train), dtype="float64")
    A_test = np.asarray(model(X_test), dtype="float64")
    fwd_ms = (time.perf_counter() - t) * 1e3

    y_train, y_test = Y_train.numpy(), Y_test.numpy()

    t = time.perf_counter()
    ridge = Ridge(alpha=1.0).fit(A_train, y_train)
    ridge_fit_ms = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    pred_test = ridge.predict(A_test)
    ridge_predict_ms = (time.perf_counter() - t) * 1e3

    test_corr = float(_corr_per_column(pred_test, y_test).mean())
    train_corr = predictivity(ridge.predict(A_train), y_train)

    return {
        "model_name": model.name,
        "n_params": model.n_params,
        "test_corr": test_corr,
        "train_corr": train_corr,
        "fwd_ms": fwd_ms,
        "ridge_fit_ms": ridge_fit_ms,
        "ridge_predict_ms": ridge_predict_ms,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=2,
                   help="number of model proposals to evaluate (cycles MODELS)")
    iterations = p.parse_args().iterations
    if iterations < 1:
        p.error("--iterations must be >= 1")

    names = list(MODELS)
    proposals = [names[i % len(names)] for i in range(iterations)]

    print(f"Dataset: {DATASET.study} ({DATASET.neural}), "
          f"imsize={DATASET.imsize}, test_frac={DATASET.test_frac}, seed={DATASET.seed}")
    print(f"Loading split once via harness (this is the slow step) ...")

    t = time.perf_counter()
    split = load_split(DATASET)
    data_load_s = time.perf_counter() - t

    X_train, X_test, Y_train, Y_test = split
    print(f"Loaded: X_train{tuple(X_train.shape)} X_test{tuple(X_test.shape)} "
          f"Y_train{tuple(Y_train.shape)} Y_test{tuple(Y_test.shape)}\n")

    rows = []
    for i, name in enumerate(proposals, start=1):
        rows.append({"iter": i, **evaluate_model(MODELS[name](), split)})

    print("Per-iteration results")
    print(f"{'iter':>4}  {'model':<18} {'n_params':>8}  {'test_corr':>9}  {'train_corr':>10}")
    print("-" * 60)
    for r in rows:
        print(f"{r['iter']:>4}  {r['model_name']:<18} {r['n_params']:>8}  "
              f"{r['test_corr']:>9.4f}  {r['train_corr']:>10.4f}")

    print("\nTiming breakdown")
    print(f"  data_load_s (once): {data_load_s:8.3f} s")
    print(f"{'':2}{'model':<18} {'fwd_ms':>10}  {'ridge_fit_ms':>13}  {'ridge_predict_ms':>17}")
    print("-" * 64)
    for r in rows:
        print(f"{'':2}{r['model_name']:<18} {r['fwd_ms']:>10.3f}  "
              f"{r['ridge_fit_ms']:>13.3f}  {r['ridge_predict_ms']:>17.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
