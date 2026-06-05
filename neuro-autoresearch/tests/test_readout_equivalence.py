"""Equivalence guard for the GPU ridge readout in harness/eval.py.

eval.py reimplements sklearn's ``RidgeCV(alpha_per_target=True)`` per-target
leave-one-out GCV in torch so the reward runs on the GPU. That readout IS the
reward signal in the fixed harness, so this test pins it to sklearn: on synthetic
data covering BOTH code branches (tall ``n>p`` and wide ``n<=p``) and the real
alpha grid, the held-out predictions must match sklearn's to a tight tolerance.

Run standalone (`python tests/test_readout_equivalence.py`) or via pytest.
"""
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
from eval import _ALPHAS, _ridge_gcv_coef  # noqa: E402


def _predict_torch(A_train, Y_train, A_test):
    """Reproduce eval.evaluate_on_split's centering + torch GCV solve on CPU."""
    At = torch.as_tensor(A_train, dtype=torch.float64)
    Yt = torch.as_tensor(Y_train, dtype=torch.float64)
    x_off, y_off = At.mean(0), Yt.mean(0)
    coef = _ridge_gcv_coef(At - x_off, Yt - y_off)
    A_te = torch.as_tensor(A_test, dtype=torch.float64)
    return ((A_te - x_off) @ coef + y_off).numpy()


def _check(n_train, p, m, n_test, seed=0, atol=1e-6):
    rng = np.random.default_rng(seed)
    A_train = rng.standard_normal((n_train, p))
    A_test = rng.standard_normal((n_test, p))
    # targets with real (sparse) signal + noise so per-target alphas vary
    W = rng.standard_normal((p, m)) * (rng.random((p, m)) < 0.5)
    Y_train = A_train @ W + 0.5 * rng.standard_normal((n_train, m))

    ref = RidgeCV(alphas=_ALPHAS, alpha_per_target=True, fit_intercept=True)
    ref.fit(A_train, Y_train)
    pred_ref = ref.predict(A_test)
    pred_torch = _predict_torch(A_train, Y_train, A_test)

    max_abs = float(np.abs(pred_torch - pred_ref).max())
    assert max_abs < atol, (
        f"readout diverged from sklearn (n={n_train}, p={p}, m={m}): "
        f"max|dpred|={max_abs:.2e} >= {atol:.0e}"
    )
    return max_abs


def test_readout_matches_sklearn_tall():
    """n > p: the branch that reconstructs the missing zero-eigenvalue directions."""
    assert _check(n_train=200, p=8, m=20, n_test=50) < 1e-6


def test_readout_matches_sklearn_wide():
    """n <= p: the reduced-SVD branch."""
    assert _check(n_train=12, p=25, m=15, n_test=8) < 1e-6


if __name__ == "__main__":
    tall = _check(n_train=200, p=8, m=20, n_test=50)
    wide = _check(n_train=12, p=25, m=15, n_test=8)
    print(f"OK: torch GPU readout matches sklearn RidgeCV  "
          f"(tall max|d|={tall:.2e}, wide max|d|={wide:.2e})")
