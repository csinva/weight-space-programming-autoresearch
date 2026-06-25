"""Fast per-target ridge readout -- a drop-in for ``harness.reward._ridge_predict``.

WHY: himalaya's ``RidgeCV(cv=5)`` recomputes a GPU SVD (cuSOLVER, ~12s on a 9000x8192 matrix)
once PER FOLD, so the readout costs ~60s and -- with no CUDA MPS -- three concurrent evals
serialize to ~3x that. GPU SVD is the entire cost; a matmul + symmetric eigendecomposition is
~25x cheaper.

WHAT: one ``eigh`` of the SMALLER gram (AtA when p<=n, else AAt) plus closed-form leave-one-out
(LOO) alpha selection per target. This is the SAME efficient-LOO scheme scikit-learn's ``RidgeCV``
uses by default; it is NOT a numerical reproduction of himalaya's 5-fold output -- the reward
``test_corr`` agrees to within ~5e-4, which is far below any model-to-model delta the search cares
about. (Bit-exact per-prediction reproduction of himalaya is impossible without its SVD: different
decompositions + per-target alpha ties differ element-wise even when the mean correlation matches.)

Dataset/region-agnostic: it only ever sees A (n x p) activations and Y (n x t) targets, so it works
unchanged for NSD (n~p, primal), TVSD (n>>p, primal), NNN (n<p when the rep is wide, dual), and the
NeuralSet text/meg/ieeg studies.
"""
from __future__ import annotations

import numpy as np
import torch


def fast_ridge_predict(A_train, A_test, Y_train, alphas=None, device=None, eps=1e-6):
    """Fit per-target LOO ridge train->Y and predict BOTH splits.

    Mirrors the signature/return of ``harness.reward._ridge_predict``: returns
    ``(pred_train, pred_test)`` as float64 numpy arrays. ``alphas`` defaults to the harness grid
    ``np.logspace(-3, 6, 10)``. Intercept is fit (inputs centered by the train mean).
    """
    if alphas is None:
        alphas = np.logspace(-3, 6, 10)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float64

    def _t(x):
        x = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
        return x.to(device=device, dtype=dtype)

    A, At, Y = _t(A_train), _t(A_test), _t(Y_train)
    a = torch.as_tensor(np.asarray(alphas), device=device, dtype=dtype)

    Am = A.mean(0, keepdim=True)
    Ym = Y.mean(0, keepdim=True)
    Ac, Yc = A - Am, Y - Ym
    n, p = Ac.shape
    t = Yc.shape[1]
    best_mse = torch.full((t,), float("inf"), device=device)
    best_a = torch.empty(t, device=device)

    if p <= n:
        # ---- primal: eigh of AtA (p x p) ----
        s, V = torch.linalg.eigh(Ac.T @ Ac)
        s = s.clamp_min(0)
        C = V.T @ (Ac.T @ Yc)            # p x t
        B = Ac @ V                       # n x p  (train samples in eigenbasis)
        B2 = B * B
        for al in a:
            d = 1.0 / (s + al)
            yhat = (B * d) @ C
            h = B2 @ d
            resid = (Yc - yhat) / (1.0 - h).clamp_min(eps).unsqueeze(1)
            mse = (resid * resid).mean(0)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_a = torch.where(better, al, best_a)
        DC = (1.0 / (s.unsqueeze(1) + best_a.unsqueeze(0))) * C       # p x t
        pred_tr = B @ DC + Ym
        pred_te = (At - Am) @ (V @ DC) + Ym
    else:
        # ---- dual: eigh of AAt (n x n), cheaper when the representation is wider than the sample ----
        s, U = torch.linalg.eigh(Ac @ Ac.T)
        s = s.clamp_min(0)
        M = U.T @ Yc                     # n x t
        U2 = U * U
        P = ((At - Am) @ Ac.T) @ U       # m x n  (test kernel in eigenbasis)
        for al in a:
            d = s / (s + al)
            yhat = (U * d) @ M
            h = U2 @ d
            resid = (Yc - yhat) / (1.0 - h).clamp_min(eps).unsqueeze(1)
            mse = (resid * resid).mean(0)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_a = torch.where(better, al, best_a)
        E = M / (s.unsqueeze(1) + best_a.unsqueeze(0))               # n x t
        pred_tr = U @ (E * s.unsqueeze(1)) + Ym
        pred_te = P @ E + Ym

    return (pred_tr.double().cpu().numpy(), pred_te.double().cpu().numpy())
