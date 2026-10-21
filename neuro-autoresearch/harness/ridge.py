"""Legacy language/archival ridge readout; active vision uses vendor.looped.metrics.encoding.

``ridge_predict(A_train, A_test, Y_train)`` fits a per-target ridge from activations to neural
targets and returns ``(pred_train, pred_test)`` as float64 numpy arrays. This IS the eval: only the
ridge is fit, the model is never trained. ``NEURO_READOUT`` picks the implementation:

  * ``"fast"`` (default) -- one symmetric eigendecomposition of the smaller gram (``AtA`` when
    p<=n, else ``AAt``) plus closed-form leave-one-out per-target alpha selection (scikit-learn's
    efficient-LOO scheme). ~20x faster than himalaya on a GPU because it avoids himalaya's per-fold
    cuSOLVER SVD (~12s each, the entire old readout cost). Dataset/region-agnostic.
  * ``"himalaya"`` -- the reference Gallant-lab ``RidgeCV`` (per-target alpha via 5-fold CV,
    ``torch_cuda`` backend). The original ground truth; kept for verification. Their ``test_corr``
    agree to ~1e-4 (bit-exactness is impossible: different decompositions + per-target alpha ties).
"""
from __future__ import annotations

import os

import numpy as np
import torch

ALPHAS = np.logspace(-3, 6, 10)   # per-target penalty grid (both methods search this)
CV = 5                            # himalaya RidgeCV fold count


def _readout_device() -> torch.device:
    """The A100 when present, else CPU. The readout is identical on either."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_np(a) -> np.ndarray:
    """Any tensor/array -> a float64 numpy array (the readout's common return type)."""
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def ridge_predict(A_train, A_test, Y_train, alphas=None, device=None):
    """Fit per-target ridge train->Y, predict BOTH splits. Returns float64 numpy ``(pred_tr, pred_te)``.

    Dispatches on ``NEURO_READOUT`` (default ``"fast"``). See the module docstring.
    """
    if alphas is None:
        alphas = ALPHAS
    if device is None:
        device = _readout_device()
    if os.environ.get("NEURO_READOUT", "fast").lower() == "himalaya":
        return _himalaya_predict(A_train, A_test, Y_train, alphas, device)
    return _fast_predict(A_train, A_test, Y_train, alphas, device)


def _himalaya_predict(A_train, A_test, Y_train, alphas, device):
    """Reference himalaya ``RidgeCV`` (per-target alpha, 5-fold CV). float32 on CUDA, else float64."""
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV

    cuda = device.type == "cuda"
    set_backend("torch_cuda" if cuda else "torch")
    dtype = torch.float32 if cuda else torch.float64

    def _t(x):
        x = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
        return x.to(device=device, dtype=dtype)

    At, Ate, Yt = _t(A_train), _t(A_test), _t(Y_train)
    est = RidgeCV(alphas=alphas, fit_intercept=True, cv=CV)
    est.fit(At, Yt)
    return _to_np(est.predict(At)), _to_np(est.predict(Ate))


def _fast_predict(A_train, A_test, Y_train, alphas, device, eps=1e-6):
    """One eigendecomposition + closed-form leave-one-out per-target alpha selection.

    Eigendecomposes the SMALLER gram -- ``AtA`` (p x p) when p<=n (primal) else ``AAt`` (n x n)
    (dual) -- so cost tracks min(n, p): fast for both wide-rep/few-stimuli and the opposite.
    Intercept is fit (inputs centered by the train mean).
    """
    dtype = torch.float32 if device.type == "cuda" else torch.float64

    def _t(x):
        x = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
        return x.to(device=device, dtype=dtype)

    A, At, Y = _t(A_train), _t(A_test), _t(Y_train)
    a = torch.as_tensor(np.asarray(alphas), device=device, dtype=dtype)

    Am, Ym = A.mean(0, keepdim=True), Y.mean(0, keepdim=True)
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
            h = B2 @ d + 1.0 / n          # +1/n: the centering-fit intercept's leverage
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
            h = U2 @ d + 1.0 / n          # +1/n: the centering-fit intercept's leverage
            resid = (Yc - yhat) / (1.0 - h).clamp_min(eps).unsqueeze(1)
            mse = (resid * resid).mean(0)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_a = torch.where(better, al, best_a)
        E = M / (s.unsqueeze(1) + best_a.unsqueeze(0))               # n x t
        pred_tr = U @ (E * s.unsqueeze(1)) + Ym
        pred_te = P @ E + Ym

    return pred_tr.double().cpu().numpy(), pred_te.double().cpu().numpy()
