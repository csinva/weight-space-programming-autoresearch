"""Fixed reward. The agent should NOT edit this file.

Reward = held-out predictivity of a ridge readout from model activations to
recorded neural activity. Only the ridge readout is fit; the model is never
trained. Persisting a scored row to disk lives in ``harness.results``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import neuralset as ns

from harness.datasets import Dataset
from harness.results import CORR_PREFIX, frozen_slices

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
    U2 = U * U                                        # reused across the alpha grid
    wide = n <= p

    best_score = best_coef = None
    for alpha in (float(a) for a in _ALPHAS):
        if wide:
            d = alpha / (s2 + alpha)
            alpha_c = U @ (d[:, None] * UT_y)
            alpha_d = (d * U2).sum(1)
            g = U @ (d * UT_1)
        else:  # n > p: reduced SVD misses (n-p) zero-eigenvalue directions
            m = alpha / (s2 + alpha) - 1.0
            alpha_c = U @ (m[:, None] * UT_y) + yc
            alpha_d = (m * U2).sum(1) + 1.0
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


def evaluate_on_split(model, split, dataset: Dataset, slices=()) -> dict:
    """Evaluate ``model`` on an already-loaded ``split`` (the reusable core).

    Times only the model + ridge readout (not the one-off data load), so callers
    that load a split once can evaluate many proposals without re-loading. The
    ridge readout (per-target GCV fit + Pearson scoring) runs on the GPU.

    ``slices`` is the agent-declared list of slice names (``LOG_SLICES``); for each
    resolvable name a ``corr_<name>`` field (mean per-target corr over that slice's
    targets) is added to the returned row. The scalar reward is unchanged.
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

    test_corr_vec = _corr_per_column_t(pred_test, Y_test)
    train_corr_vec = _corr_per_column_t(pred_train, Y_train)
    test_np = test_corr_vec.detach().cpu().numpy()
    train_np = train_corr_vec.detach().cpu().numpy()

    # The reward narrows the per-target scores to the chosen region (default
    # "nsdgeneral" = all voxels); the readout above still fit ALL voxels, and every
    # declared region is still logged below as corr_<name> extra information.
    study = _study_for(dataset)
    reward_mask = _slice_target_mask(study, dataset.reward_roi, test_np.shape[0])
    if reward_mask is None:                 # unresolved (e.g. text/meg) -> all targets
        reward_mask = np.ones(test_np.shape[0], dtype=bool)
    test_corr = float(test_np[reward_mask].mean())
    train_corr = float(train_np[reward_mask].mean())
    row = {
        "study": dataset.study,
        "model_name": model.name,
        "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
        "n_params": model.n_params,
        "n_train": int(Y_train.shape[0]),
        "n_test": int(Y_test.shape[0]),
        "n_targets": int(reward_mask.sum()),
        "test_corr": test_corr,
        "train_corr": train_corr,
        "gap": train_corr - test_corr,
        "frac_above_0.1": float((test_np[reward_mask] > 0.1).mean()),
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
        "reward_roi": dataset.reward_roi,
    }
    row.update(_slice_corrs(test_np, dataset, _effective_slices(dataset, slices, study), study))
    return row


def _effective_slices(dataset, slices, study):
    """The region panel to log this iteration (the ``corr_<name>`` feedback columns).

    Priority: (1) if the region's results CSV exists, its already-frozen columns are
    authoritative; (2) else, for a study with an ROI resolver (NSD), the harness's own
    FIXED region menu (``data._ROI_DEFS``) -- so every region's predictivity is logged
    automatically, no per-run declaration needed; (3) else (text/meg, no resolver) the
    caller's ``slices``.
    """
    frozen = frozen_slices(dataset.study, reward_roi=dataset.reward_roi)
    if frozen is not None:
        return frozen
    from harness.data import _ROI_DEFS
    return list(_ROI_DEFS) if hasattr(study, "_roi_path") else slices


def _study_for(d: Dataset):
    """Construct the (no-download) NeuralSet study used to resolve slice names."""
    path = d.path if d.path is not None else (None if d.download else ns.CACHE_FOLDER)
    return ns.Study(name=d.study, path=path)


def _slice_target_mask(study, name: str, n_targets: int):
    """Boolean mask (length ``n_targets``) selecting slice ``name``'s target columns.

    NSD: ``name`` is an ROI resolved by ``data._roi_voxel_keep`` to a per-voxel keep
    mask over the nsdgeneral voxels. The eval targets are those voxels in voxel-major
    order (``n_targets = n_voxels * n_tr``), so the per-voxel mask is repeated across
    the TRs. Returns ``None`` (the clean extension point) when the study has no ROI
    resolver, the name is unknown, or the targets do not map cleanly onto the voxels
    (a sub-ROI dataset or dropped columns).
    """
    if not hasattr(study, "_roi_path"):     # non-volumetric studies (text/meg): no resolver
        return None
    from harness.data import _roi_voxel_keep

    try:
        keep, n_voxels = _roi_voxel_keep(study, name)
    except (ValueError, FileNotFoundError):
        return None
    if keep is None:                         # whole-region name (e.g. nsdgeneral)
        return np.ones(n_targets, dtype=bool)
    if n_targets % n_voxels:                 # targets are not a clean multiple of the voxels
        return None
    return np.repeat(keep, n_targets // n_voxels)


def _slice_corrs(test_corr: np.ndarray, dataset: Dataset, slices, study=None) -> dict:
    """Mean per-target test-corr over each declared slice -> ``{f"corr_{name}": val}``.

    Names with no resolver for this study are skipped with a warning, leaving a clean
    extension point for modalities (text/meg) that declare no slices.
    """
    if not slices:
        return {}
    study = study if study is not None else _study_for(dataset)
    out = {}
    for name in slices:
        mask = _slice_target_mask(study, name, len(test_corr))
        if mask is None:
            warnings.warn(f"log slice {name!r} unresolved for study {dataset.study!r}; skipping")
            continue
        sel = test_corr[mask]
        out[f"{CORR_PREFIX}{name}"] = float(sel.mean()) if sel.size else float("nan")
    return out


def evaluate(model, dataset: Dataset, slices=()) -> dict:
    """Load the dataset split and evaluate ``model`` on it.

    ``seconds`` covers the whole from-scratch evaluation (data load + readout);
    use :func:`evaluate_on_split` to reuse a pre-loaded split and time only the
    readout.
    """
    t0 = time.time()
    from harness.data import load_split  # lazy: keeps score-only callers off the data/study stack
    row = evaluate_on_split(model, load_split(dataset), dataset, slices)
    row["seconds"] = round(time.time() - t0, 2)
    return row
