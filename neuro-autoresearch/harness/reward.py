"""Fixed reward. The agent should NOT edit this file.

Reward = held-out predictivity of a ridge readout from model activations to recorded neural
activity, on the dataset's reward region (NSD: the ventral visual stream). Only the ridge readout
is fit; the model is never trained. The readout is Gallant-lab `himalaya`'s GPU ``RidgeCV``
(per-target alpha via k-fold CV, ``torch_cuda`` backend) -- no custom ridge math. Persisting a
scored row to disk lives in ``harness.results``.
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import torch
import neuralset as ns

from harness.datasets import Dataset

# Ridge penalties searched per target by himalaya's RidgeCV (it picks the best alpha PER target
# via k-fold CV). A fixed alpha massively overfits small samples; per-target CV closes that gap.
_ALPHAS = np.logspace(-3, 6, 10)
_CV = 5                       # himalaya RidgeCV folds for per-target alpha selection
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


def _readout_device() -> torch.device:
    """The A100 when present, else CPU. The readout is identical on either."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_np(a) -> np.ndarray:
    """Any tensor/array -> a float64 numpy array (for the Pearson scoring)."""
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def _select_targets(Y, mask):
    """Select the reward-region columns of the neural targets ``Y`` (or all if ``mask`` is None)."""
    if mask is None:
        return Y
    return Y[:, mask]


def _ridge_predict(A_train, A_test, Y_train, device):
    """Fit himalaya RidgeCV (per-target alpha) train->Y and predict BOTH splits.

    Uses the GPU ``torch_cuda`` backend (float32) when on CUDA, else the ``torch`` backend
    (float64). Returns ``(pred_train, pred_test)`` as float64 numpy arrays. This is the entire
    readout -- no hand-rolled ridge math.
    """
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV

    cuda = device.type == "cuda"
    set_backend("torch_cuda" if cuda else "torch")
    dtype = torch.float32 if cuda else torch.float64

    def _t(x):
        x = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
        return x.to(device=device, dtype=dtype)

    At, Ate, Yt = _t(A_train), _t(A_test), _t(Y_train)
    est = RidgeCV(alphas=_ALPHAS, fit_intercept=True, cv=_CV)
    est.fit(At, Yt)
    return _to_np(est.predict(At)), _to_np(est.predict(Ate))


def evaluate_on_split(model, split, dataset: Dataset, slices=()) -> dict:
    """Evaluate ``model`` on an already-loaded ``split`` (the reusable core).

    Runs the model forward, then fits himalaya's GPU ``RidgeCV`` from the activations to the
    dataset's REWARD-REGION neural targets only (NSD: the ventral stream), and returns the mean
    per-target held-out Pearson r as ``test_corr``. Times only the model + readout (not the one-off
    data load). ``slices`` is accepted for back-compat and ignored. A representation wider than
    ``MAX_DOUT`` is rejected (raised -> recorded as a ``status='error'`` row by the safe wrapper).
    """
    X_train, X_test, Y_train, Y_test = split
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
    # exactly those targets. For studies with no ROI resolver (text/meg/table) the loader already
    # selected the targets, so we fit all of them.
    study = _study_for(dataset)
    mask = _slice_target_mask(study, dataset.reward_roi, int(Y_train.shape[1]))
    Yv_train, Yv_test = _select_targets(Y_train, mask), _select_targets(Y_test, mask)

    pred_train, pred_test = _ridge_predict(A_train, A_test, Yv_train, device)
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
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
        "reward_roi": dataset.reward_roi,
        "run_tag": dataset.run_tag,
    }


def _error_status(exc: BaseException) -> str:
    """A compact, single-line ``status`` value for a failed evaluation."""
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())
    return ("error: " + msg)[:300]


def _error_row(model, split, dataset: Dataset, exc: BaseException,
               seconds: float, slices=()) -> dict:
    """A durable ``status='error'`` row for a model/readout fault (NaN scores)."""
    _, _, Y_train, Y_test = split
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
        "seconds": round(seconds, 2),
        "status": _error_status(exc),
        "reward_roi": dataset.reward_roi,
        "run_tag": dataset.run_tag,
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


def _study_for(d: Dataset):
    """Construct the (no-download) NeuralSet study used to resolve the reward region.

    Table-loader and prepared-betas datasets are not NeuralSet studies and carry their region
    labels in the loader (TVSD) or as a per-ROI betas key (NSD-betas), not an ns.Study ROI
    resolver -- return None so the reward averages over the targets the loader already selected.
    """
    if getattr(d, "loader", "segmenter") != "segmenter":
        return None
    from harness.loaders.common import _study_path
    return ns.Study(name=d.study, path=_study_path(d))


def _slice_target_mask(study, name: str, n_targets: int):
    """Boolean mask (length ``n_targets``) selecting region ``name``'s target columns.

    NSD: ``name`` is an ROI resolved by ``common._roi_voxel_keep`` to a per-voxel keep mask over the
    nsdgeneral voxels; the eval targets are those voxels (voxel-major if multi-TR), so the per-voxel
    mask is repeated across the TRs. Returns ``None`` (the clean extension point) when the study has
    no ROI resolver, the name is unknown, or the targets do not map cleanly onto the voxels.
    """
    if not hasattr(study, "_roi_path"):      # non-volumetric studies (text/meg): no resolver
        return None
    from harness.loaders.common import _roi_voxel_keep

    try:
        keep, n_voxels = _roi_voxel_keep(study, name)
    except (ValueError, FileNotFoundError):
        return None
    if keep is None:                          # whole-region name (e.g. nsdgeneral)
        return None
    if n_targets % n_voxels:                  # targets are not a clean multiple of the voxels
        return None
    return np.repeat(keep, n_targets // n_voxels)


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
