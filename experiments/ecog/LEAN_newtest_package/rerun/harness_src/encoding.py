"""Bootstrapped ridge regression for fMRI encoding.

Simplified from `neuro/encoding/{ridge,fit,eval}.py`. Per-voxel ridge with the
regularization strength chosen by bootstrapped cross-validation on the training
set, then evaluated by correlation on held-out test responses.

Set env ECOG_RIDGE_GPU=1 (with CUDA available) to run the linear algebra on GPU
via the XtX eigendecomposition (mathematically identical to the SVD path, but the
huge N x p design matrix is never held on the GPU — XtX/XtY are streamed in row
blocks, so it is memory-safe on any GPU and ~10-20x faster than the CPU SVD).
"""
import itertools as itools
import os

import numpy as np
import torch


def _zs(v):
    return (v - v.mean(0)) / v.std(0)


def _mult_diag(d, mtx, left=True):
    """Multiply by a diagonal matrix given as a 1-D vector `d` (fast)."""
    if left:
        return (d * mtx.T).T
    return d * mtx


def _temporal_chunk_splits(num_splits, num_examples, chunk_len, num_chunks, seed=42):
    rng = np.random.RandomState(seed)
    all_idx = range(num_examples)
    chunks = list(zip(*[iter(all_idx)] * chunk_len))
    splits = []
    for _ in range(num_splits):
        rng.shuffle(chunks)
        tune = list(itools.chain(*chunks[:num_chunks]))
        train = list(set(all_idx) - set(tune))
        splits.append((train, tune))
    return splits


def _ridge_weights(stim, resp, alpha, singcutoff=1e-10):
    """Closed-form ridge weights for `stim @ wt ~= resp` with per-voxel alpha."""
    U, S, Vh = np.linalg.svd(stim, full_matrices=False)
    UR = U.T @ np.nan_to_num(resp)
    if np.isscalar(alpha):
        alpha = np.ones(resp.shape[1]) * alpha
    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in np.unique(alpha):
        sel = np.nonzero(alpha == ua)[0]
        wt[:, sel] = Vh.T @ np.diag(S / (S ** 2 + ua ** 2)) @ UR[:, sel]
    return wt


def _ridge_corr(stim_train, stim_test, resp_train, resp_test, alphas, singcutoff=1e-10):
    """Correlation between predicted and actual `resp_test` for each alpha (no weights)."""
    U, S, Vh = np.linalg.svd(stim_train, full_matrices=False)
    ngood = np.sum(S > singcutoff)
    U, S, Vh = U[:, :ngood], S[:ngood], Vh[:ngood]
    UR = U.T @ resp_train
    PVh = stim_test @ Vh.T
    zPresp = _zs(resp_test)
    corrs = []
    for a in alphas:
        D = S / (S ** 2 + a ** 2)
        pred = _mult_diag(D, PVh, left=False) @ UR
        Rcorr = (zPresp * _zs(pred)).mean(0)
        Rcorr[np.isnan(Rcorr)] = 0
        corrs.append(Rcorr)
    return corrs


def _use_gpu():
    # GPU ridge is the default when CUDA is available (validated identical to the
    # CPU SVD path, ~21x faster). Set ECOG_RIDGE_GPU=0 to force the CPU path.
    return os.environ.get("ECOG_RIDGE_GPU", "1") != "0" and torch.cuda.is_available()


# --------------------------- GPU path (XtX eigendecomposition) ---------------------------

def _accum_XtX_XtY(stim, resp, dev, block=None):
    """Stream row-blocks to GPU and accumulate XtX (p,p) and XtY (p,nvox), float64.

    Block size is bounded by feature dim p (a row-block is block x p float64) so it
    stays ~<=3 GB and large-p models don't OOM smaller GPUs.
    """
    p = stim.shape[1]
    if block is None:
        block = max(10_000, min(200_000, int(3.0e9 / (p * 8))))
    XtX = torch.zeros(p, p, dtype=torch.float64, device=dev)
    XtY = torch.zeros(p, resp.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, stim.shape[0], block):
        Xb = torch.from_numpy(np.ascontiguousarray(stim[i:i + block])).to(dev, torch.float64)
        Yb = torch.from_numpy(np.ascontiguousarray(resp[i:i + block])).to(dev, torch.float64)
        XtX += Xb.T @ Xb
        XtY += Xb.T @ Yb
        del Xb, Yb
    return XtX, XtY


def _eig_good(XtX, singcutoff):
    Lam, V = torch.linalg.eigh(XtX)            # XtX = V diag(Lam) V^T, Lam ascending
    S = torch.sqrt(torch.clamp(Lam, min=0.0))
    keep = S > singcutoff
    return V[:, keep], Lam[keep]


def _zs_t(v):
    s = v.std(0)
    s[s == 0] = 1.0
    return (v - v.mean(0)) / s


def bootstrap_ridge_gpu(stim_train, resp_train, stim_test, resp_test, alphas,
                        nboots=5, chunklen=40, nchunks=20, singcutoff=1e-10):
    """GPU ridge via XtX eigendecomposition. Returns (wt, corrs, alphas_best) like
    `bootstrap_ridge`. The full N x p design matrix is never resident on the GPU."""
    dev = "cuda"
    nresp, nvox = resp_train.shape
    nalphas = len(alphas)
    a2 = torch.tensor(np.asarray(alphas) ** 2, dtype=torch.float64, device=dev)  # (A,)

    XtX_full, XtY_full = _accum_XtX_XtY(stim_train, resp_train, dev)

    splits = _temporal_chunk_splits(nboots, nresp, chunklen, nchunks)
    all_corrs = torch.zeros(nalphas, nvox, dtype=torch.float64, device=dev)
    for train_idx, tune_idx in splits:
        Xtu = torch.from_numpy(np.ascontiguousarray(stim_train[tune_idx])).to(dev, torch.float64)
        Ytu = torch.from_numpy(np.ascontiguousarray(resp_train[tune_idx])).to(dev, torch.float64)
        XtX_sub = XtX_full - Xtu.T @ Xtu
        XtY_sub = XtY_full - Xtu.T @ Ytu
        V, Lam = _eig_good(XtX_sub, singcutoff)
        A = Xtu @ V                              # (ntu, k)
        B = V.T @ XtY_sub                        # (k, nvox)
        zYtu = _zs_t(Ytu)
        for ai in range(nalphas):
            pred = A @ (B / (Lam + a2[ai]).unsqueeze(1))   # (ntu, nvox)
            all_corrs[ai] += torch.nan_to_num((zYtu * _zs_t(pred)).mean(0))
        del Xtu, Ytu, XtX_sub, XtY_sub, V, Lam, A, B, zYtu
    # mean over boots == argmax of the sum
    best_alpha_idx = torch.argmax(all_corrs, dim=0).cpu().numpy()
    alphas_best = np.asarray(alphas)[best_alpha_idx]

    # final weights on full train: wt = V diag(1/(Lam+a^2)) V^T XtY_full
    V, Lam = _eig_good(XtX_full, singcutoff)
    B = V.T @ XtY_full                           # (k, nvox)
    wt = torch.zeros(stim_train.shape[1], nvox, dtype=torch.float64, device=dev)
    for ua in np.unique(alphas_best):
        sel = np.nonzero(alphas_best == ua)[0]
        scaled = B[:, sel] / (Lam + float(ua) ** 2).unsqueeze(1)
        wt[:, sel] = V @ scaled

    # test correlations on the held-out test set (small)
    Xte = torch.from_numpy(np.ascontiguousarray(stim_test)).to(dev, torch.float64)
    Yte = torch.from_numpy(np.ascontiguousarray(resp_test)).to(dev, torch.float64)
    pred = torch.nan_to_num(Xte @ wt)
    corrs = torch.nan_to_num((_zs_t(Yte) * _zs_t(pred)).mean(0))
    return wt.cpu().numpy(), corrs.cpu().numpy(), alphas_best


def bootstrap_ridge(
        stim_train, resp_train, stim_test, resp_test, alphas,
        nboots=5, chunklen=40, nchunks=20, singcutoff=1e-10):
    """Fit per-voxel ridge, picking alpha by bootstrapped CV. Returns (wt, corrs, alphas_best)."""
    if _use_gpu():
        return bootstrap_ridge_gpu(stim_train, resp_train, stim_test, resp_test, alphas,
                                   nboots=nboots, chunklen=chunklen, nchunks=nchunks,
                                   singcutoff=singcutoff)
    nresp, nvox = resp_train.shape
    splits = _temporal_chunk_splits(nboots, nresp, chunklen, nchunks)

    boot_corrs = []
    for train_idx, tune_idx in splits:
        boot_corrs.append(_ridge_corr(
            stim_train[train_idx], stim_train[tune_idx],
            resp_train[train_idx], resp_train[tune_idx],
            alphas, singcutoff=singcutoff))
    # (alphas, voxels, boots)
    all_corrs = np.dstack(boot_corrs)
    best_alpha_idx = np.argmax(all_corrs.mean(2), axis=0)
    alphas_best = alphas[best_alpha_idx]

    # fit on full training set with the chosen alphas, predict test set
    wt = _ridge_weights(stim_train, resp_train, alphas_best, singcutoff=singcutoff)
    pred = np.nan_to_num(stim_test @ wt)
    corrs = np.array([
        np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(nvox)
    ])
    corrs = np.nan_to_num(corrs)
    return wt, corrs, alphas_best


def _per_voxel_corr(actual, pred, chunk=20000):
    """Per-voxel correlation between two (T, M) matrices (chunked to bound memory)."""
    az = _zs(np.nan_to_num(actual))
    out = np.zeros(actual.shape[1])
    for i in range(0, actual.shape[1], chunk):
        out[i:i + chunk] = (az[:, i:i + chunk] * _zs(np.nan_to_num(pred[:, i:i + chunk]))).mean(0)
    return np.nan_to_num(out)


def fit_encoding(stim_train, resp_train, stim_test, resp_test,
                 nboots=5, chunklen=40, nchunks=20):
    """Fit the encoding model and return a results dict of correlation summaries."""
    alphas = np.logspace(1, 4, 12)
    wt, corrs, alphas_best = bootstrap_ridge(
        stim_train, resp_train, stim_test, resp_test, alphas,
        nboots=nboots, chunklen=chunklen, nchunks=nchunks)

    # in-sample training correlation (predict the train set with the fitted weights).
    # Cast wt to float32 to avoid upcasting the (huge) stim_train matrix to float64
    # in the matmul temporary (purely a memory optimization; same math).
    _wt_f = wt.astype(np.float32) if (hasattr(stim_train, 'dtype') and stim_train.dtype == np.float32) else wt
    corrs_train = _per_voxel_corr(resp_train, stim_train @ _wt_f)

    return {
        'weights': wt,
        'alphas_best': alphas_best,
        'corrs_test': corrs,
        'corrs_train': corrs_train,
        'corrs_train_mean': float(np.nanmean(corrs_train)),
        'corrs_test_mean': float(np.nanmean(corrs)),
        'corrs_test_median': float(np.nanmedian(corrs)),
        'corrs_test_frac>0': float(np.nanmean(corrs > 0)),
        'corrs_test_frac>0.2': float(np.nanmean(corrs > 0.2)),
        'corrs_test_mean_top1_percentile': float(
            np.nanmean(np.sort(corrs)[-len(corrs) // 100:])),
        'corrs_test_mean_top5_percentile': float(
            np.nanmean(np.sort(corrs)[-len(corrs) // 20:])),
    }
