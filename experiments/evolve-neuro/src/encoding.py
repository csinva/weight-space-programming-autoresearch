"""Bootstrapped ridge regression for fMRI encoding.

Simplified from `neuro/encoding/{ridge,fit,eval}.py`. Per-voxel ridge with the
regularization strength chosen by bootstrapped cross-validation on the training
set, then evaluated by correlation on held-out test responses.
"""
import itertools as itools

import numpy as np


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


def bootstrap_ridge(
        stim_train, resp_train, stim_test, resp_test, alphas,
        nboots=5, chunklen=40, nchunks=20, singcutoff=1e-10):
    """Fit per-voxel ridge, picking alpha by bootstrapped CV. Returns (wt, corrs, alphas_best)."""
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

    # in-sample training correlation (predict the train set with the fitted weights)
    corrs_train = _per_voxel_corr(resp_train, stim_train @ wt)

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
