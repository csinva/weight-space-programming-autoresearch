"""Noise-ceiling scoring for NSD encoding, the field-standard way (Allen 2021 + Algonauts-2023).

NSD repeats each image n=3x. ``ncsnr`` (per voxel; provisioned by
``scripts/process_dataset/nsd_ncsnr.py``) is the noise-ceiling SNR. For a target that
averages n reps:

    NC_r2 = ncsnr^2 / (ncsnr^2 + 1/n)     # ceiling on R^2 (fraction of EXPLAINABLE variance)
    NC_r  = sqrt(NC_r2)                    # ceiling on Pearson r

This is the reference cross-check for ``harness.reward``: it reproduces the harness's reward metric,
the MEAN fraction-of-explainable-variance over RELIABLE voxels (sign-preserving):

    mean_v( sign(r_v) * r_v^2 / NC_r2_v )   for voxels with NC_r >= thresh

Sign-preserving so an anti-correlated voxel scores negative (matching ``reward._noise_normalize``).
The threshold drops voxels with essentially no ceiling.
"""
from __future__ import annotations

import numpy as np

N_REPS = 3
THRESH = 0.2          # min NC_r to count a voxel as reliable (~NC_r2 >= 0.04)


def noise_ceiling_r(ncsnr, n_reps=N_REPS):
    """Per-voxel ceiling on Pearson r for a response averaged over ``n_reps`` trials."""
    ncsnr = np.asarray(ncsnr, dtype=float)
    return np.sqrt(ncsnr ** 2 / (ncsnr ** 2 + 1.0 / n_reps))


def noise_normalized(test_corr, ncsnr, n_reps=N_REPS, thresh=THRESH):
    """(per-voxel r, per-voxel ncsnr) -> (mean sign(r)*r^2/NC_r2 over reliable voxels, n_reliable).

    Matches ``harness.reward._noise_normalize`` + the reward's mean over voxels, so this is a
    genuine cross-check of the harness number rather than a different statistic."""
    nc = noise_ceiling_r(ncsnr, n_reps)
    keep = nc >= thresh
    if not keep.any():
        return float("nan"), 0
    r = np.asarray(test_corr, float)[keep]
    nc_r2 = np.maximum(nc[keep] ** 2, 1e-12)
    return float(np.mean(np.sign(r) * r ** 2 / nc_r2)), int(keep.sum())


def per_roi(test_corr_full, roi_masks, ncsnr_full, n_reps=N_REPS, thresh=THRESH):
    """``{roi: (mean noise-normalized r^2/NC_r2, n_reliable)}`` over each ROI's reliable voxels."""
    return {roi: noise_normalized(test_corr_full[m], ncsnr_full[m], n_reps, thresh)
            for roi, m in roi_masks.items() if np.asarray(m, bool).any()}


def aligned_ncsnr(blob):
    """Per-voxel ncsnr in early+ventral order with the train-constant drop applied. Illustrative
    reference only -- it does NOT reproduce the loader's exact scored set: ``load_nsd`` applies the
    ``NSD_NCSNR_MIN`` reliability cut + the absolute ``_NC_R2_FLOOR`` *before* the constant-col drop,
    which this skips. ``noise_normalized`` re-applies its own ``thresh``, so the two converge for a
    spot-check but are not column-identical. Returns None if ncsnr was not provisioned."""
    if "ncsnr" not in blob:
        return None
    import torch

    from harness.loaders.shared import surviving_cols

    rois, nc = blob["rois"], blob["ncsnr"]
    Y = np.concatenate([np.asarray(rois["early"], "float32"),
                        np.asarray(rois["ventral"], "float32")], axis=1)
    tr = ~np.asarray(blob["shared"], bool)
    keep = surviving_cols(torch.as_tensor(Y[tr])).numpy()
    return np.concatenate([nc["early"], nc["ventral"]])[keep]
