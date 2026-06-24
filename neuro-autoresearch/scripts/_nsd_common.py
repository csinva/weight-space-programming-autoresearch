"""Shared machinery for the standalone NSD reference scripts.

``ref_alexnet_nsd.py`` and ``nsd_signal_sweep.py`` both: extract frozen-AlexNet
fc7 features for each unique NSD image, build the canonical shared-1000
image-disjoint split, and read the features out with the harness ridge readout,
scored per visual region. That common code lived (duplicated) in both scripts; it
is collected here. Everything reuses the FIXED harness internals (the readout, the
ROI resolver, the split-cache dir) rather than re-implementing them, so the
numbers stay directly comparable to the loop's reward.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from harness.data import _cache_dir, _canonical_test_ids, _image_id, _roi_voxel_keep
from harness.reward import _as_f64, _corr_per_column, _ridge_predict

# The famous functional regions reported alongside the overall + ventral scores.
ROIS = ["V1", "hV4", "FFA", "PPA", "EBA", "VWFA"]


# --- AlexNet fc7 features -------------------------------------------------------
def _feature_cache_path() -> str:
    return os.path.join(_cache_dir(), "alexnet_fc7_nsd_subj01.npz")


def alexnet_fc7(segments, device, batch_size: int = 256, cache: bool = True) -> dict:
    """``{filepath: fc7 4096-d}`` for each unique image, optionally disk-cached.

    Reads images through NeuralSet (``trigger.read()``) and batches them through a
    frozen ImageNet-pretrained AlexNet truncated to fc7 (the 4096-d penultimate
    representation). When ``cache`` is set the feature matrix is memoised under the
    split-cache dir, so only the neural side varies across an offset sweep.
    """
    by_path = {}
    for s in segments:
        fp = getattr(s.trigger, "filepath", None)
        if fp is not None and fp not in by_path:
            by_path[fp] = s.trigger
    paths = sorted(by_path)

    cache_path = _feature_cache_path()
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cached = dict(zip(z["paths"], z["feats"]))
        if set(paths).issubset(cached):
            print(f"  AlexNet fc7: loaded {len(cached)} cached features from {cache_path}")
            return {p: cached[p] for p in paths}

    from torchvision.models import AlexNet_Weights, alexnet

    weights = AlexNet_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()            # resize 256 -> centre-crop 224 -> normalise
    model = alexnet(weights=weights).to(device).eval()
    # Truncate the classifier to fc7: keep [Dropout, Linear, ReLU, Dropout, Linear,
    # ReLU] -> the 4096-d post-ReLU representation, dropping the 1000-way head.
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:6])
    print(f"  AlexNet fc7 on {len(paths)} unique images (frozen, {device}) ...")

    feats: dict[str, np.ndarray] = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            imgs = torch.stack([preprocess(by_path[p].read().convert("RGB")) for p in chunk])
            for p, f in zip(chunk, model(imgs.to(device)).float().cpu().numpy()):
                feats[p] = f
            if (i // batch_size) % 10 == 0:
                print(f"    {min(i + batch_size, len(paths))}/{len(paths)} ({time.time() - t0:.0f}s)")
    print(f"  features ready in {time.time() - t0:.0f}s; dim={next(iter(feats.values())).shape[0]}")

    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, paths=np.array(paths, dtype=object),
                 feats=np.stack([feats[p] for p in paths]).astype("float32"))
        print(f"  cached features -> {cache_path}")
    return feats


def feature_matrix(feats: dict, uniq) -> np.ndarray:
    """Stack ``feats`` into an ``(n_unique, dim)`` matrix in ``uniq`` order.

    Raises if any unique image is missing a feature (a silent drop would
    misalign the rows against the neural targets).
    """
    missing = [p for p in uniq if p not in feats]
    if missing:
        raise SystemExit(f"{len(missing)} unique images had no AlexNet feature "
                         f"(e.g. {missing[:2]})")
    return np.stack([feats[p] for p in uniq]).astype("float32")


# --- regions + canonical split --------------------------------------------------
def region_masks(study, include_ventral: bool = True) -> dict:
    """``{region_name: per-voxel keep-mask over nsdgeneral}`` for the reported regions.

    The ventral visual stream (the NSD reward region) and each functional ROI in
    ``ROIS`` are resolved through the harness's own ``_roi_voxel_keep`` -- the same
    resolver the loop uses -- so there is no second copy of the atlas logic.
    """
    masks: dict[str, np.ndarray] = {}
    if include_ventral:
        masks["VENTRAL"] = _roi_voxel_keep(study, "ventral")[0]
    for roi in ROIS:
        keep, _ = _roi_voxel_keep(study, roi)
        if keep is not None:
            masks[roi] = keep
    return masks


def canonical_split(uniq, study) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx) for the canonical shared-1000 image-disjoint split."""
    shared = _canonical_test_ids(study)
    is_test = np.array([_image_id(s) in shared for s in uniq])
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    assert set(uniq[tr]).isdisjoint(set(uniq[te])), "train/test share images -> leakage"
    assert tr.size and te.size, "empty train or test split"
    return tr, te


# --- ridge readout --------------------------------------------------------------
@dataclass
class Readout:
    """Result of fitting the harness ridge readout on a feature matrix.

    ``test_vec`` is the per-target test Pearson r over the surviving (train-non
    -constant) targets; ``kept`` is the bool mask of those targets over the full
    target set; ``n_tr`` is the TRs per voxel (targets are voxel-major). The kept
    train/test tensors are retained for callers that need them (e.g. a
    cross-check against another solver).
    """

    test_vec: np.ndarray
    train_corr: float
    kept: np.ndarray
    n_tr: int
    A_tr: torch.Tensor
    A_te: torch.Tensor
    Y_tr: torch.Tensor
    Y_te: torch.Tensor

    def slice_corr(self, voxel_keep) -> tuple[float, int]:
        """(mean test_corr, n_targets) over the voxels in ``voxel_keep``.

        The per-voxel mask is repeated across the TRs and restricted to the kept
        target columns, mirroring how the targets are laid out.
        """
        m = np.repeat(voxel_keep, self.n_tr)[self.kept]
        return (float(self.test_vec[m].mean()) if m.any() else float("nan")), int(m.sum())

    @property
    def overall(self) -> float:
        return float(self.test_vec.mean())


def fit_readout(A: np.ndarray, Y, tr, te, device, n_tr: int) -> Readout:
    """Fit the harness himalaya RidgeCV readout of features ``A`` onto targets ``Y``.

    Drops train-constant targets, fits per-target ``RidgeCV`` (GPU) on ALL voxels jointly, and
    returns per-target train/test correlations -- the exact readout
    ``harness.reward.evaluate_on_split`` runs, so the score matches the loop's reward.
    """
    A_t = _as_f64(torch.as_tensor(A), device)
    Y_t = _as_f64(Y, device)
    A_tr, A_te = A_t[tr], A_t[te]
    Y_tr, Y_te = Y_t[tr], Y_t[te]

    kept = (Y_tr.std(0) > 0).cpu().numpy()           # drop train-constant targets
    keep_t = torch.from_numpy(kept).to(device)
    Y_tr, Y_te = Y_tr[:, keep_t], Y_te[:, keep_t]

    pred_tr, pred_te = _ridge_predict(A_tr, A_te, Y_tr, device)
    test_vec = _corr_per_column(pred_te, Y_te.cpu().numpy())
    train_corr = float(_corr_per_column(pred_tr, Y_tr.cpu().numpy()).mean())
    return Readout(test_vec, train_corr, kept, n_tr, A_tr, A_te, Y_tr, Y_te)
