"""NSD prepared-betas loader (``betas_fithrf_GLMdenoise_RR``, Algonauts-2023 style).

This is the FINE-GRAINED NSD encoding target -- the post-processed GLM amplitudes most NSD
encoding work uses -- NOT the coarse raw-timeseries HRF-peak average. Y is the per-ROI single-
trial betas from ``nsd_betas/per_subject/subjNN.pkl`` (z-scored within each scan session, the 3
image repeats averaged to one response per image), built by ``scripts/cross_subject/
build_nsd_betas.py``. X is the NSD images (by ``nsd_id`` from ``nsd_stimuli.hdf5``), [0,1] RGB
resized to ``imsize``. The split is the image-disjoint **shared-1000** (test) vs the rest (train).
``reward_roi`` picks the ROI (default ``ventral`` ~8.6k voxels). All 8 subjects via ``NEURO_SUBJECT``.

All raw I/O for this dataset (the betas pickle + the stimulus HDF5) is confined to this module.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from harness.datasets import Dataset
from harness.loaders.shared import (
    _cache_dir, drop_constant_cols_with_rois, processed_dir, surviving_cols,
)


def _subject(d: Dataset) -> str:
    """Two-digit subject id (1-8), from NEURO_SUBJECT or the preset's ``subject`` (default 01)."""
    s = os.environ.get("NEURO_SUBJECT") or d.subject or "01"
    return f"{int(s):02d}"


def _stimuli_path(d: Dataset) -> str:
    path = os.path.join(d.path, "nsd_stimuli.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NSD stimuli not found at {path!r} (the 73k-image imgBrick HDF5 must live next to the "
            f"betas, e.g. {os.path.join(d.path, 'nsd_stimuli.hdf5')}).")
    return path


def _build_X(d: Dataset, nsd_ids: np.ndarray, subject: str) -> np.ndarray:
    """[0,1] RGB pixels of each image (by nsd_id) resized to ``imsize`` -> (N, imsize*imsize*3),
    in the same H,W,C-flattened layout as ``RawPixels``. Cached by (subject, imsize)."""
    cache = os.path.join(_cache_dir(), f"nsdbetas_X_subj{subject}_im{d.imsize}.pt")
    if not os.environ.get("NEURO_NO_CACHE") and os.path.exists(cache):
        return torch.load(cache, weights_only=False)["X"]
    import h5py

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    im = d.imsize
    pdir = processed_dir("nsd")
    images_h5 = os.path.join(pdir, "stimuli", "images.h5")
    if os.path.exists(images_h5):                              # in-repo store: one (425,425,3) per id
        with h5py.File(images_h5, "r") as h5:
            X = np.empty((len(nsd_ids), im * im * 3), dtype="float32")
            for s in range(0, len(nsd_ids), 1000):
                ids = nsd_ids[s:s + 1000]
                imgs = np.stack([h5[str(int(i))][:] for i in ids])  # (b, 425, 425, 3) uint8
                t = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255.0
                t = F.interpolate(t, size=(im, im), mode="bilinear", align_corners=False)
                X[s:s + 1000] = t.permute(0, 2, 3, 1).reshape(t.shape[0], -1).cpu().numpy()
    else:
        with h5py.File(_stimuli_path(d), "r") as h5:
            brick = h5["imgBrick"]                                  # (73000, 425, 425, 3) uint8
            # NSD ids are 1-based; imgBrick is 0-based -> row = nsd_id - 1 (matches store_stimuli).
            X = np.empty((len(nsd_ids), im * im * 3), dtype="float32")
            order = np.argsort(nsd_ids)                             # h5 fancy-index needs increasing rows
            for s in range(0, len(order), 1000):
                idx = order[s:s + 1000]
                rows = (nsd_ids[idx] - 1).astype(int)
                sb = np.argsort(rows)
                imgs = brick[np.sort(rows)]                         # (b, 425, 425, 3)
                t = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255.0
                t = F.interpolate(t, size=(im, im), mode="bilinear", align_corners=False)
                X[idx[sb]] = t.permute(0, 2, 3, 1).reshape(t.shape[0], -1).cpu().numpy()
    if not os.environ.get("NEURO_NO_CACHE"):
        os.makedirs(_cache_dir(), exist_ok=True)
        torch.save({"X": X}, cache, pickle_protocol=4)          # ~1-4 GB at imsize 128
    return X


_NSD_FINE = ("V1", "V2", "V3", "hV4", "FFA", "PPA", "VWFA")


def _nsd_roi_masks(rois, Y, n_early):
    """ROI masks over the early+ventral region columns: the two streams (disjoint -> a clean split
    at ``n_early``) plus the functional patches, byte-matched to their in-region voxels (a patch's
    out-of-stream part is not in Y, so it is simply not scored).

    The voxel<->column join is by the column's exact bytes. The hash only buckets candidates -- the
    full bytes are then compared -- so a 64-bit hash collision can't mismatch, and byte-identical
    voxels (duplicates/all-zero) each get marked rather than one silently stealing the other."""
    from collections import defaultdict

    n = Y.shape[1]
    cols = np.arange(n)
    masks = {"early": cols < n_early, "ventral": cols >= n_early}
    index: dict = defaultdict(list)
    for j in range(n):
        index[hash(Y[:, j].tobytes())].append(j)
    for name in _NSD_FINE:
        m = np.zeros(n, dtype=bool)
        M = np.asarray(rois[name], dtype="float32")
        for j in range(M.shape[1]):
            col = M[:, j].tobytes()
            for k in index.get(hash(col), ()):
                if Y[:, k].tobytes() == col:
                    m[k] = True
        masks[name] = m
    return masks


def _ncsnr_select(blob, Ytr, Yte, roi_masks):
    """Per-voxel NSD noise ceiling: keep only reliable voxels AND return the ceiling for scoring.

    NSD ships a per-voxel ncsnr (provisioned into ``blob['ncsnr']`` by scripts.process_dataset.nsd,
    aligned column-for-column to the early+ventral betas). From it the averaged-over-n R^2 ceiling
    is ``NC_r2 = ncsnr^2 / (ncsnr^2 + 1/n)`` (n=3 reps), whose sqrt ``NC_r`` is the Pearson-r ceiling.
    Voxels with ``NC_r < NSD_NCSNR_MIN`` (operator knob; default 0.2, the field-standard reliability
    cut; set 0 to keep all) are dropped from Y AND every roi_mask, so each breakdown scores only its
    signal voxels. Returns ``(Ytr, Yte, roi_masks, nc_r2)`` with ``nc_r2`` aligned to the surviving
    columns -- the reward noise-normalizes per voxel by it (Algonauts r^2/NC_r2). ``nc_r2`` is None
    only for a legacy blob with no ncsnr when the cut is disabled."""
    thresh = float(os.environ.get("NSD_NCSNR_MIN", "0.2"))
    if "ncsnr" not in blob:
        if thresh > 0:
            raise RuntimeError(
                "NSD_NCSNR_MIN is set but this NSD blob has no 'ncsnr' -- (re)run "
                "`python -m scripts.process_dataset.nsd` to fetch + align it.")
        return Ytr, Yte, roi_masks, None
    nc = blob["ncsnr"]
    ncsnr = np.concatenate([np.asarray(nc["early"], "float32"),
                            np.asarray(nc["ventral"], "float32")])
    assert ncsnr.size == Ytr.shape[1], (
        f"ncsnr ({ncsnr.size}) misaligned to early+ventral Y ({Ytr.shape[1]})")
    nc_r2 = ncsnr ** 2 / (ncsnr ** 2 + 1.0 / 3)               # R^2 ceiling, n=3 repeats
    keep = np.sqrt(nc_r2) >= thresh                           # all-True when thresh <= 0
    print(f"  NSD ncsnr select: NC_r>={thresh} keeps {int(keep.sum())}/{keep.size} voxels")
    keep_t = torch.as_tensor(keep)
    roi_masks = {k: np.asarray(v, bool)[keep] for k, v in roi_masks.items()}
    return Ytr[:, keep_t], Yte[:, keep_t], roi_masks, nc_r2[keep]


def load_nsd(d: Dataset):
    """NSD GLMdenoise betas: reward = the joint EARLY+VENTRAL stream; X from the images, shared-1000
    split. Returns (X_train, X_test, Y_train, Y_test, roi_masks, nc). ``roi_masks`` carries the
    per-ROI breakdown (early, ventral + V1/V2/V3/hV4/FFA/PPA/VWFA); ``nc`` is the per-voxel R^2 noise
    ceiling (aligned to the scored columns) the reward normalizes by. Train-constant voxels dropped."""
    subject = _subject(d)
    pdir = processed_dir("nsd")
    pt = os.path.join(pdir, "neural", f"subj{subject}.pt")
    if os.path.isdir(pdir) and os.path.exists(pt):             # in-repo repacked blob
        blob = torch.load(pt, weights_only=False)
    else:
        pkl = os.path.join(d.path, "per_subject", f"subj{subject}.pkl")
        with open(pkl, "rb") as f:
            blob = pickle.load(f)
    rois = blob["rois"]
    early = np.asarray(rois["early"], dtype="float32")
    Y = np.concatenate([early, np.asarray(rois["ventral"], dtype="float32")], axis=1)
    roi_masks = _nsd_roi_masks(rois, Y, early.shape[1])
    nsd_ids = np.asarray(blob["nsd_ids"])
    shared = np.asarray(blob["shared"], dtype=bool)            # the shared-1000 test images

    X = _build_X(d, nsd_ids, subject)
    tr, te = ~shared, shared
    Xtr, Xte = torch.as_tensor(X[tr]), torch.as_tensor(X[te])
    Ytr, Yte = torch.as_tensor(Y[tr]), torch.as_tensor(Y[te])
    Ytr, Yte, roi_masks, nc = _ncsnr_select(blob, Ytr, Yte, roi_masks)
    if nc is not None:                                         # ride the same constant-col drop as Y
        keep = surviving_cols(Ytr)
        nc = nc[keep.numpy() if torch.is_tensor(keep) else np.asarray(keep)]
    Ytr, Yte, roi_masks = drop_constant_cols_with_rois(Ytr, Yte, roi_masks)
    print(f"  NSD-betas subj{subject}: {Xtr.shape[0]} train / {Xte.shape[0]} test images; "
          f"region=early+ventral -> {Ytr.shape[1]} voxels ({Xtr.shape[1]} stim dims)")
    return Xtr, Xte, Ytr, Yte, roi_masks, nc
