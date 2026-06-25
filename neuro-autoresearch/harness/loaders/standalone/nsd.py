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
from harness.loaders.shared import _cache_dir, drop_constant_cols


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
    with h5py.File(_stimuli_path(d), "r") as h5:
        brick = h5["imgBrick"]                                  # (73000, 425, 425, 3) uint8
        off = 1 if int(nsd_ids.max()) >= brick.shape[0] else 0  # NSD ids are 1-based
        X = np.empty((len(nsd_ids), im * im * 3), dtype="float32")
        order = np.argsort(nsd_ids)                             # h5 fancy-index needs increasing rows
        for s in range(0, len(order), 1000):
            idx = order[s:s + 1000]
            rows = (nsd_ids[idx] - off).astype(int)
            sb = np.argsort(rows)
            imgs = brick[np.sort(rows)]                         # (b, 425, 425, 3)
            t = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255.0
            t = F.interpolate(t, size=(im, im), mode="bilinear", align_corners=False)
            X[idx[sb]] = t.permute(0, 2, 3, 1).reshape(t.shape[0], -1).cpu().numpy()
    if not os.environ.get("NEURO_NO_CACHE"):
        os.makedirs(_cache_dir(), exist_ok=True)
        torch.save({"X": X}, cache, pickle_protocol=4)          # ~1-4 GB at imsize 128
    return X


def load_nsd(d: Dataset):
    """NSD GLMdenoise betas: Y from subjNN.pkl[reward_roi], X from the images, shared-1000 split.

    Returns (X_train, X_test, Y_train, Y_test) as 2-D float tensors -- the same contract as the
    other loaders. Train-constant target voxels are dropped (mirrors the readout's own guard).
    """
    subject = _subject(d)
    pkl = os.path.join(d.path, "per_subject", f"subj{subject}.pkl")
    with open(pkl, "rb") as f:
        blob = pickle.load(f)
    region = d.reward_roi if d.reward_roi in blob["rois"] else "ventral"
    Y = np.asarray(blob["rois"][region], dtype="float32")      # (N_images, n_vox)
    nsd_ids = np.asarray(blob["nsd_ids"])
    shared = np.asarray(blob["shared"], dtype=bool)            # the shared-1000 test images

    X = _build_X(d, nsd_ids, subject)
    tr, te = ~shared, shared
    Xtr, Xte = torch.as_tensor(X[tr]), torch.as_tensor(X[te])
    Ytr, Yte = torch.as_tensor(Y[tr]), torch.as_tensor(Y[te])
    Ytr, Yte = drop_constant_cols(Ytr, Yte)
    print(f"  NSD-betas subj{subject}: {Xtr.shape[0]} train / {Xte.shape[0]} test images; "
          f"region={region} -> {Ytr.shape[1]} voxels ({Xtr.shape[1]} stim dims)")
    return Xtr, Xte, Ytr, Yte
