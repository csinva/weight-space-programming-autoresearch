"""Shared machinery for the data loaders.

Stimulus extractor (``RawPixels``), the on-disk build cache, neural-target ROI
selection, and the repeat-collapse + canonical-split helpers -- used by BOTH the
NeuralSet/Segmenter path (``neuralset``) and the pre-averaged table loaders
(``tvsd``/``nnn``). The autoresearch agent should NOT need to edit this file.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess

import numpy as np
import torch
import neuralset as ns
from neuralset.extractors.image import BaseClassicImageExtractor

from harness.datasets import Dataset


class RawPixels(BaseClassicImageExtractor):
    """Flattened, [0,1]-normalised RGB pixels of each stimulus image."""

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        return (image.astype("float32") / 255.0).reshape(-1)


def _study_path(d):
    """Resolve the NeuralSet study path: explicit ``d.path`` if set, else ``None`` to
    download-from-scratch when ``d.download``, else the shared ``ns.CACHE_FOLDER``."""
    return d.path if d.path is not None else (None if d.download else ns.CACHE_FOLDER)


def _load_workers() -> int:
    """DataLoader workers for the per-segment neural reads (env NEURO_LOAD_WORKERS).

    The per-segment fMRI/MEG reads are I/O-bound, so parallel workers cut the load
    by ~10x (e.g. the NSD subj01 load drops from ~25 min to a few min). Defaults to
    16; override via NEURO_LOAD_WORKERS.
    """
    return int(os.environ.get("NEURO_LOAD_WORKERS", "16"))


# --- on-disk build cache --------------------------------------------------------
# The slow thing is _build_xy: minutes of per-segment neural reads to assemble the
# full (X, Y) over ALL loaded voxels. That output is independent of the cheap,
# downstream choices -- the target region (``roi``, a column subset) and the split
# fraction -- so it is cached keyed by every field EXCEPT those. The first call
# builds and saves it; later calls (including a DIFFERENT roi) load it from disk in
# seconds and just re-subset/re-split, never re-reading the timeseries.
_CACHE_VERSION = "v2"

# Fields that only affect post-build (cheap) steps, so they are excluded from the
# build-cache key: ``roi`` subsets columns; ``test_frac`` only sizes a temporal tail;
# ``reward_roi`` only selects which targets the scalar reward averages over; ``run_tag``
# only names the output CSV. All region/run loops therefore share ONE build (the
# timeseries is read only once).
_BUILD_AGNOSTIC_FIELDS = ("roi", "test_frac", "reward_roi", "run_tag")


def _cache_dir() -> str:
    return os.environ.get(
        "NEURO_SPLIT_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "neuro-autoresearch", "splits"),
    )


def _build_cache_key(d: Dataset) -> str:
    """A hash over the Dataset fields that affect _build_xy (all but roi/test_frac)."""
    fields = {k: v for k, v in dataclasses.asdict(d).items()
              if k not in _BUILD_AGNOSTIC_FIELDS}
    payload = json.dumps(fields, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{_CACHE_VERSION}:{payload}".encode()).hexdigest()[:16]
    return f"{d.study}_build_{digest}.pt"


def _build_cache_load(path: str):
    blob = torch.load(path, weights_only=False)
    return blob["X"], blob["Y"], blob["onsets"], blob["stim_ids"]


def _build_cache_save(path: str, X, Y, onsets, stim_ids) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save({"X": X, "Y": Y, "onsets": onsets, "stim_ids": stim_ids}, tmp)
    os.replace(tmp, path)


# --- neural-target region (ROI) selection ---------------------------------------
# Volumetric studies (NSD) load a fixed set of voxels (the nsdgeneral mask) as the
# target Y. A sub-ROI is a COLUMN-SUBSET of those already-loaded voxels -- not a
# re-read of the 65 GB timeseries: we map a small ROI label volume onto the SAME
# voxel order the loader used (``label_volume[nsdgeneral_mask]``), then keep only
# the requested labels. Voxels outside nsdgeneral are not available (clipped).
#
# Each name maps to (NSD ROI label file, the label values within it to keep). The menu
# is the famous specific functional regions + the early/ventral visual streams. The NSD
# reward covers the "ventral" stream (a fixed per-dataset region); the rest aid analysis.
_ROI_DEFS: dict[str, tuple[str, tuple[int, ...]]] = {
    "V1": ("prf-visualrois", (1, 2)),    # primary visual cortex (V1v + V1d)
    "V2": ("prf-visualrois", (3, 4)),    # secondary visual cortex (V2v + V2d)
    "V3": ("prf-visualrois", (5, 6)),    # third visual area (V3v + V3d)
    "hV4": ("prf-visualrois", (7,)),     # human V4
    "FFA": ("floc-faces", (2, 3)),       # fusiform face area (FFA-1 + FFA-2)
    "PPA": ("floc-places", (2,)),        # parahippocampal place area
    "EBA": ("floc-bodies", (1,)),        # extrastriate body area
    "VWFA": ("floc-words", (2, 3)),      # visual word form area (VWFA-1 + VWFA-2)
    "early": ("streams", (1,)),          # early visual stream
    "ventral": ("streams", (2, 3)),      # ventral visual stream (midventral + ventral)
}

_S3_NSD = "s3://natural-scenes-dataset"


def _nsd_subject(study) -> int:
    return min(type(study).SESSIONS_PER_SUBJECT)


def _ensure_roi_label_file(study, label_file: str):
    """Path to a small NSD ROI label volume, downloaded next to nsdgeneral if absent.

    The label .nii.gz lives in the study's func1pt8mm/roi dir (OUTSIDE the repo);
    it is tiny (~64 KB) and fetched once via the AWS CLI (anonymous).
    """
    subject = _nsd_subject(study)
    dst = study._roi_path(subject).parent / f"{label_file}.nii.gz"
    if not dst.exists():
        src = f"{_S3_NSD}/nsddata/ppdata/subj{subject:02}/func1pt8mm/roi/{label_file}.nii.gz"
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", src, str(dst)], check=True)
    return dst


def _roi_voxel_keep(study, roi: str):
    """(boolean keep-mask over the nsdgeneral voxels, n_nsdgeneral_voxels) for ``roi``.

    Returns ``(None, n_voxels)`` for the default full ``nsdgeneral`` region.
    """
    from neuralfetch.studies.allen2022massive import _get_roi_mask

    mask = _get_roi_mask(study._roi_path(_nsd_subject(study)))  # nsdgeneral 3-D bool
    n_voxels = int(mask.sum())
    if roi == "nsdgeneral":
        return None, n_voxels
    if roi not in _ROI_DEFS:
        raise ValueError(f"unknown roi {roi!r}; known: {['nsdgeneral', *_ROI_DEFS]}")
    import nibabel

    label_file, wanted = _ROI_DEFS[roi]
    labels = nibabel.load(_ensure_roi_label_file(study, label_file)).get_fdata()
    return np.isin(labels[mask], wanted), n_voxels


def _apply_roi(d: Dataset, study, Y):
    """Column-subset Y to the dataset's ``roi`` (no-op for the default nsdgeneral)."""
    if d.roi == "nsdgeneral":
        return Y
    if not hasattr(study, "_roi_path"):
        raise ValueError(f"roi={d.roi!r} requested but study {d.study!r} has no ROI masks")
    keep, n_voxels = _roi_voxel_keep(study, d.roi)
    if keep is None:
        return Y
    n_tr = Y.shape[1] // n_voxels  # voxels are flattened voxel-major: (voxel, tr)
    if Y.shape[1] != n_voxels * n_tr:
        raise RuntimeError(
            f"Y has {Y.shape[1]} columns, not a multiple of {n_voxels} nsdgeneral voxels"
        )
    col_keep = torch.as_tensor(np.repeat(keep, n_tr))
    return Y[:, col_keep]


# --- unique-stimulus collapse + canonical held-out split ------------------------
def _collapse_repeats(X, Y, onsets, stim_ids):
    """Average the neural target over the repeats of each unique stimulus.

    NSD shows each image ~3x; the stimulus X is identical across a repeat, so its
    group mean is unchanged, while averaging Y reduces trial noise. Returns one row
    per unique stimulus (X, Y, onsets, stim_ids), ordered by stimulus id.
    """
    uniq, first_idx, inv = np.unique(stim_ids, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    inv_t = torch.as_tensor(inv, dtype=torch.long)
    g = len(uniq)
    # Per-group sizes, computed ONCE; both averages divide by the clamped (>=1) count.
    counts_t = torch.bincount(inv_t, minlength=g).clamp(min=1)
    counts = counts_t.unsqueeze(1).float()
    counts_np = counts_t.numpy()

    # Y is averaged over repeats (this is what reduces trial noise).
    Y_g = torch.zeros(g, Y.shape[1], dtype=torch.float32).index_add_(0, inv_t, Y.float()) / counts
    # X is identical across a stimulus's repeats, so gather one row instead of
    # averaging the (wide) stimulus tensor.
    X_g = X[torch.as_tensor(first_idx, dtype=torch.long)]
    # Mean onset per group (used only by non-canonical splits).
    onsets_g = np.bincount(inv, weights=onsets, minlength=g) / counts_np
    return X_g, Y_g, onsets_g, uniq


def _image_id(filepath: str) -> int:
    """NSD 1-based image id from a stimulus path (``.../{id-1}.png``)."""
    return int(os.path.splitext(os.path.basename(filepath))[0]) + 1


def _canonical_test_ids(study) -> set[int]:
    """The NSD shared-1000 image ids (``sharedix``), via the study's data interface."""
    return {int(x) for x in np.asarray(study._get_test_image_ids()).reshape(-1).tolist()}
