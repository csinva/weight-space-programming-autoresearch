"""Shared machinery for the data loaders.

Stimulus extractor (``RawPixels``), the on-disk build cache, neural-target ROI
selection, and the repeat-collapse + canonical-split helpers -- used by BOTH the
NeuralSet/Segmenter path (``neuralset``) and the pre-averaged table loaders
(``tvsd``/``nnn``). The autoresearch agent should NOT need to edit this file.
"""
from __future__ import annotations

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


def _cache_dir() -> str:
    return os.environ.get(
        "NEURO_SPLIT_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "neuro-autoresearch", "splits"),
    )


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
