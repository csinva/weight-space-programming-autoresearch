"""NSD volumetric ROI resolver: maps an ROI name to a per-voxel keep mask over the
nsdgeneral voxels. Used by harness/reward.py for the optional per-region test_corr
slice diagnostic (not by the loaders themselves). The autoresearch agent should NOT
edit this file.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np


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
    "ventral": ("streams", (2, 5)),      # ventral visual stream (midventral + ventral)
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
