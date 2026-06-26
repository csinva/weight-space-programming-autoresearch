"""Fetch NSD's per-voxel noise-ceiling SNR (ncsnr) and align it to our prepared betas.

NSD presents each image 3x; from those repeats Allen et al. (2021) derive a per-voxel
``ncsnr`` (= signal SD / noise SD). Our prep averaged the repeats away, so we never stored it.
This pulls the tiny ``ncsnr.nii.gz`` + the ROI label volumes from the public S3 bucket
(anonymous, ~a few hundred KB/subject) and writes ``blob['ncsnr'] = {roi: (n_vox,) float32}``,
aligned column-for-column to ``blob['rois']`` -- verified by a hard count assert per ROI.

    python -m scripts.process_dataset.nsd_ncsnr --subjects 01 02 05 07

Requires the AWS CLI (anonymous --no-sign-request) + nibabel. One-time; re-run is a no-op
(files cached under data/nsd/ncsnr_src/).
"""
from __future__ import annotations

import argparse
import os
import subprocess

import nibabel as nib
import numpy as np
import torch

from harness.loaders.shared import processed_dir

# NSD public bucket. roi labels live under nsddata; ncsnr under nsddata_betas.
_ROI_BASE = "s3://natural-scenes-dataset/nsddata/ppdata/subj{s}/func1pt8mm/roi/{f}.nii.gz"
_NCSNR = ("s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{s}/func1pt8mm/"
          "betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz")

# Atlas + label selection MATCHING THE PREPARED BLOB (verified by exact voxel-count match per ROI).
# NSD streams.mgz.ctab labels: 1 early · 2 midventral · 3 midlateral · 4 midparietal · 5 ventral ·
# 6 lateral · 7 parietal. So the reward region is early (1) + ventral stream (midventral+ventral =
# {2,5}) -- the NSD-defined early and ventral visual streams. (The harness neuralset advisory
# resolver was likewise corrected to ventral={2,5}; both codepaths now agree.)
ROI_DEFS = {
    "V1": ("prf-visualrois", (1, 2)), "V2": ("prf-visualrois", (3, 4)),
    "V3": ("prf-visualrois", (5, 6)), "hV4": ("prf-visualrois", (7,)),
    "early": ("streams", (1,)), "ventral": ("streams", (2, 5)),
    "FFA": ("floc-faces", (2, 3)), "PPA": ("floc-places", (2,)), "VWFA": ("floc-words", (2, 3)),
}


def _fetch(src, dst):
    if not os.path.exists(dst):
        tmp = dst + ".tmp"                          # atomic: a broken download must not poison cache
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", src, tmp], check=True)
        os.replace(tmp, dst)
    return nib.load(dst).get_fdata()


def build(subj, out, cache):
    pt = os.path.join(out, "neural", f"subj{subj}.pt")
    blob = torch.load(pt, weights_only=False)
    ncsnr_vol = _fetch(_NCSNR.format(s=subj), os.path.join(cache, f"subj{subj}_ncsnr.nii.gz"))
    labels, ncsnr = {}, {}
    for roi, (lab_file, labs) in ROI_DEFS.items():
        if lab_file not in labels:
            labels[lab_file] = _fetch(_ROI_BASE.format(s=subj, f=lab_file),
                                      os.path.join(cache, f"subj{subj}_{lab_file}.nii.gz"))
        keep = np.isin(labels[lab_file], labs)               # C-order over the masked voxels
        vals = ncsnr_vol[keep].astype("float32")
        # CONTRACT (count-checked only, not order-checked): the prepared blob's per-ROI betas MUST
        # have been extracted in this same np.isin C-order over the label volume. The assert below
        # catches a count mismatch but CANNOT detect a same-count voxel REORDER -- if the betas prep
        # ever changes its voxel ordering, this silently maps each ncsnr to the wrong voxel.
        n_have = np.asarray(blob["rois"][roi]).shape[1]
        assert vals.size == n_have, (
            f"subj{subj} {roi}: ncsnr has {vals.size} voxels but betas has {n_have} "
            f"-- atlas selection/order differs from the prepared blob; not safe to align.")
        ncsnr[roi] = vals
    blob["ncsnr"] = ncsnr
    torch.save(blob, pt, pickle_protocol=4)
    print(f"  subj{subj}: ncsnr aligned for {len(ncsnr)} ROIs "
          f"(early {ncsnr['early'].size} / ventral {ncsnr['ventral'].size} vox)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["01", "02", "05", "07"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or processed_dir("nsd")
    cache = os.path.join(out, "ncsnr_src")
    os.makedirs(cache, exist_ok=True)
    for s in a.subjects:
        build(s, out, cache)


if __name__ == "__main__":
    main()
