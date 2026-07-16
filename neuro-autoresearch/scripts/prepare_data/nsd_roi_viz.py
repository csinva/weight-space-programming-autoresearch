"""Visualize the NSD reward regions (early + ventral visual streams) with nilearn, to CONFIRM
the voxel selection on the brain rather than trusting label numbers.

Uses the streams atlas + T1 (func1pt8mm) fetched under data/nsd/ncsnr_src/ by nsd_ncsnr.py.
NSD streams.mgz.ctab: 1 early · 2 midventral · 3 midlateral · 4 midparietal · 5 ventral ·
6 lateral · 7 parietal. The reward region = early (1) + ventral stream (midventral+ventral = {2,5}).

    python -m scripts.prepare_data.nsd_roi_viz --subject 01
Writes PNGs to scripts/prepare_data/results/.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting

SRC = "data/nsd/ncsnr_src"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EARLY, VENTRAL = (1,), (2, 5)   # streams labels: early; midventral+ventral


def _img(data, ref):
    """Wrap a data array in a Nifti1Image borrowing ref's affine/header."""
    return nib.Nifti1Image(data.astype("float32"), ref.affine, ref.header)


def main():
    """Render one subject's streams atlas as PNGs: early-vs-ventral reward region, the full 7-label atlas, and (if provisioned) the ncsnr reliability map."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="01")
    a = ap.parse_args()
    s = a.subject
    streams = nib.load(f"{SRC}/subj{s}_streams.nii.gz")
    sd = streams.get_fdata()
    bg = f"{SRC}/subj{s}_T1_to_func1pt8mm.nii.gz"
    os.makedirs(OUT, exist_ok=True)

    # 1) Early vs ventral, two colours -- the confirmation figure.
    twoc = np.zeros_like(sd)
    twoc[np.isin(sd, EARLY)] = 1
    twoc[np.isin(sd, VENTRAL)] = 2
    disp = plotting.plot_roi(
        _img(twoc, streams), bg_img=bg, display_mode="z", cut_coords=7,
        cmap=ListedColormap(["#2e75b6", "#c0392b"]), alpha=0.75, black_bg=False,
        title=f"NSD subj{s}: early (blue, {int((twoc==1).sum())}) + "
              f"ventral (red, {int((twoc==2).sum())}) visual streams")
    p1 = f"{OUT}/nsd_streams_early_ventral.png"
    disp.savefig(p1, dpi=130); disp.close()

    # 2) Full streams atlas, all 7 labels.
    names = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    cols = ["#2e75b6", "#c0392b", "#9b59b6", "#7f8c8d", "#e67e22", "#16a085", "#f1c40f"]
    disp = plotting.plot_roi(
        streams, bg_img=bg, display_mode="z", cut_coords=7, cmap=ListedColormap(cols),
        alpha=0.8, black_bg=False, title=f"NSD subj{s}: full streams atlas (1-7)")
    p2 = f"{OUT}/nsd_streams_all.png"
    disp.savefig(p2, dpi=130); disp.close()

    # 3) ncsnr (reliability) map over the reward region, if provisioned in the blob.
    p3 = None
    region = np.isin(sd, EARLY + VENTRAL)
    ncsnr_vol = f"{SRC}/subj{s}_ncsnr.nii.gz"
    if os.path.exists(ncsnr_vol):
        nc = nib.load(ncsnr_vol).get_fdata()
        ncr = np.sqrt(nc**2 / (nc**2 + 1/3))      # NC_r at n=3
        ncr[~region] = 0
        disp = plotting.plot_stat_map(
            _img(ncr, streams), bg_img=bg, display_mode="z", cut_coords=7, cmap="inferno",
            vmax=1.0, black_bg=False, colorbar=True,
            title=f"NSD subj{s}: per-voxel noise ceiling NC_r (early+ventral)")
        p3 = f"{OUT}/nsd_ncsnr.png"
        disp.savefig(p3, dpi=130); disp.close()

    print("saved:", p1, p2, p3)


if __name__ == "__main__":
    main()
