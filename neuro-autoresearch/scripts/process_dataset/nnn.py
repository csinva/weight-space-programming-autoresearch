"""Process the raw NNN (triple-N macaque spiking) dataset into the in-repo data/nnn/ tree.

Reads the raw Processed_ses*.mat + exclude_area.xls + img_pool.mat from d.path and writes:
  data/nnn/stimuli/img_pool.npy   : the 1000 NSD images at NATIVE resolution (uint8 [1000,H,W,3])
  data/nnn/neural/all_units.pt    : ALL units UNFILTERED -- responses [1000, total_units] (NSD cols),
                                    per-unit reliability [total_units], per-unit stream area [total_units].

NNN_MIN_RELIABILITY and imsize stay LOAD-time knobs (no filtering / no resize here).

Run (full, ~seconds): python -m scripts.process_dataset.nnn
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import scipy.io as sio
import torch

from harness.loaders.shared import processed_dir
from harness.loaders.vision.nnn import _NNN_N_NSD_IMAGES, _nnn_area_table, _nnn_unit_area

# Raw source (the offline build reads here once; runtime reads only data/nnn/).
RAW = "/home/t-ymehta/Research/neuralset/data/nnn_processed"


def main():
    root = RAW
    out = processed_dir("nnn")
    os.makedirs(os.path.join(out, "stimuli"), exist_ok=True)
    os.makedirs(os.path.join(out, "neural"), exist_ok=True)

    # Stimuli: the 1000 NSD images at native resolution (force RGB).
    img_pool = sio.loadmat(os.path.join(root, "img_pool.mat"), simplify_cells=True)["img_pool"]
    imgs = []
    for im in img_pool[:_NNN_N_NSD_IMAGES]:
        a = np.asarray(im)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        imgs.append(a.astype("uint8"))
    imgs = np.stack(imgs)  # [1000, H, W, 3]
    np.save(os.path.join(out, "stimuli", "img_pool.npy"), imgs)

    # Neural: ALL units pooled across sessions, NO reliability filter.
    table = _nnn_area_table(root)
    cols, rel, areas = [], [], []
    for f in sorted(glob.glob(os.path.join(root, "Processed_ses*.mat"))):
        ses = int(re.search(r"ses(\d+)", os.path.basename(f)).group(1))
        m = sio.loadmat(f, simplify_cells=True)
        rb = np.asarray(m["response_best"], dtype="float32")[:, :_NNN_N_NSD_IMAGES]
        r = np.asarray(m["reliability_best"]).ravel()
        pos = np.asarray(m["pos"]).ravel()
        for i in range(rb.shape[0]):
            areas.append(_nnn_unit_area(table, ses, float(pos[i]))[1] or "?")
        cols.append(rb)
        rel.append(r)
    Y = np.concatenate(cols, axis=0).T            # [1000 images, total_units]
    rel = np.concatenate(rel).astype("float32")   # [total_units]
    areas = np.asarray(areas)                      # [total_units]
    torch.save({"Y": Y, "reliability": rel, "areas": areas},
               os.path.join(out, "neural", "all_units.pt"))
    print(f"wrote {out}: stimuli {imgs.shape} uint8; neural Y {Y.shape}, "
          f"{len(rel)} units ({(rel > 0.4).sum()} with rel>0.4)")


if __name__ == "__main__":
    main()
