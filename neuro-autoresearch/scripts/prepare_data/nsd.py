"""Process NSD GLMdenoise betas into the in-repo, rsync-able ``data/nsd/`` tree.

Writes, for subjects 01/02/05/07 only:
  data/nsd/neural/subj{01,02,05,07}.pt  -- the per-subject blob (rois, nsd_ids, shared, ...),
                                            repacked from per_subject/subjNN.pkl, with
                                            ``blob['ncsnr'] = {roi: (n_vox,)}`` -- NSD's per-voxel
                                            noise-ceiling SNR, fetched + aligned (unless --no-ncsnr).
  data/nsd/stimuli/images.h5            -- NATIVE 425x425x3 uint8 NSD images, ONE dataset
                                            per nsd_id (key = str(nsd_id)), ONLY for the union
                                            of nsd_ids referenced by those 4 subjects (~37k of
                                            73k). NOT pre-resized -- imsize stays a load-time knob.

Once data/nsd/ exists, harness.loaders.vision.nsd prefers it automatically (no env flag).

Full run (all 4 subjects, ~20GB stimuli, slow -- streams the 39GB imgBrick):
    python -m scripts.prepare_data.nsd

Flags:
    --subjects 01 02 05 07   subset of subjects (default: all four)
    --ids-limit N            cap the stimuli union to the first N ids (smoke test)
    --out DIR                output root (default: processed_dir("nsd"))
    --neural-only            skip the (huge) stimuli HDF5
"""
from __future__ import annotations

import argparse
import os
import pickle

import h5py
import numpy as np
import torch

from harness.loaders.shared import processed_dir

RAW = "/home/t-ymehta/Research/neuralset/data/nsd_betas"
SUBJECTS = ["01", "02", "05", "07"]


def repack_neural(subjects, out):
    """Copy each per_subject pkl into data/nsd/neural/subjNN.pt (a faithful torch.save of the blob)."""
    nd = os.path.join(out, "neural")
    os.makedirs(nd, exist_ok=True)
    for s in subjects:
        dst = os.path.join(nd, f"subj{s}.pt")
        with open(os.path.join(RAW, "per_subject", f"subj{s}.pkl"), "rb") as f:
            blob = pickle.load(f)
        if os.path.exists(dst):                    # carry forward a prior ncsnr so a re-pack (esp.
            prev = torch.load(dst, weights_only=False)   # with --no-ncsnr) never silently strips it
            if "ncsnr" in prev:
                # CONTRACT: the carried-forward ncsnr is NOT re-validated here against the new betas.
                # A default run re-aligns it (build_ncsnr, count-asserted); but under --no-ncsnr the
                # stale ncsnr rides along untouched. Only --no-ncsnr re-pack when the betas voxel SET
                # is unchanged -- if the ROI/stream definition changed, drop ncsnr and rebuild it.
                blob["ncsnr"] = prev["ncsnr"]
        torch.save(blob, dst, pickle_protocol=4)
        print(f"  neural subj{s}: {len(np.asarray(blob['nsd_ids']))} images repacked")


def union_ids(subjects):
    """Sorted union of the nsd_ids seen by any of the given subjects (the stimuli to stage)."""
    ids = set()
    for s in subjects:
        with open(os.path.join(RAW, "per_subject", f"subj{s}.pkl"), "rb") as f:
            ids |= set(np.asarray(pickle.load(f)["nsd_ids"]).tolist())
    return np.array(sorted(ids), dtype=np.int64)


def store_stimuli(ids, out):
    """Write data/nsd/stimuli/images.h5: one (425,425,3) uint8 dataset per nsd_id (key=str(id)).
    nsd_ids are 1-based; imgBrick is 0-based -> row = nsd_id - 1."""
    sd = os.path.join(out, "stimuli")
    os.makedirs(sd, exist_ok=True)
    with h5py.File(os.path.join(RAW, "nsd_stimuli.hdf5"), "r") as src, \
         h5py.File(os.path.join(sd, "images.h5"), "w") as dst:
        brick = src["imgBrick"]                       # (73000, 425, 425, 3) uint8
        for i in range(0, len(ids), 500):
            chunk_ids = ids[i:i + 500]                # already sorted ascending -> rows ascending
            imgs = brick[chunk_ids - 1]              # (b, 425, 425, 3); h5 needs increasing rows
            for nid, img in zip(chunk_ids, imgs):
                dst.create_dataset(str(int(nid)), data=img)
            print(f"  stimuli: {min(i + 500, len(ids))}/{len(ids)} ids", end="\r")
    print(f"\n  stimuli: {len(ids)} native images stored")


def main():
    """Repack each subject's prepared betas into data/nsd/, align per-voxel ncsnr (unless --no-ncsnr), and store the union of stimuli images."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--ids-limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--neural-only", action="store_true")
    ap.add_argument("--no-ncsnr", action="store_true",
                    help="skip fetching/aligning NSD per-voxel ncsnr (the reliability signal)")
    a = ap.parse_args()
    out = a.out or processed_dir("nsd")
    print(f"NSD -> {out}  subjects={a.subjects}")
    repack_neural(a.subjects, out)
    if not a.no_ncsnr:
        from scripts.prepare_data.nsd_ncsnr import build as build_ncsnr
        cache = os.path.join(out, "ncsnr_src")
        os.makedirs(cache, exist_ok=True)
        for s in a.subjects:
            build_ncsnr(s, out, cache)
    if not a.neural_only:
        ids = union_ids(a.subjects)
        if a.ids_limit:
            ids = ids[:a.ids_limit]
        store_stimuli(ids, out)


if __name__ == "__main__":
    main()
