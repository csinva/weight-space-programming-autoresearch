"""Process NSD GLMdenoise betas into the in-repo, rsync-able ``data/nsd/`` tree.

Writes, for subjects 01/02/05/07 only:
  data/nsd/neural/subj{01,02,05,07}.pt  -- the per-subject blob (rois, nsd_ids, shared, ...),
                                            repacked from per_subject/subjNN.pkl.
  data/nsd/stimuli/images.h5            -- NATIVE 425x425x3 uint8 NSD images, ONE dataset
                                            per nsd_id (key = str(nsd_id)), ONLY for the union
                                            of nsd_ids referenced by those 4 subjects (~37k of
                                            73k). NOT pre-resized -- imsize stays a load-time knob.

Once data/nsd/ exists, harness.loaders.standalone.nsd prefers it automatically (no env flag).

Full run (all 4 subjects, ~20GB stimuli, slow -- streams the 39GB imgBrick):
    python -m scripts.process_dataset.nsd

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
        with open(os.path.join(RAW, "per_subject", f"subj{s}.pkl"), "rb") as f:
            blob = pickle.load(f)
        torch.save(blob, os.path.join(nd, f"subj{s}.pt"), pickle_protocol=4)
        print(f"  neural subj{s}: {len(np.asarray(blob['nsd_ids']))} images repacked")


def union_ids(subjects):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--ids-limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--neural-only", action="store_true")
    a = ap.parse_args()
    out = a.out or processed_dir("nsd")
    print(f"NSD -> {out}  subjects={a.subjects}")
    repack_neural(a.subjects, out)
    if not a.neural_only:
        ids = union_ids(a.subjects)
        if a.ids_limit:
            ids = ids[:a.ids_limit]
        store_stimuli(ids, out)


if __name__ == "__main__":
    main()
