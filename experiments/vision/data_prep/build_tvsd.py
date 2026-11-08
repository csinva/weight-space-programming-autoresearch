#!/usr/bin/env python3
"""Rebuild tvsd_data.pkl and tvsd_stimuli.h5 (formats of neuro_eval/readers.load_all_tvsd_data) from the public TVSD release
(Papale et al. 2025, https://gin.g-node.org/paolo_papale/TVSD, CC BY 4.0) and the THINGS images (osf.io/jum2f, separate terms).

Reconstruction of the looped-networks builder, following the repository's old harness loader (git history ff7fff30^):
  - monkey{F,N}/THINGS_normMUA.mat: train_MUA (22,248 x 1024) and test_MUA (100 x 1024, repeats averaged), oracle (1024)
  - electrode -> area: F: V1 1-512, IT 513-832, V4 833-1024; N: V1 1-512, V4 513-768, IT 769-1024 (from _code/norm_MUA.m)
  - regions: V1, V4, IT, early = V1 + V4, ventral = IT; electrodes kept when oracle > 0.4
  - stimulus ids: 0..22247 = training images in things_imgs.mat order, 22248..22347 = the 100 test images (same for both monkeys);
    a 'split' coord on stimulus marks train / test
Usage (from experiments/vision):
  python data_prep/build_tvsd.py --raw <dir with monkeyF/, monkeyN/> --out <TVSD_DATA_DIR>                 # neural pickle
  python data_prep/build_tvsd.py --raw <dir> --out <TVSD_DATA_DIR> --things <THINGS object_images dir>    # + tvsd_stimuli.h5
"""
import argparse
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import xarray as xr

AREAS = {"F": {"V1": (1, 512), "IT": (513, 832), "V4": (833, 1024)},
         "N": {"V1": (1, 512), "V4": (513, 768), "IT": (769, 1024)}}
GROUPS = {"V1": ["V1"], "V4": ["V4"], "IT": ["IT"], "early": ["V1", "V4"], "ventral": ["IT"]}
THRESH = 0.4


def read_mat(path):
    with h5py.File(path) as f:
        train = np.asarray(f["train_MUA"], dtype=np.float32)
        test = np.asarray(f["test_MUA"], dtype=np.float32)
        oracle = np.asarray(f["oracle"], dtype=np.float32).ravel()
    if train.shape[0] != 22248:      # h5py may return the transposed MATLAB layout
        train = train.T
    if test.shape[0] != 100:
        test = test.T
    return train, test, oracle


def image_paths(raw):
    with h5py.File(os.path.join(raw, "monkeyF", "_logs", "things_imgs.mat")) as f:
        out = {}
        for split in ("train_imgs", "test_imgs"):
            g = f[split]
            ref = g["things_path"] if "things_path" in g else g
            paths = []
            for r in np.asarray(ref).ravel():
                obj = f[r]
                paths.append("".join(chr(int(c)) for c in np.asarray(obj).ravel()))
            out[split] = paths
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--things", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_train, n_test = 22248, 100
    ids = np.arange(n_train + n_test)
    split = np.array(["train"] * n_train + ["test"] * n_test)
    data = {k: {} for k in GROUPS}
    counts = {k: {} for k in GROUPS}
    for subj, monkey in enumerate(("F", "N")):
        train, test, oracle = read_mat(os.path.join(a.raw, f"monkey{monkey}", "THINGS_normMUA.mat"))
        resp = np.concatenate([train, test])
        for key, areas in GROUPS.items():
            elec = np.concatenate([np.arange(AREAS[monkey][ar][0] - 1, AREAS[monkey][ar][1]) for ar in areas])
            keep = elec[oracle[elec] > THRESH]
            counts[key][subj] = (int(len(keep)), int(len(elec)))
            if len(keep) == 0:
                continue
            data[key][subj] = xr.DataArray(resp[:, keep], dims=("stimulus", "neuroid"),
                                           coords={"stimulus": ids, "split": ("stimulus", split),
                                                   "oracle": ("neuroid", oracle[keep]), "electrode": ("neuroid", keep)})
        print(monkey, {k: counts[k][subj] for k in GROUPS}, flush=True)
    paths = image_paths(a.raw)
    stim = pd.DataFrame({"stimulus_id": ids, "things_path": paths["train_imgs"] + paths["test_imgs"], "split": split})
    with open(os.path.join(a.out, "tvsd_data.pkl"), "wb") as f:
        pickle.dump({"data": data, "unit_counts": counts, "oracle_threshold": THRESH, "stimuli": stim}, f, protocol=4)
    print("wrote", os.path.join(a.out, "tvsd_data.pkl"))
    if a.things:
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        with h5py.File(os.path.join(a.out, "tvsd_stimuli.h5"), "w") as h:
            ds = h.create_dataset("images", (len(ids),), dtype=dt)
            for i, p in enumerate(stim["things_path"]):
                rel = p.replace("\\", "/").split("object_images/")[-1]
                ds[i] = np.frombuffer(open(os.path.join(a.things, rel), "rb").read(), dtype=np.uint8)
        print("wrote", os.path.join(a.out, "tvsd_stimuli.h5"))


if __name__ == "__main__":
    main()
