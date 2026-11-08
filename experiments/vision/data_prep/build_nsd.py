#!/usr/bin/env python3
"""Rebuild nsd_data.pkl (the format neuro_eval/readers.load_all_nsd_data expects) from the public NSD bucket.

The original builder lived in the private looped-networks repo; this reconstruction follows its readers:
  - betas: nsddata_betas/ppdata/subjNN/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_sessionXX.nii.gz (int16, /300 = % signal change),
    streamed one session at a time and deleted after use
  - each voxel z-scored within each session, then the repeats of each image averaged
  - stimulus coord = 0-based nsdId (row of imgBrick in nsd_stimuli.hdf5); neuroid coords x, y, z (0-based func1pt8mm indices)
  - voxels stored unfiltered; the reader applies ncsnr > 0.2 from the official ncsnr.nii.gz at load time
  - regions: prf-visualrois V1 {1,2}, V2 {3,4}, V3 {5,6}, hV4 {7}; streams early {1}, ventral {5}; floc-faces FFA {2,3}; floc-places PPA {2}
  - shared_ids = the 1000 shared images (sharedix - 1)
Usage (from experiments/vision):
  python data_prep/build_nsd.py --root $NSD_ROOT --subject 1        # one subject -> <root>/prepared/subj01.pkl
  python data_prep/build_nsd.py --root $NSD_ROOT --assemble --out $NSD_DATA_DIR/nsd_data.pkl
"""
import argparse
import os
import pickle
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import scipy.io
import xarray as xr

SESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
REGIONS = {
    "V1": ("prf-visualrois", (1, 2)), "V2": ("prf-visualrois", (3, 4)), "V3": ("prf-visualrois", (5, 6)),
    "hV4": ("prf-visualrois", (7,)), "early": ("streams", (1,)), "ventral": ("streams", (5,)),
    "FFA": ("floc-faces", (2, 3)), "PPA": ("floc-places", (2,)),
}
BUCKET = "s3://natural-scenes-dataset"
TRIALS = 750


def masks(root, subj):
    out = {}
    for name, (atlas, labels) in REGIONS.items():
        vol = np.asanyarray(nib.load(f"{root}/nsddata/ppdata/subj{subj:02d}/func1pt8mm/roi/{atlas}.nii.gz").dataobj)
        out[name] = np.isin(vol, labels)
    return out


def build_subject(root, subj):
    design = scipy.io.loadmat(f"{root}/nsddata/experiments/nsd/nsd_expdesign.mat")
    order = design["masterordering"].ravel() - 1                       # trial -> column of subjectim (0-based)
    nsd_ids = design["subjectim"][subj - 1, order] - 1                  # trial -> 0-based nsdId
    m = masks(root, subj)
    union = np.zeros_like(next(iter(m.values())))
    for v in m.values():
        union |= v
    coords = np.argwhere(union)                                         # (V, 3) x, y, z
    sums, counts = {}, {}
    tmp = tempfile.mkdtemp(dir=os.environ.get("NSD_TMP", "/data19/chandan/wsp_vision/tmp"))
    for ses in range(1, SESSIONS[subj] + 1):
        key = f"nsddata_betas/ppdata/subj{subj:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{ses:02d}.nii.gz"
        local = os.path.join(tmp, os.path.basename(key))
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", f"{BUCKET}/{key}", local], check=True)
        data = np.asanyarray(nib.load(local).dataobj)                    # (X, Y, Z, 750) int16
        b = data[union].T.astype(np.float32) / 300.0                     # (750, V)
        os.remove(local)
        b = (b - b.mean(0)) / np.maximum(b.std(0), 1e-6)
        ids = nsd_ids[(ses - 1) * TRIALS: ses * TRIALS]
        for i, sid in enumerate(ids):
            sid = int(sid)
            if sid in sums:
                sums[sid] += b[i]
                counts[sid] += 1
            else:
                sums[sid] = b[i].copy()
                counts[sid] = 1
        print(f"subj{subj:02d} session {ses}/{SESSIONS[subj]}: {len(sums)} images", flush=True)
    os.rmdir(tmp)
    sids = np.array(sorted(sums))
    resp = np.stack([sums[s] / counts[s] for s in sids]).astype(np.float32)
    region_idx = {name: np.flatnonzero(mask[union]) for name, mask in m.items()}
    out = os.path.join(root, "prepared", f"subj{subj:02d}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"stimulus": sids, "resp": resp, "coords": coords, "region_idx": region_idx,
                     "n_repeats": np.array([counts[s] for s in sids])}, f, protocol=4)
    print(f"wrote {out}: {resp.shape}", flush=True)


def assemble(root, out):
    design = scipy.io.loadmat(f"{root}/nsddata/experiments/nsd/nsd_expdesign.mat")
    data = {name: {} for name in REGIONS}
    for subj in range(1, 9):
        with open(os.path.join(root, "prepared", f"subj{subj:02d}.pkl"), "rb") as f:
            p = pickle.load(f)
        for name, idx in p["region_idx"].items():
            c = p["coords"][idx]
            data[name][subj - 1] = xr.DataArray(
                p["resp"][:, idx], dims=("stimulus", "neuroid"),
                coords={"stimulus": p["stimulus"].astype(int), "x": ("neuroid", c[:, 0]), "y": ("neuroid", c[:, 1]),
                        "z": ("neuroid", c[:, 2])},
                attrs={"resolution": "1pt8mm", "preprocessing": "fithrf_GLMdenoise_RR"})
    nsd = {"data": data, "shared_ids": {int(i) - 1 for i in design["sharedix"].ravel()}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(nsd, f, protocol=4)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subject", type=int)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.assemble:
        assemble(a.root, a.out)
    else:
        build_subject(a.root, a.subject)


if __name__ == "__main__":
    main()
