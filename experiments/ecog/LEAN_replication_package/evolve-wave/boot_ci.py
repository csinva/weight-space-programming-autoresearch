"""Paired bootstrap over the 23 speech-rich test chunks: is the 0.0768 plateau
real, and are small inter-model differences distinguishable from noise?

For each model we fit the ridge on the (fixed) train set and compute the per-chunk
per-electrode correlation array on the 23 test chunks. Then we bootstrap-resample
WHOLE CHUNKS (the exchangeable unit) with replacement; for paired comparisons we
use the SAME resampled chunk indices for both models (removes shared between-chunk
variance). Weights are fixed (we are characterizing test-set sampling noise of the
fitted models, not refitting).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ECOG_RIDGE_GPU", "1")
os.environ.setdefault("ECOG_TEST_CHUNKS_FILE",
                      os.path.dirname(os.path.abspath(__file__)) + "/speech_rich_test_chunks.json")
os.environ.setdefault("ECOG_EXCLUDE_PREFIX", "450a36e4")

from src import data, encoding, features
from src.eval import _per_chunk_elec_corr

LIB = "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/wave-slurm-23628821/wave_encoders_lib"
MODELS = [
    ("A_best_0.0768", f"{LIB}/Gammatone-BEST-PB-EBAC.py"),
    ("B_relpitch_0.0763", f"{LIB}/Gammatone-BEST-rel-pitch.py"),
    ("C_earlier_0.0612", f"{LIB}/Gammatone-3win-6tau-AMfull-loud-centroid.py"),
]


def per_chunk_r(snap_path):
    spec = importlib.util.spec_from_file_location("m", snap_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(snap_path))
    spec.loader.exec_module(mod)
    emb = mod.build_embedder(device="cuda")
    train, test = data.get_chunk_names(500, 100)
    stim = data.load_stim(train + test)
    resp = data.load_responses(train + test)
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    wt, _, _ = encoding.bootstrap_ridge(np.vstack(ftr), np.vstack(rtr),
                                        np.vstack(fte), np.vstack(rte), np.logspace(1, 4, 12))
    mean, chunk_r = _per_chunk_elec_corr(fte, rte, wt)
    return mean, np.asarray(chunk_r)


def main():
    arrs = {}
    for lab, path in MODELS:
        print(f"fitting {lab} ...", flush=True)
        mean, cr = per_chunk_r(path)
        arrs[lab] = cr
        print(f"  {lab}: mean={mean:.4f}  n_chunks={len(cr)}", flush=True)
    np.savez(os.path.dirname(__file__) + "/boot_perchunk.npz", **arrs)

    labs = list(arrs.keys())
    n = len(arrs[labs[0]])
    B = 20000
    rng = np.random.RandomState(0)
    idx = rng.randint(0, n, size=(B, n))            # B x n resampled chunk indices
    boot = {lab: arrs[lab][idx].mean(1) for lab in labs}   # B bootstrap means per model

    print(f"\n=== single-model 95% CI (B={B}, resampling {n} chunks) ===")
    for lab in labs:
        m = arrs[lab].mean(); lo, hi = np.percentile(boot[lab], [2.5, 97.5])
        print(f"  {lab:18s} mean={m:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  half-width={(hi-lo)/2:.4f}")

    print("\n=== paired differences (same resampled chunks) ===")
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]
            d = boot[a] - boot[b]
            md = arrs[a].mean() - arrs[b].mean()
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_gt = float((d > 0).mean())     # fraction of resamples where a>b
            sig = "REAL (CI excludes 0)" if (lo > 0 or hi < 0) else "NOT distinguishable (CI spans 0)"
            print(f"  {a} - {b}: Δ={md:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  P(a>b)={p_gt:.3f}  -> {sig}")


if __name__ == "__main__":
    main()
