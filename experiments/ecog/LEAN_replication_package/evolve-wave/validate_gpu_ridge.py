"""Validate the GPU ridge against the CPU SVD ridge on a real model+split.

Builds features with the default gammatone front-end on the speech-rich split,
then runs encoding.bootstrap_ridge on CPU and bootstrap_ridge_gpu, and compares
the fitted weights, chosen alphas, and the per-chunk per-electrode test_corr.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data, encoding, features
from src.eval import _per_chunk_elec_corr
from wave_encoder import build_embedder

NUM_TRAIN = 150


def main():
    emb = build_embedder(device="cuda")
    train, test = data.get_chunk_names(NUM_TRAIN, 100)
    print(f"train {len(train)} test {len(test)}")
    stim = data.load_stim(train + test)
    resp = data.load_responses(train + test)
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    Xtr, Ytr = np.vstack(ftr), np.vstack(rtr)
    Xte, Yte = np.vstack(fte), np.vstack(rte)
    alphas = np.logspace(1, 4, 12)
    print(f"Xtr {Xtr.shape}  Xte {Xte.shape}")

    os.environ["ECOG_RIDGE_GPU"] = "0"
    t = time.time()
    wt_c, corr_c, ab_c = encoding.bootstrap_ridge(Xtr, Ytr, Xte, Yte, alphas)
    t_cpu = time.time() - t
    tc_c, _ = _per_chunk_elec_corr(fte, rte, wt_c)

    t = time.time()
    wt_g, corr_g, ab_g = encoding.bootstrap_ridge_gpu(Xtr, Ytr, Xte, Yte, alphas)
    t_gpu = time.time() - t
    tc_g, _ = _per_chunk_elec_corr(fte, rte, wt_g)

    print("\n=== CPU vs GPU ridge ===")
    print(f"  time:            CPU {t_cpu:.1f}s   GPU {t_gpu:.1f}s   speedup {t_cpu/max(t_gpu,1e-9):.1f}x")
    print(f"  test_corr:       CPU {tc_c:.6f}   GPU {tc_g:.6f}   |Δ| {abs(tc_c-tc_g):.2e}")
    print(f"  weights max|Δ|:  {np.abs(wt_c-wt_g).max():.3e}   (||wt||~{np.abs(wt_c).mean():.3e})")
    print(f"  alphas agree:    {int((ab_c==ab_g).sum())}/{len(ab_c)} electrodes")
    print(f"  corrs(frac) max|Δ|: {np.abs(corr_c-corr_g).max():.2e}")
    ok = abs(tc_c - tc_g) < 1e-3
    print(f"\n  VERDICT: {'PASS — GPU matches CPU' if ok else 'FAIL — investigate'}")


if __name__ == "__main__":
    main()
