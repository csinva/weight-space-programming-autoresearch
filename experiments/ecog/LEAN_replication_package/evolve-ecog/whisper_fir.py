"""Whisper-reference ceiling WITH FIR delays — does adding temporal delays to the
pretrained Whisper features lift the ceiling above the ndelays=1 value (0.0794)?

Same speech-rich split, same per-chunk per-electrode metric, same bootstrapped
ridge as the branches (GPU). Sweeps ndelays so the comparison to the hand-built
branches (which use ndelays=4) is apples-to-apples.
"""
from __future__ import annotations

import os
import sys
import time
from os.path import join

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ECOG_RIDGE_GPU", "1")
os.environ.setdefault("ECOG_TEST_CHUNKS_FILE",
                      os.path.dirname(os.path.abspath(__file__)) + "/speech_rich_test_chunks.json")
os.environ.setdefault("ECOG_EXCLUDE_PREFIX", "450a36e4")

from src import data, features, encoding
from src.eval import _per_chunk_elec_corr


def load(chunks):
    stim, resp = {}, {}
    for c in chunks:
        w = np.load(join(data.WAV_DIR, f"{c}_whisper.npy")).astype(np.float32)          # (T,1280)
        h = np.load(join(data.WAV_DIR, f"{c}_ecog_hasson_highgamma.npy")).T.astype(np.float32)
        T = min(len(w), len(h))
        stim[c], resp[c] = w[:T], h[:T]
    return stim, resp


def main():
    train, test = data.get_chunk_names(500, 100)
    print(f"train {len(train)}  test {len(test)} (speech-rich)")
    print("loading whisper + high-gamma...")
    stim, resp = load(train + test)
    identity = lambda a: a
    alphas = np.logspace(1, 4, 12)

    print(f"\n{'ndelays':>7} {'ridge_feats':>11} {'test_corr':>10} {'secs':>6}")
    for nd in (1, 2, 3, 4):
        t = time.time()
        ftr, rtr = features.get_features_per_chunk(stim, resp, train, identity, ndelays=nd)
        fte, rte = features.get_features_per_chunk(stim, resp, test, identity, ndelays=nd)
        Xtr, Ytr = np.vstack(ftr), np.vstack(rtr)
        Xte, Yte = np.vstack(fte), np.vstack(rte)
        wt, _, _ = encoding.bootstrap_ridge(Xtr, Ytr, Xte, Yte, alphas)
        tc, _ = _per_chunk_elec_corr(fte, rte, wt)
        print(f"{nd:>7} {Xtr.shape[1]:>11} {tc:>10.4f} {time.time()-t:>6.0f}", flush=True)


if __name__ == "__main__":
    main()
