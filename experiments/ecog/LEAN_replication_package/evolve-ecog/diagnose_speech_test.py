"""Diagnostic: how do the encoders score on a SPEECH-RICH held-out test set?

Evaluates three encoders on the monkey-selfie-copyright podcast block
(450a36e4_0190..0212, all speech-rich) and compares to their random-split
numbers. The whole 450a36e4 recording is excluded from training to avoid
adjacency leakage. Ridge readouts are refit per encoder; metric is the project
per-chunk per-electrode correlation. No changes to the running loop.

Usage:
    python diagnose_speech_test.py --run-dir ../runs-ecog/ecog-slurm-23542093
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from os.path import join

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data, features, encoding
from src.eval import _per_chunk_elec_corr
from src.baseline import MelspecIdentityEmbedder

TEST_RECORDING = "450a36e4"
TEST_CHUNKS = [f"{TEST_RECORDING}_{i:04d}" for i in range(190, 213)]  # extended block


def valid(c):
    return (os.path.exists(join(data.WAV_DIR, f"{c}_melspec.npy"))
            and os.path.exists(join(data.WAV_DIR, f"{c}_ecog_hasson_highgamma.npy")))


def melspec_pipeline_corr(embedder, train, test, ndelays):
    stim = data.load_stim(train + test)
    resp = data.load_responses(train + test)
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, embedder, ndelays=ndelays)
    fte, rte = features.get_features_per_chunk(stim, resp, test, embedder, ndelays=ndelays)
    Xtr, Ytr = np.vstack(ftr), np.vstack(rtr)
    Xte, Yte = np.vstack(fte), np.vstack(rte)
    wt, _, _ = encoding.bootstrap_ridge(
        Xtr, Ytr, Xte, Yte, np.logspace(1, 4, 12), nboots=5, chunklen=40, nchunks=20)
    tc, _ = _per_chunk_elec_corr(fte, rte, wt)
    return tc


def whisper_corr(train, test, device):
    lag, trim = features.LAG_FRAMES, features.TRIM_FRAMES
    n_val = max(1, len(train) // 5)
    fit, val = train[n_val:], train[:n_val]

    def load(chunks):
        stim, resp = {}, {}
        for c in chunks:
            w = np.load(join(data.WAV_DIR, f"{c}_whisper.npy")).astype(np.float32)
            h = np.load(join(data.WAV_DIR, f"{c}_ecog_hasson_highgamma.npy")).T.astype(np.float32)
            T = min(w.shape[0], h.shape[0]); stim[c], resp[c] = w[:T], h[:T]
        return stim, resp

    def feats(chunks, stim, resp):
        return features.get_features_per_chunk(stim, resp, chunks, lambda a: a, ndelays=1, lag=lag, trim=trim)

    stim, resp = load(train + test)
    ffit, rfit = feats(fit, stim, resp)
    fval, rval = feats(val, stim, resp)
    fte, rte = feats(test, stim, resp)
    p = ffit[0].shape[1]

    def accum(fl, rl):
        XtX = torch.zeros(p, p, dtype=torch.float64, device=device)
        XtY = torch.zeros(p, rl[0].shape[1], dtype=torch.float64, device=device)
        for f, r in zip(fl, rl):
            ft = torch.from_numpy(f).to(device, torch.float64)
            rt = torch.from_numpy(r).to(device, torch.float64)
            XtX += ft.T @ ft; XtY += ft.T @ rt
        return XtX, XtY

    XtX_f, XtY_f = accum(ffit, rfit)
    I = torch.eye(p, dtype=torch.float64, device=device)
    best_a, best_v = None, -1e9
    for a in np.logspace(2, 7, 11):
        wt = torch.linalg.solve(XtX_f + a * I, XtY_f).cpu().numpy()
        vc, _ = _per_chunk_elec_corr(fval, rval, wt)
        if vc > best_v: best_v, best_a = vc, a
    XtX_v, XtY_v = accum(fval, rval)
    wt = torch.linalg.solve(XtX_f + XtX_v + best_a * I, XtY_f + XtY_v).cpu().numpy()
    tc, _ = _per_chunk_elec_corr(fte, rte, wt)
    return tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="../runs-ecog/ecog-slurm-23542093")
    ap.add_argument("--num-train", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    test = [c for c in TEST_CHUNKS if valid(c)]
    pool = [c for c in data._valid_chunks(seed=42) if not c.startswith(TEST_RECORDING)]
    train = pool[:args.num_train]
    print(f"train {len(train)} chunks (450a36e4 excluded)  |  speech-rich test {len(test)} chunks")

    random_split = {"Ridge-melspec-baseline": 0.0422, "MelBank-v8-onset-rect": 0.0477,
                    "Whisper-reference": 0.0651}
    out = {}

    t = time.time()
    print("\n[1/3] melspec ridge baseline...")
    out["Ridge-melspec-baseline"] = melspec_pipeline_corr(MelspecIdentityEmbedder(), train, test, ndelays=4)
    print(f"  speech-rich test_corr = {out['Ridge-melspec-baseline']:.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    print("\n[2/3] MelBank-v8-onset-rect (best hand-written)...")
    snap = join(args.run_dir, "interpretable_transformers_lib", "MelBank-v8-onset-rect.py")
    sys.path.insert(0, os.path.abspath(args.run_dir))  # so snapshot's `from src...` resolves
    spec = importlib.util.spec_from_file_location("melbank_v8", snap)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    emb = mod.build_embedder(device=device)
    out["MelBank-v8-onset-rect"] = melspec_pipeline_corr(emb, train, test, ndelays=4)
    print(f"  speech-rich test_corr = {out['MelBank-v8-onset-rect']:.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    print("\n[3/3] Whisper-reference...")
    out["Whisper-reference"] = whisper_corr(train, test, device)
    print(f"  speech-rich test_corr = {out['Whisper-reference']:.4f}  ({time.time()-t:.0f}s)")

    print("\n=== speech-rich (monkey/copyright 0190-0212) vs random-100 split ===")
    print(f"{'model':28s} {'random-split':>12s} {'speech-rich':>12s} {'delta':>8s}")
    for k in ["Ridge-melspec-baseline", "MelBank-v8-onset-rect", "Whisper-reference"]:
        rs, sr = random_split[k], out[k]
        print(f"{k:28s} {rs:>12.4f} {sr:>12.4f} {sr-rs:>+8.4f}")


if __name__ == "__main__":
    main()
