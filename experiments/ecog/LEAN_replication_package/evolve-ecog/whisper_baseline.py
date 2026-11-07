"""Pretrained-encoder reference baseline: ridge on Whisper features -> ECoG.

Analogous to the fMRI project's GPT-2 XL reference row. Whisper-large encoder
features (`{chunk}_whisper.npy`, shape (T, 1280), already on the 30 Hz timeline)
are run through the SAME alignment + per-chunk z-scoring + per-chunk per-electrode
metric as the rest of the pipeline, then a linear ridge readout to the 166
electrodes is fit. This is a strong PRETRAINED ceiling to contextualize the
hand-written, no-training mel circuits the autoresearch loop produces.

The loop's bootstrap-SVD ridge is impractical at 1280 features, so this uses a
normal-equation ridge (XtX/XtY streamed per chunk on GPU) with the regularizer
chosen on a train-internal validation split (no test leakage). FIR delays are NOT
used here (ndelays=1): Whisper already carries temporal context, and 1280*delays
would blow up memory.

Usage (writes a `Whisper-reference` row into <run_dir>/results/overall_results.csv):
    python whisper_baseline.py <run_dir> --num-train 500 --num-test 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from os.path import join

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data, features
from src.eval import (make_result_row, upsert_overall_results,
                      plot_corr_over_iterations, _per_chunk_elec_corr)

BASELINE_NAME = "Whisper-reference"


def load_whisper_and_resp(chunks):
    """{c: (T,1280) f32 whisper}, {c: (T,166) f32 highgamma}, truncated to common T."""
    stim, resp = {}, {}
    for c in chunks:
        w = np.load(join(data.WAV_DIR, f"{c}_whisper.npy")).astype(np.float32)      # (T,1280)
        h = np.load(join(data.WAV_DIR, f"{c}_ecog_hasson_highgamma.npy")).T.astype(np.float32)  # (T,166)
        T = min(w.shape[0], h.shape[0])
        stim[c], resp[c] = w[:T], h[:T]
    return stim, resp


def feats_for(chunks, stim, resp, lag, trim):
    return features.get_features_per_chunk(
        stim, resp, chunks, lambda a: a, ndelays=1, lag=lag, trim=trim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--num-train", type=int, default=500)
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    t0 = time.time()

    lag, trim = features.LAG_FRAMES, features.TRIM_FRAMES
    train_chunks, test_chunks = data.get_chunk_names(args.num_train, args.num_test)
    # train-internal validation split for alpha selection (no test leakage)
    n_val = max(1, len(train_chunks) // 5)
    fit_chunks, val_chunks = train_chunks[n_val:], train_chunks[:n_val]
    print(f"fit {len(fit_chunks)}  val {len(val_chunks)}  test {len(test_chunks)} chunks")

    print("loading whisper + high-gamma...")
    all_chunks = train_chunks + test_chunks
    stim, resp = load_whisper_and_resp(all_chunks)

    print("building features (identity embedder, ndelays=1)...")
    ffit, rfit = feats_for(fit_chunks, stim, resp, lag, trim)
    fval, rval = feats_for(val_chunks, stim, resp, lag, trim)
    fte, rte = feats_for(test_chunks, stim, resp, lag, trim)
    p = ffit[0].shape[1]
    print(f"feature dim p={p}")

    # stream XtX / XtY over fit chunks (bounded memory)
    def accumulate(feat_list, resp_list):
        XtX = torch.zeros(p, p, dtype=torch.float64, device=device)
        XtY = torch.zeros(p, resp_list[0].shape[1], dtype=torch.float64, device=device)
        for f, r in zip(feat_list, resp_list):
            ft = torch.from_numpy(f).to(device, torch.float64)
            rt = torch.from_numpy(r).to(device, torch.float64)
            XtX += ft.T @ ft
            XtY += ft.T @ rt
        return XtX, XtY

    print("accumulating XtX/XtY on fit set...")
    XtX_fit, XtY_fit = accumulate(ffit, rfit)
    I = torch.eye(p, dtype=torch.float64, device=device)

    print("selecting alpha on val set...")
    best_alpha, best_val = None, -1e9
    for alpha in np.logspace(2, 7, 11):
        w = torch.linalg.solve(XtX_fit + alpha * I, XtY_fit)
        wt = w.cpu().numpy()
        val_corr, _ = _per_chunk_elec_corr(fval, rval, wt)
        print(f"  alpha={alpha:9.1f}  val_corr={val_corr:.4f}")
        if val_corr > best_val:
            best_val, best_alpha = val_corr, alpha
    print(f"best alpha={best_alpha:.1f} (val_corr={best_val:.4f})")

    print("refitting on full train (fit+val) with best alpha...")
    XtX_val, XtY_val = accumulate(fval, rval)
    XtX = XtX_fit + XtX_val
    XtY = XtY_fit + XtY_val
    wt = torch.linalg.solve(XtX + best_alpha * I, XtY).cpu().numpy()

    test_corr, _ = _per_chunk_elec_corr(fte, rte, wt)
    train_corr, _ = _per_chunk_elec_corr(ffit + fval, rfit + rval, wt)

    # per-electrode correlation over concatenated test (for frac>0.1)
    Xte = np.vstack(fte); Yte = np.vstack(rte)
    pred = Xte @ wt
    a = Yte - Yte.mean(0); b = pred - pred.mean(0)
    denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    elec_corr = np.where(denom == 0, 0.0, (a * b).sum(0) / np.where(denom == 0, 1.0, denom))
    frac = float(np.nanmean(elec_corr > 0.1))

    r = {
        "subject": "patient1", "test_corr": test_corr, "train_corr": train_corr,
        "frac_test_elec_above_0.1": frac, "encoding_seconds": time.time() - t0,
    }
    n_params = 635_000_000  # whisper-large encoder param count (pretrained reference)
    desc = (f"PRETRAINED REFERENCE: ridge on Whisper-large encoder features "
            f"(1280-dim, 30Hz aligned, ndelays=1), alpha={best_alpha:.0f}. Ceiling "
            f"for the hand-written no-training mel circuits.")
    results_dir = join(args.run_dir, "results")
    upsert_overall_results([make_result_row(r, BASELINE_NAME, n_params, desc)], results_dir)
    plot_corr_over_iterations(results_dir)

    print(f"\n--- {BASELINE_NAME}: test_corr={test_corr:.4f}  train_corr={train_corr:.4f}  "
          f"frac_elec>0.1={frac:.4f}  ({r['encoding_seconds']:.1f}s) ---")


if __name__ == "__main__":
    main()
