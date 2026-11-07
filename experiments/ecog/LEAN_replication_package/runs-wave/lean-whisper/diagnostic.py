"""Diagnose what Whisper captures that LEAN-v17 misses, split by electrode group.
The 166 model electrodes are ordered: GRID 0-124 (G2-G128), STRIPS 125-131,
DEPTH 132-165. The surface grid is what matters; report each group separately.
Memory-careful: free per-chunk lists, fit sub-models sequentially."""
import sys, os, gc
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from src import data, features, encoding
import wave_encoder as we  # combined LEAN-v17 + Whisper

LEAN_RAW, WH_RAW = 1963, 1280
TOTAL = LEAN_RAW + WH_RAW
ALPHAS = np.logspace(1, 4, 12)
NUM_TRAIN = 400
GROUPS = [("ALL 166", slice(0, 166)), ("GRID 0-124", slice(0, 125)),
          ("STRIPS 125-131", slice(125, 132)), ("DEPTH 132-165", slice(132, 166))]


def main():
    train, test = data.get_chunk_names(NUM_TRAIN, 100)
    stim = data.load_stim(train + test); resp = data.load_responses(train + test)
    emb = we.build_embedder(device="cuda")
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    Xtr = np.vstack(ftr).astype(np.float32); Ytr = np.vstack(rtr).astype(np.float32)
    del ftr, rtr; gc.collect()
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    Xte = np.vstack(fte).astype(np.float32); Yte = np.vstack(rte).astype(np.float32)
    del fte, rte, stim, resp; gc.collect()
    p = Xtr.shape[1]; ndelays = p // TOTAL
    print(f"features p={p} ndelays={ndelays} nvox={Ytr.shape[1]} Xtr={Xtr.nbytes/1e9:.0f}GB", flush=True)

    lean_cols, wh_cols = [], []
    for d in range(ndelays):
        lean_cols += list(range(d * TOTAL, d * TOTAL + LEAN_RAW))
        wh_cols += list(range(d * TOTAL + LEAN_RAW, (d + 1) * TOTAL))
    lean_cols = np.array(lean_cols); wh_cols = np.array(wh_cols)

    def fit(cols):
        Xa = np.ascontiguousarray(Xtr[:, cols]); Xb = np.ascontiguousarray(Xte[:, cols])
        _, corrs, _ = encoding.bootstrap_ridge_gpu(Xa, Ytr, Xb, Yte, ALPHAS)
        del Xa, Xb; gc.collect()
        return np.nan_to_num(corrs)

    print("fitting LEAN-only...", flush=True);    r_lean = fit(lean_cols)
    print("fitting Whisper-only...", flush=True); r_wh = fit(wh_cols)
    print("fitting combined...", flush=True);     r_comb = fit(np.arange(p))
    np.savez(os.path.join(os.path.dirname(__file__), "diag_perelec.npz"),
             r_lean=r_lean, r_wh=r_wh, r_comb=r_comb)

    print(f"\n{'group':16s} {'n':>4} {'LEAN':>8} {'Whisper':>8} {'combined':>9} {'W-L':>8} {'comb-W':>8}")
    for lbl, sl in GROUPS:
        l, w, c = r_lean[sl].mean(), r_wh[sl].mean(), r_comb[sl].mean()
        n = len(r_lean[sl])
        print(f"{lbl:16s} {n:>4} {l:>8.4f} {w:>8.4f} {c:>9.4f} {w-l:>+8.4f} {c-w:>+8.4f}")
    print("\n(reported test_corr metric = the ALL-166 row; grid is 0-124)")


if __name__ == "__main__":
    main()
