"""Per-electrode R for LEAN-v20 (current best) on the CANONICAL test set.
Whisper side is version-independent -> reuse r_wh from diag_perelec_canonical.npz.
Sanity check: grid mean should reproduce the canonical v20 grid score ~0.0905."""
import sys, os, gc
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from src import data, features, encoding
import lean_v20_enc as L

ALPHAS = np.logspace(1, 4, 12)


def per_elec_canonical(feat_list, resp_list, wt):
    acc = []
    for f, r in zip(feat_list, resp_list):
        pred = f @ wt
        a = r - r.mean(0); b = pred - pred.mean(0)
        denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        rr = np.where(denom == 0, 0.0, (a * b).sum(0) / np.where(denom == 0, 1.0, denom))
        acc.append(np.nan_to_num(rr))
    return np.mean(np.array(acc), axis=0)


def main():
    train, test = data.get_chunk_names(500, 100)
    print(f"train={len(train)} test={len(test)} (canonical fixed test set)", flush=True)
    stim = data.load_stim(train + test); resp = data.load_responses(train + test)
    emb = L.build_embedder(device="cuda", interact_set="v3e")  # canonical v20 config
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    del stim, resp; gc.collect()
    Xtr = np.vstack(ftr).astype(np.float32); Ytr = np.vstack(rtr).astype(np.float32)
    Xte = np.vstack(fte).astype(np.float32); Yte = np.vstack(rte).astype(np.float32)
    print(f"p={Xtr.shape[1]} nvox={Ytr.shape[1]} train_rows={Xtr.shape[0]}", flush=True)
    wt, _, _ = encoding.bootstrap_ridge_gpu(Xtr, Ytr, Xte, Yte, ALPHAS)
    r_lean = per_elec_canonical(fte, rte, wt)
    np.savez(os.path.join(os.path.dirname(__file__), "diag_perelec_v20.npz"), r_lean=r_lean)

    def g(sl): return f"{r_lean[sl].mean():.4f}"
    print("\n=== LEAN-v20 per-electrode canonical metric (canonical test set) ===")
    print(f"ALL-166 {g(slice(0,166))}  GRID {g(slice(0,125))}  STRIPS {g(slice(125,132))}  DEPTH {g(slice(132,166))}")
    print("(grid should be ~0.0905 to confirm correct v20 config)")


if __name__ == "__main__":
    main()
