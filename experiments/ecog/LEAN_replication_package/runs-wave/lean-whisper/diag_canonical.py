"""Per-electrode LEAN vs Whisper R on the CANONICAL harness test set.
Matches the exceed harness exactly: get_chunk_names(500,100) with
ECOG_TEST_CHUNKS_FILE=speech_rich_test_chunks.json, EXCLUDE 450a36e4,
GPU ridge, alphas logspace(1,4,12). Uses the canonical per-chunk metric
(_per_chunk_elec_corr) decomposed PER ELECTRODE: for each electrode, mean
over test chunks of its per-chunk Pearson r. All 166 electrodes (not grid-sliced)."""
import sys, os, gc
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from src import data, features, encoding
import wave_encoder as we

LEAN_RAW, WH_RAW = 1963, 1280
TOTAL = LEAN_RAW + WH_RAW
ALPHAS = np.logspace(1, 4, 12)


def per_elec_canonical(feat_list, resp_list, cols, wt):
    """mean over chunks of per-chunk per-electrode Pearson r -> (n_elec,)."""
    acc = []
    for f, r in zip(feat_list, resp_list):
        pred = f[:, cols] @ wt                      # (T, n_elec)
        a = r - r.mean(0); b = pred - pred.mean(0)
        denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        rr = np.where(denom == 0, 0.0, (a * b).sum(0) / np.where(denom == 0, 1.0, denom))
        acc.append(np.nan_to_num(rr))
    return np.mean(np.array(acc), axis=0)           # (n_elec,)


def main():
    train, test = data.get_chunk_names(500, 100)
    print(f"train={len(train)} test={len(test)} (canonical fixed test set)", flush=True)
    stim = data.load_stim(train + test); resp = data.load_responses(train + test)
    emb = we.build_embedder(device="cuda")
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    del stim, resp; gc.collect()
    Xtr = np.vstack(ftr).astype(np.float32); Ytr = np.vstack(rtr).astype(np.float32)
    Xte = np.vstack(fte).astype(np.float32); Yte = np.vstack(rte).astype(np.float32)
    p = Xtr.shape[1]; ndelays = p // TOTAL
    print(f"p={p} ndelays={ndelays} nvox={Ytr.shape[1]} train_rows={Xtr.shape[0]}", flush=True)

    lean_cols, wh_cols = [], []
    for dd in range(ndelays):
        lean_cols += list(range(dd * TOTAL, dd * TOTAL + LEAN_RAW))
        wh_cols += list(range(dd * TOTAL + LEAN_RAW, (dd + 1) * TOTAL))
    lean_cols = np.array(lean_cols); wh_cols = np.array(wh_cols)

    def fit_perelec(cols):
        Xa = np.ascontiguousarray(Xtr[:, cols]); Xb = np.ascontiguousarray(Xte[:, cols])
        wt, _, _ = encoding.bootstrap_ridge_gpu(Xa, Ytr, Xb, Yte, ALPHAS)
        del Xa, Xb; gc.collect()
        return per_elec_canonical(fte, rte, cols, wt)

    print("fitting LEAN-only...", flush=True);    r_lean = fit_perelec(lean_cols)
    print("fitting Whisper-only...", flush=True); r_wh = fit_perelec(wh_cols)
    np.savez(os.path.join(os.path.dirname(__file__), "diag_perelec_canonical.npz"),
             r_lean=r_lean, r_wh=r_wh)

    adv = r_wh - r_lean
    def g(sl): return f"LEAN {r_lean[sl].mean():.4f}  Whisper {r_wh[sl].mean():.4f}  W-L {adv[sl].mean():+.4f}"
    print("\n=== CANONICAL test set, per-electrode canonical metric ===")
    print("ALL-166      :", g(slice(0, 166)))
    print("GRID 0-124   :", g(slice(0, 125)))
    print("STRIPS 125-31:", g(slice(125, 132)))
    print("DEPTH 132-165:", g(slice(132, 166)))


if __name__ == "__main__":
    main()
