"""Per-electrode R for the COMBINED LEAN-v20 + Whisper model on the CANONICAL test set.
Wraps the v20 embedder to append Whisper features (same fingerprint injection as
wave_encoder.py) BEFORE the harness delay-embeds, fits one joint ridge, scores
per-electrode with the canonical per-chunk metric. Whisper r_wh reused for the diff."""
import sys, os, gc, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from src import data, features, encoding
import lean_v20_enc as L

ALPHAS = np.logspace(1, 4, 12)
WHISPER_DIR = "/path/to/ecog_data/wav_files"
FP_PATH = ("/path/to/ecog_data/"
           "circuit-building-autoresearch/evolve-wave/ling_features/fingerprint_map.json")
N_WHISPER = 1280


class V20PlusWhisper:
    def __init__(self, base):
        self.base = base
        self.fp_map = json.load(open(FP_PATH))
        self._cache = {}

    def __getattr__(self, name):  # forward .win and any harness-read attrs to base
        return getattr(object.__getattribute__(self, "base"), name)

    def __call__(self, windows):
        lean = np.asarray(self.base(windows), dtype=np.float32)  # (T, D_lean)
        T = lean.shape[0]
        w0 = np.ascontiguousarray(np.asarray(windows, dtype=np.float32)[0])
        chunk = self.fp_map.get(hashlib.md5(w0.tobytes()).hexdigest())
        if chunk is None:
            wh = np.zeros((T, N_WHISPER), dtype=np.float32)
        else:
            arr = self._cache.get(chunk)
            if arr is None:
                arr = np.load(os.path.join(WHISPER_DIR, f"{chunk}_whisper.npy")).astype(np.float32)
                self._cache[chunk] = arr
            arr = arr[:T] if arr.shape[0] >= T else np.pad(arr, ((0, T - arr.shape[0]), (0, 0)))
            wh = arr
        return np.concatenate([lean, wh], axis=1)


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
    emb = V20PlusWhisper(L.build_embedder(device="cuda", interact_set="v3e"))
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    del stim, resp; gc.collect()
    Xtr = np.vstack(ftr).astype(np.float32); Ytr = np.vstack(rtr).astype(np.float32)
    del ftr, rtr; gc.collect()  # free 140+GB of train feature list before building test/ridge
    Xte = np.vstack(fte).astype(np.float32); Yte = np.vstack(rte).astype(np.float32)
    print(f"p={Xtr.shape[1]} nvox={Ytr.shape[1]} train_rows={Xtr.shape[0]} Xtr={Xtr.nbytes/1e9:.0f}GB", flush=True)
    wt, _, _ = encoding.bootstrap_ridge_gpu(Xtr, Ytr, Xte, Yte, ALPHAS)
    del Xtr, Xte; gc.collect()
    r_comb = per_elec_canonical(fte, rte, wt)
    np.savez(os.path.join(os.path.dirname(__file__), "diag_perelec_comb_v20.npz"), r_comb=r_comb)

    r_wh = np.load(os.path.join(os.path.dirname(__file__), "diag_perelec_canonical.npz"))["r_wh"]
    adv = r_comb - r_wh
    def g(sl): return f"comb {r_comb[sl].mean():.4f}  Whisper {r_wh[sl].mean():.4f}  C-W {adv[sl].mean():+.4f}"
    print("\n=== COMBINED LEAN-v20+Whisper vs Whisper (canonical test set) ===")
    print("ALL-166      :", g(slice(0, 166)))
    print("GRID 0-124   :", g(slice(0, 125)))
    print("STRIPS 125-31:", g(slice(125, 132)))
    print("DEPTH 132-165:", g(slice(132, 166)))


if __name__ == "__main__":
    main()
