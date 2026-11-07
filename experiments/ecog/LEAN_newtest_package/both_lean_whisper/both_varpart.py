"""LEAN-v20 + Whisper ("Both") and the LEAN-vs-Whisper variance partitioning, on the old 23-chunk test
and the new held-out test (runs-wave/newtest).

Features: canonical LEAN-v20 (lean_v20_enc, interact_set='v3e', 3,852 raw) with the 1,280 Whisper
features appended per frame, exactly as lean-whisper/diag_combined_v20.py (chunk found by the MD5 of
its first 4,096-sample window; the June map is extended with the new chunks, and an unknown chunk
raises instead of returning zeros). The harness delay-embeds (4 delays, delay-major) and z-scores
per chunk. One streamed pass accumulates the training statistics; then three fits use the same
harness ridge (per-electrode alpha by bootstrapped CV) on column subsets:
  both = all 20,528 columns, lean = LEAN columns of each delay, whisper = Whisper columns.
Per-chunk z-scoring is per column, so the subset fits equal standalone LEAN-only / Whisper-only
fits (as in lean-whisper/diag_canonical.py). Each model is scored on the old test (must reproduce
the June values) and on all 493 new candidates.

Variance partitioning per electrode (R2_e = r_e|r_e|, r_e = chunk-averaged r):
  unique LEAN = R2(both) - R2(whisper); unique Whisper = R2(both) - R2(lean);
  shared = R2(lean) + R2(whisper) - R2(both). Electrode-set summaries average R2_e; 95% CIs from a
  paired bootstrap over test chunks (B=2000, same resamples for all three models).

usage: both_varpart.py fit        (GPU; writes perchunk_r.npz)
       both_varpart.py aggregate  (CPU; writes RESULTS.md, results.csv, perelec.npz)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RW = os.path.dirname(HERE)
NT = os.path.join(RW, 'newtest')
FP_PATH = os.path.join(os.path.dirname(RW), 'evolve-wave', 'ling_features', 'fingerprint_map.json')
N_LEAN, N_WH, NDEL = 3852, 1280, 4
D = N_LEAN + N_WH
OLD_REF = {  # June values, old test (lean-whisper/diag_*.out, varpart-v20 full fit)
    'both': {'all166': 0.0920, 'grid': 0.1038},
    'lean': {'all166': 0.0808, 'grid': 0.0904},
    'whisper': {'all166': 0.0817, 'grid': 0.0933},
}
TOL = 0.00015
MODELS = ('both', 'lean', 'whisper')
ESETS = {'grid': slice(0, 125), 'all166': slice(0, 166)}


def model_cols():
    lean = np.concatenate([d * D + np.arange(N_LEAN) for d in range(NDEL)])
    wh = np.concatenate([d * D + N_LEAN + np.arange(N_WH) for d in range(NDEL)])
    return {'both': np.arange(NDEL * D), 'lean': lean, 'whisper': wh}


class Both:
    def __init__(self, base, wav_dir):
        self.base, self.wav_dir = base, wav_dir
        self.fp = json.load(open(FP_PATH))
        win = int(getattr(base, 'win', 2048))
        s0 = (4096 - win) // 2               # data.frame_windows: first window = seg[s0:s0+win]
        for c in json.load(open(os.path.join(NT, 'test_candidates.json'))):
            seg = np.load(os.path.join(NT, 'wave_aligned', f'{c}_wave.npy'), mmap_mode='r')
            w0 = np.ascontiguousarray(np.asarray(seg[s0:s0 + win]).astype(np.float32))
            self.fp[hashlib.md5(w0.tobytes()).hexdigest()] = c

    def __getattr__(self, name):  # forward .win and other harness-read attributes
        return getattr(object.__getattribute__(self, 'base'), name)

    def __call__(self, windows):
        lean = np.asarray(self.base(windows), dtype=np.float32)
        T = lean.shape[0]
        w0 = np.ascontiguousarray(np.asarray(windows, dtype=np.float32)[0])
        c = self.fp.get(hashlib.md5(w0.tobytes()).hexdigest())
        if c is None:
            raise KeyError('waveform fingerprint not in map')
        wh = np.load(os.path.join(self.wav_dir, f'{c}_whisper.npy')).astype(np.float32)
        wh = wh[:T] if wh.shape[0] >= T else np.pad(wh, ((0, T - wh.shape[0]), (0, 0)))
        return np.concatenate([lean, wh], axis=1)


def fit():
    import torch
    sys.path.insert(0, HERE)
    from src import data, encoding, features
    import lean_v20_enc as L
    sys.path.insert(0, os.path.join(RW, 'varpart-v20-newtest'))
    from phase2_newtest import chunk_r, fit_weights
    from phase1_stats import acquire_device, aligned_len

    t0 = time.time()
    dev = acquire_device()
    train, test = data.get_chunk_names(500, 100)
    assert train == json.load(open(os.path.join(NT, 'train_frozen.json')))
    assert test == json.load(open(os.path.join(NT, 'test_old.json')))
    cand = json.load(open(os.path.join(NT, 'test_candidates.json')))
    emb = Both(L.build_embedder(device=dev, interact_set='v3e'), data.WAV_DIR)
    P = NDEL * D

    def one(c, new=False):
        stim = ({c: np.load(os.path.join(NT, 'wave_aligned', f'{c}_wave.npy')).astype(np.float32)}
                if new else data.load_stim([c]))
        f, r = features.get_features_per_chunk(stim, data.load_responses([c]), [c], emb)
        return (f[0], r[0]) if f else (None, None)

    lens = [aligned_len(c) for c in train]
    nresp = int(sum(lens))
    splits = encoding._temporal_chunk_splits(5, nresp, 40, 20)
    union = np.unique(np.concatenate([np.asarray(tu, dtype=np.int64) for _, tu in splits]))
    pos_of = {int(g): i for i, g in enumerate(union)}
    XtX = torch.zeros(P, P, dtype=torch.float64, device=dev)
    XtY = torch.zeros(P, data.N_ELECTRODES, dtype=torch.float64, device=dev)
    tune_X = np.zeros((len(union), P), np.float32)
    tune_Y = np.zeros((len(union), data.N_ELECTRODES), np.float32)
    off = 0
    for i, c in enumerate(train):
        X, Y = one(c)
        n = 0 if X is None else X.shape[0]
        assert n == lens[i], f'{c}: {n} rows != {lens[i]}'
        if n == 0:
            continue
        assert X.shape[1] == P, X.shape
        Xg = torch.from_numpy(X).to(dev, torch.float64)
        XtX += Xg.T @ Xg
        XtY += Xg.T @ torch.from_numpy(Y).to(dev, torch.float64)
        del Xg
        lo, hi = np.searchsorted(union, [off, off + n])
        for g in union[lo:hi]:
            tune_X[pos_of[int(g)]] = X[g - off]
            tune_Y[pos_of[int(g)]] = Y[g - off]
        off += n
        if i % 50 == 0:
            print(f'  train {i + 1}/{len(train)} {time.time() - t0:.0f}s', flush=True)
    assert off == nresp
    XtX_host = XtX.cpu().numpy()
    del XtX
    torch.cuda.empty_cache()
    tX = torch.from_numpy(tune_X).to(dev); tY = torch.from_numpy(tune_Y).to(dev)
    tune_pos = [np.array([pos_of[int(g)] for g in tu], dtype=np.int64) for _, tu in splits]

    cols = model_cols()
    W, ab, idx = {}, {}, {}
    for m in MODELS:
        W[m], ab[m] = fit_weights(cols[m], XtX_host, XtY, tX, tY, tune_pos, dev)
        idx[m] = torch.as_tensor(cols[m], device=dev)
        print(f'fit {m}: {len(cols[m])} columns {time.time() - t0:.0f}s', flush=True)
        torch.cuda.empty_cache()
    del tX, tY

    out = {f'alphas_{m}': ab[m] for m in MODELS}
    for name, chunks, new in (('old', test, False), ('new', cand, True)):
        rr = {m: [] for m in MODELS}
        kept = []
        for c in chunks:
            X, Y = one(c, new)
            if X is None:
                continue
            Xg = torch.from_numpy(X).to(dev).double(); Yg = torch.from_numpy(Y).to(dev).double()
            for m in MODELS:
                rr[m].append(chunk_r(Xg.index_select(1, idx[m]), Yg, W[m]))
            kept.append(c)
            del Xg, Yg
        for m in MODELS:
            out[f'r_{name}_{m}'] = np.stack(rr[m])
        out[f'chunks_{name}'] = np.array(kept)
        print(f'scored {name} test: {len(kept)} chunks {time.time() - t0:.0f}s', flush=True)
    np.savez(os.path.join(HERE, 'perchunk_r.npz'), **out)
    for m in MODELS:
        e = out[f'r_old_{m}'].mean(0)
        print(f'{m:8s} old all166 {e.mean():.5f} (June {OLD_REF[m]["all166"]}) grid {e[:125].mean():.5f} '
              f'(June {OLD_REF[m]["grid"]})', flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


def r2(r):
    return r * np.abs(r)


def partition(rr, chunk_idx=None):
    """rr: {model: (n_chunks, 166)} -> per-electrode r and R2 of each model, unique and shared."""
    e = {m: (v if chunk_idx is None else v[chunk_idx]).mean(0) for m, v in rr.items()}
    R = {m: r2(v) for m, v in e.items()}
    return e, R, {'unique_lean': R['both'] - R['whisper'], 'unique_whisper': R['both'] - R['lean'],
                  'shared': R['lean'] + R['whisper'] - R['both']}


def aggregate():
    z = np.load(os.path.join(HERE, 'perchunk_r.npz'))
    prim = set(json.load(open(os.path.join(NT, 'test_primary.json'))))
    cn = [str(c) for c in z['chunks_new']]
    pm = np.array([c in prim for c in cn])
    sets = {'old (23 chunks)': {m: z[f'r_old_{m}'] for m in MODELS},
            'new primary (317 chunks)': {m: z[f'r_new_{m}'][pm] for m in MODELS},
            'new all (493 chunks)': {m: z[f'r_new_{m}'] for m in MODELS}}
    rows, perelec = [], {}
    lines = ['# LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning', '',
             'Same 500 training chunks and harness ridge as every LEAN fit; Both = LEAN-v20 (3,852) + '
             'Whisper (1,280) features in one ridge (4 delays). LEAN-only and Whisper-only are column subsets '
             'of the same pass. R² per electrode = r·|r| (r = chunk-averaged per-chunk Pearson r), averaged over '
             'electrodes. Unique LEAN = R²(Both) − R²(Whisper); unique Whisper = R²(Both) − R²(LEAN); '
             'shared = R²(LEAN) + R²(Whisper) − R²(Both). 95% CIs: paired bootstrap over test chunks (B = 2000). '
             'Primary new test = runs-wave/newtest/test_primary.json.', '']
    old_e = {m: z[f'r_old_{m}'].mean(0) for m in MODELS}
    val = []
    for m in MODELS:
        for k, es in ESETS.items():
            v = float(old_e[m][es].mean())
            ok = abs(v - OLD_REF[m][k]) <= TOL
            val.append(f'- {m}, {k}: {v:.5f} vs June {OLD_REF[m][k]:.4f} — {"reproduced" if ok else "**MISMATCH**"}')
    lines += ['## Old-test reproduction', ''] + val + ['']
    for sname, rr in sets.items():
        n = next(iter(rr.values())).shape[0]
        e, R, P = partition(rr)
        rng = np.random.RandomState(0)
        boots = [partition(rr, rng.randint(0, n, n)) for _ in range(2000)]
        tag = sname.split(' (')[0].replace(' ', '_')
        perelec[tag] = dict(**{f'r_{m}': e[m] for m in MODELS}, **{f'R2_{m}': R[m] for m in MODELS}, **P)
        lines += [f'## {sname}', '', '| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |',
                  '|---|---|---|---|---|---|---|---|']
        for k, es in ESETS.items():
            tot = float(R['both'][es].mean())

            def stat(key, src='P'):
                val = float((P if src == 'P' else R)[key][es].mean())
                bs = np.array([float((b[2] if src == 'P' else b[1])[key][es].mean()) for b in boots])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                return val, lo, hi

            cells = []
            row = dict(test=sname, electrodes=k, n_chunks=n,
                       r_both=float(e['both'][es].mean()), r_lean=float(e['lean'][es].mean()),
                       r_whisper=float(e['whisper'][es].mean()), R2_both=tot)
            for key in ('unique_lean', 'unique_whisper', 'shared'):
                v, lo, hi = stat(key)
                row.update({key: v, key + '_lo': lo, key + '_hi': hi, key + '_share': v / tot})
                cells.append(f'{100 * v / tot:.1f}% [{100 * lo / tot:.1f}, {100 * hi / tot:.1f}]')
            for m in ('lean', 'whisper'):
                row[f'R2_{m}'] = float(R[m][es].mean())
            rows.append(row)
            lines.append(f'| {k} | {row["r_both"]:.4f} | {row["r_lean"]:.4f} | {row["r_whisper"]:.4f} | '
                         f'{100 * tot:.3f} | ' + ' | '.join(cells) + ' |')
        lines += ['', 'Unique/shared as % of R²(Both) [95% CI].', '']
    with open(os.path.join(HERE, 'results.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.savez(os.path.join(HERE, 'perelec.npz'),
             **{f'{t}__{k}': v for t, d in perelec.items() for k, v in d.items()})
    open(os.path.join(HERE, 'RESULTS.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'fit': fit, 'aggregate': aggregate}[sys.argv[1]]()
