"""Multi-subject LEAN-v20 variance partitioning and LEAN+Whisper ("Both") on each subject's held-out
session. Same code paths as patient1 (varpart-v20/phase1_stats.py, varpart-v20-newtest/phase2_newtest.py,
lean-whisper-newtest/both_varpart.py); the subject's directories, electrode count and frozen lists come
from <SUBJ>/split.json through src_multi (MS_SPLIT, set from the subject here).

  vp_stats SUBJ            stream the 500 training chunks once with canonical LEAN-v20 features ->
                           <SUBJ>/varpart/stats (XtX, XtY, bootstrap tune rows)
  vp_fit SUBJ i N          fits i::N of groups.fit_list() from the stats, then score them on the
                           held-out session -> <SUBJ>/varpart/fits/*.npz (r: n_test x n_elec)
  both_fit SUBJ            Both / LEAN-only / Whisper-only in one streamed pass -> <SUBJ>/both/perchunk_r.npz
  both_agg SUBJ            LEAN-vs-Whisper partitioning -> <SUBJ>/both/{RESULTS.md,results.csv,perelec.npz}

patient1VAL (original patient1 split, old 23-chunk test) checks these against stored patient1 results:
vp_fit asserts r equals varpart-v20/fits, both_fit prints the June values.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

MS = os.path.dirname(os.path.abspath(__file__))
RW = os.path.dirname(MS)
MODE, SUBJ = sys.argv[1], sys.argv[2]
os.environ['MS_SPLIT'] = os.path.join(MS, SUBJ, 'split.json')
CFG = json.load(open(os.environ['MS_SPLIT']))
sys.path.insert(0, os.path.join(MS, 'pkg'))                       # pkg/src -> src_multi
from src import data, encoding, features  # noqa: E402
sys.path.insert(0, os.path.join(RW, 'varpart-v20-newtest'))
from phase2_newtest import chunk_r, fit_weights  # noqa: E402      (also puts varpart-v20 on the path)
from phase1_stats import NBOOTS, CHUNKLEN, NCHUNKS, acquire_device, aligned_len  # noqa: E402
import groups as G  # noqa: E402
import lean_v20_enc as L  # noqa: E402
sys.path.insert(0, os.path.join(RW, 'lean-whisper-newtest'))
import both_varpart as BV  # noqa: E402

VP = os.path.join(MS, SUBJ, 'varpart')
ST = os.path.join(VP, 'stats')
FITS = os.path.join(VP, 'fits')
BOTH = os.path.join(MS, SUBJ, 'both')


def one(c, emb):
    f, r = features.get_features_per_chunk(data.load_stim([c]), data.load_responses([c]), [c], emb)
    return (f[0], r[0]) if f else (None, None)


def check_lists():
    train, test = data.get_chunk_names(500, 100)
    assert train == CFG['train'] and test == CFG['test']
    return train, test


def vp_stats():
    import torch
    t0 = time.time()
    dev = acquire_device()
    os.makedirs(ST, exist_ok=True)
    train, _ = check_lists()
    emb = L.build_embedder(device=dev, interact_set='v3e')
    lens = [aligned_len(c) for c in train]
    nresp = int(sum(lens))
    splits = encoding._temporal_chunk_splits(NBOOTS, nresp, CHUNKLEN, NCHUNKS)
    union = np.unique(np.concatenate([np.asarray(tu, dtype=np.int64) for _, tu in splits]))
    pos_of = {int(g): i for i, g in enumerate(union)}
    P, V = G.RAW_DIM * G.NDELAYS, data.N_ELECTRODES
    XtX = torch.zeros(P, P, dtype=torch.float64, device=dev)
    XtY = torch.zeros(P, V, dtype=torch.float64, device=dev)
    tune_X = np.zeros((len(union), P), np.float32)
    tune_Y = np.zeros((len(union), V), np.float32)
    off = 0
    for i, c in enumerate(train):
        X, Y = one(c, emb)
        n = 0 if X is None else X.shape[0]
        assert n == lens[i], f'{c}: rows {n} != {lens[i]}'
        if n == 0:
            continue
        assert X.shape[1] == P and Y.shape[1] == V, (X.shape, Y.shape)
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
    np.save(os.path.join(ST, 'XtX.npy'), XtX.cpu().numpy())
    np.save(os.path.join(ST, 'XtY.npy'), XtY.cpu().numpy())
    np.save(os.path.join(ST, 'tune_X.npy'), tune_X)
    np.save(os.path.join(ST, 'tune_Y.npy'), tune_Y)
    np.savez(os.path.join(ST, 'tune_pos.npz'), **{f's{b}': np.array([pos_of[int(g)] for g in tu], dtype=np.int64)
                                                  for b, (_, tu) in enumerate(splits)})
    json.dump(dict(subject=SUBJ, train=train, nresp=nresp, P=P, n_electrodes=V, seconds=time.time() - t0),
              open(os.path.join(ST, 'meta.json'), 'w'), indent=1)
    open(os.path.join(ST, 'DONE'), 'w').write(f'{time.time() - t0:.0f}s\n')
    print(f'stats done {time.time() - t0:.0f}s', flush=True)


def vp_fit(task, ntasks):
    import torch
    from phase2_fit import safe_name
    mine = [f for i, f in enumerate(G.fit_list()) if i % ntasks == task]
    todo = [(fid, raw) for fid, raw in mine if not os.path.exists(os.path.join(FITS, safe_name(fid) + '.npz'))]
    print(f'{SUBJ} task {task}/{ntasks}: {len(mine)} fits, {len(todo)} to do', flush=True)
    if not todo:
        return
    assert os.path.exists(os.path.join(ST, 'DONE')), 'stats not finished'
    os.makedirs(FITS, exist_ok=True)
    dev = acquire_device()
    t0 = time.time()
    _, test = check_lists()
    assert json.load(open(os.path.join(ST, 'meta.json')))['train'] == CFG['train']
    XtX = np.load(os.path.join(ST, 'XtX.npy'))
    XtY = torch.from_numpy(np.load(os.path.join(ST, 'XtY.npy'))).to(dev)
    tX = torch.from_numpy(np.load(os.path.join(ST, 'tune_X.npy'))).to(dev)
    tY = torch.from_numpy(np.load(os.path.join(ST, 'tune_Y.npy'))).to(dev)
    tp = np.load(os.path.join(ST, 'tune_pos.npz'))
    tune_pos = [tp[f's{b}'] for b in range(len(tp.files))]
    W, cols_of, ab_of = {}, {}, {}
    for fid, raw in todo:
        cols = G.delayed(raw)
        W[fid], ab_of[fid] = fit_weights(cols, XtX, XtY, tX, tY, tune_pos, dev)
        cols_of[fid] = torch.as_tensor(cols, device=dev)
        torch.cuda.empty_cache()
    del tX, tY
    print(f'fitted {len(todo)} {time.time() - t0:.0f}s', flush=True)
    emb = L.build_embedder(device=dev, interact_set='v3e')
    rr = {fid: [] for fid in W}
    kept = []
    for c in test:
        X, Y = one(c, emb)
        if X is None:
            continue
        Xg = torch.from_numpy(X).to(dev).double(); Yg = torch.from_numpy(Y).to(dev).double()
        for fid in W:
            rr[fid].append(chunk_r(Xg.index_select(1, cols_of[fid]), Yg, W[fid]))
        kept.append(c)
    for fid, raw in todo:
        r = np.stack(rr[fid])
        if SUBJ == 'patient1VAL':
            ref = np.load(os.path.join(RW, 'varpart-v20', 'fits', safe_name(fid) + '.npz'))['r_chunk_elec']
            err = float(np.abs(r - ref).max())
            assert err < 1e-6, f'{fid}: differs from varpart-v20 by {err}'
            print(f'  {fid[:60]:60s} matches varpart-v20 (max|dr| {err:.1e})', flush=True)
        out = os.path.join(FITS, safe_name(fid) + '.npz')
        np.savez(out + '.tmp.npz', fit_id=fid, raw_cols=raw, r_chunk_elec=r, chunks=np.array(kept),
                 alphas_best=ab_of[fid])
        os.replace(out + '.tmp.npz', out)
    full = [fid for fid in W if fid == 'full']
    if full:
        m = np.stack(rr['full']).mean(0)
        print(f'full: grid {m[:CFG["n_grid"]].mean():.4f} all {m.mean():.4f}', flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


class BothSubj(BV.Both):
    """LEAN-v20 + Whisper for this subject; the fingerprint map covers its train + test chunks."""
    def __init__(self, base):
        self.base, self.wav_dir = base, CFG['whisper_dir']
        win = int(getattr(base, 'win', 2048)); s0 = (4096 - win) // 2
        self.fp = {}
        for c in CFG['train'] + CFG['test']:
            seg = np.load(os.path.join(CFG['wave_dir'], f'{c}_wave.npy'), mmap_mode='r')
            w0 = np.ascontiguousarray(np.asarray(seg[s0:s0 + win]).astype(np.float32))
            self.fp[hashlib.md5(w0.tobytes()).hexdigest()] = c
        assert len(self.fp) == len(CFG['train']) + len(CFG['test']), 'fingerprint collision'


def both_fit():
    import torch
    t0 = time.time()
    dev = acquire_device()
    os.makedirs(BOTH, exist_ok=True)
    train, test = check_lists()
    emb = BothSubj(L.build_embedder(device=dev, interact_set='v3e'))
    P = BV.NDEL * BV.D
    lens = [aligned_len(c) for c in train]
    nresp = int(sum(lens))
    splits = encoding._temporal_chunk_splits(5, nresp, 40, 20)
    union = np.unique(np.concatenate([np.asarray(tu, dtype=np.int64) for _, tu in splits]))
    pos_of = {int(g): i for i, g in enumerate(union)}
    V = data.N_ELECTRODES
    XtX = torch.zeros(P, P, dtype=torch.float64, device=dev)
    XtY = torch.zeros(P, V, dtype=torch.float64, device=dev)
    tune_X = np.zeros((len(union), P), np.float32); tune_Y = np.zeros((len(union), V), np.float32)
    off = 0
    for i, c in enumerate(train):
        X, Y = one(c, emb)
        n = 0 if X is None else X.shape[0]
        assert n == lens[i]
        if n == 0:
            continue
        assert X.shape[1] == P
        Xg = torch.from_numpy(X).to(dev, torch.float64)
        XtX += Xg.T @ Xg
        XtY += Xg.T @ torch.from_numpy(Y).to(dev, torch.float64)
        del Xg
        lo, hi = np.searchsorted(union, [off, off + n])
        for g in union[lo:hi]:
            tune_X[pos_of[int(g)]] = X[g - off]; tune_Y[pos_of[int(g)]] = Y[g - off]
        off += n
        if i % 100 == 0:
            print(f'  train {i + 1}/{len(train)} {time.time() - t0:.0f}s', flush=True)
    XtX_host = XtX.cpu().numpy(); del XtX; torch.cuda.empty_cache()
    tX = torch.from_numpy(tune_X).to(dev); tY = torch.from_numpy(tune_Y).to(dev)
    tune_pos = [np.array([pos_of[int(g)] for g in tu], dtype=np.int64) for _, tu in splits]
    cols = BV.model_cols()
    W, ab, idx = {}, {}, {}
    for m in BV.MODELS:
        W[m], ab[m] = fit_weights(cols[m], XtX_host, XtY, tX, tY, tune_pos, dev)
        idx[m] = torch.as_tensor(cols[m], device=dev)
        torch.cuda.empty_cache()
    del tX, tY
    rr = {m: [] for m in BV.MODELS}
    kept = []
    for c in test:
        X, Y = one(c, emb)
        if X is None:
            continue
        Xg = torch.from_numpy(X).to(dev).double(); Yg = torch.from_numpy(Y).to(dev).double()
        for m in BV.MODELS:
            rr[m].append(chunk_r(Xg.index_select(1, idx[m]), Yg, W[m]))
        kept.append(c)
    out = {f'r_{m}': np.stack(rr[m]) for m in BV.MODELS}
    out.update({f'alphas_{m}': ab[m] for m in BV.MODELS}, chunks=np.array(kept))
    np.savez(os.path.join(BOTH, 'perchunk_r.npz'), **out)
    for m in BV.MODELS:
        e = out[f'r_{m}'].mean(0)
        ref = f" (June {BV.OLD_REF[m]['all166']}/{BV.OLD_REF[m]['grid']})" if SUBJ == 'patient1VAL' else ''
        print(f'{m:8s} all {e.mean():.5f} grid {e[:CFG["n_grid"]].mean():.5f}{ref}', flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


def both_agg():
    z = np.load(os.path.join(BOTH, 'perchunk_r.npz'))
    rr = {m: z[f'r_{m}'] for m in BV.MODELS}
    n = rr['both'].shape[0]
    e, R, P = BV.partition(rr)
    rng = np.random.RandomState(0)
    boots = [BV.partition(rr, rng.randint(0, n, n)) for _ in range(2000)]
    esets = {'grid': slice(0, CFG['n_grid']), 'all': slice(0, CFG['n_electrodes'])}
    lines = [f'# {SUBJ}: LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning', '',
             f'Held-out session {CFG["held_out_session"]} ({n} chunks); 500 training chunks from the other sessions; '
             f'{CFG["n_electrodes"]} electrodes, grid = first {CFG["n_grid"]}. Same definitions as '
             'runs-wave/lean-whisper-newtest (unique/shared as % of R²(Both), 95% CI from a paired chunk bootstrap).', '',
             '| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |',
             '|---|---|---|---|---|---|---|---|']
    rows = []
    for k, es in esets.items():
        tot = float(R['both'][es].mean())
        row = dict(subject=SUBJ, electrodes=k, n_chunks=n, n_electrodes=es.stop - es.start,
                   r_both=float(e['both'][es].mean()), r_lean=float(e['lean'][es].mean()),
                   r_whisper=float(e['whisper'][es].mean()), R2_both=tot,
                   R2_lean=float(R['lean'][es].mean()), R2_whisper=float(R['whisper'][es].mean()))
        cells = []
        for key in ('unique_lean', 'unique_whisper', 'shared'):
            v = float(P[key][es].mean())
            bs = np.array([float(b[2][key][es].mean()) for b in boots])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.update({key: v, key + '_lo': lo, key + '_hi': hi, key + '_share': v / tot})
            cells.append(f'{100 * v / tot:.1f}% [{100 * lo / tot:.1f}, {100 * hi / tot:.1f}]')
        rows.append(row)
        lines.append(f'| {k} | {row["r_both"]:.4f} | {row["r_lean"]:.4f} | {row["r_whisper"]:.4f} | '
                     f'{100 * tot:.3f} | ' + ' | '.join(cells) + ' |')
    with open(os.path.join(BOTH, 'results.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(os.path.join(BOTH, 'perelec.npz'), **{f'r_{m}': e[m] for m in BV.MODELS},
             **{f'R2_{m}': R[m] for m in BV.MODELS}, **P)
    open(os.path.join(BOTH, 'RESULTS.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if MODE == 'vp_stats':
        vp_stats()
    elif MODE == 'vp_fit':
        vp_fit(int(sys.argv[3]), int(sys.argv[4]))
    elif MODE == 'both_fit':
        both_fit()
    elif MODE == 'both_agg':
        both_agg()
