"""Variance partitioning of LEAN-v20 on the NEW held-out test set (runs-wave/newtest).

Training is unchanged, so every fit reuses the phase-1 training statistics in
../varpart-v20/stats (XtX, XtY, bootstrap tune rows). Per fit: refit exactly as
varpart-v20/phase2_fit.py (same code, returning the weights), assert the old 23-chunk test r
equals the stored ../varpart-v20/fits result (proves identical weights), then score the same
weights on the 493 new candidate chunks with canonical v20 features (interact_set='v3e',
features.get_features_per_chunk, segments from newtest/wave_aligned). Saves r for the primary
new test (317 chunks, `r_chunk_elec`, read by phase3_aggregate.py), all candidates (`r_all`)
and the old test (`r_old`).

usage: phase2_newtest.py --task i --ntasks N      (round-robin over groups.fit_list())
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(os.path.dirname(HERE), 'varpart-v20')
NT = os.path.join(os.path.dirname(HERE), 'newtest')
sys.path.insert(0, HERE)
from src import data, encoding, features  # noqa: E402
import groups as G  # noqa: E402
import lean_v20_enc as L  # noqa: E402
sys.path.insert(0, OLD)
from phase1_stats import acquire_device  # noqa: E402
from phase2_fit import ALPHAS, SINGCUT, safe_name  # noqa: E402

ST = os.path.join(OLD, 'stats')
FITS = os.path.join(HERE, 'fits')


def fit_weights(cols, XtX, XtY, tune_X, tune_Y, tune_pos, dev):
    """varpart-v20/phase2_fit.fit_subset up to the final weights (identical code)."""
    idx = torch.as_tensor(cols, device=dev)
    XtX_S = torch.from_numpy(XtX[np.ix_(cols, cols)]).to(dev)
    XtY_S = XtY.index_select(0, idx)
    nvox = XtY.shape[1]
    a2 = torch.tensor(ALPHAS ** 2, dtype=torch.float64, device=dev)
    all_corrs = torch.zeros(len(ALPHAS), nvox, dtype=torch.float64, device=dev)
    for pos in tune_pos:
        p = torch.as_tensor(pos, device=dev)
        Xtu = tune_X.index_select(0, p).index_select(1, idx).double()
        Ytu = tune_Y.index_select(0, p).double()
        XtX_sub = Xtu.T @ Xtu
        XtX_sub.neg_().add_(XtX_S)
        V, Lam = encoding._eig_good(XtX_sub, SINGCUT)
        del XtX_sub
        A = Xtu @ V
        B = V.T @ (XtY_S - Xtu.T @ Ytu)
        zY = encoding._zs_t(Ytu)
        for ai in range(len(ALPHAS)):
            pred = A @ (B / (Lam + a2[ai]).unsqueeze(1))
            all_corrs[ai] += torch.nan_to_num((zY * encoding._zs_t(pred)).mean(0))
        del Xtu, Ytu, V, Lam, A, B, zY
    best = torch.argmax(all_corrs, dim=0).cpu().numpy()
    alphas_best = ALPHAS[best]
    V, Lam = encoding._eig_good(XtX_S, SINGCUT)
    B = V.T @ XtY_S
    wt = torch.zeros(len(cols), nvox, dtype=torch.float64, device=dev)
    for ua in np.unique(alphas_best):
        sel = torch.as_tensor(np.nonzero(alphas_best == ua)[0], device=dev)
        wt[:, sel] = V @ (B[:, sel] / (Lam + float(ua) ** 2).unsqueeze(1))
    del V, Lam, B, XtX_S
    return wt, alphas_best


def chunk_r(X, Y, wt):
    pred = X @ wt
    a = Y - Y.mean(0); b = pred - pred.mean(0)
    den = torch.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    rr = torch.where(den == 0, torch.zeros_like(den), (a * b).sum(0) / torch.where(den == 0, torch.ones_like(den), den))
    return torch.nan_to_num(rr).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--ntasks', type=int, required=True)
    args = ap.parse_args()
    mine = [f for i, f in enumerate(G.fit_list()) if i % args.ntasks == args.task]
    todo = [(fid, raw) for fid, raw in mine if not os.path.exists(os.path.join(FITS, safe_name(fid) + '.npz'))]
    print(f'task {args.task}/{args.ntasks}: {len(mine)} fits, {len(todo)} to do', flush=True)
    if not todo:
        return
    dev = acquire_device()
    t0 = time.time()
    meta = json.load(open(os.path.join(ST, 'meta.json')))
    assert meta['train'] == json.load(open(os.path.join(NT, 'train_frozen.json')))
    XtX = np.load(os.path.join(ST, 'XtX.npy'))
    XtY = torch.from_numpy(np.load(os.path.join(ST, 'XtY.npy'))).to(dev)
    tune_X = torch.from_numpy(np.load(os.path.join(ST, 'tune_X.npy'))).to(dev)
    tune_Y = torch.from_numpy(np.load(os.path.join(ST, 'tune_Y.npy'))).to(dev)
    tp = np.load(os.path.join(ST, 'tune_pos.npz'))
    tune_pos = [tp[f's{b}'] for b in range(len(tp.files))]
    test_X = np.load(os.path.join(ST, 'test_X.npy'), mmap_mode='r')
    test_Y = np.load(os.path.join(ST, 'test_Y.npy'))
    offsets = np.load(os.path.join(ST, 'test_offsets.npy'))

    W, cols_of, ab_of, rold_of = {}, {}, {}, {}
    for fid, raw in todo:
        cols = G.delayed(raw)
        wt, ab = fit_weights(cols, XtX, XtY, tune_X, tune_Y, tune_pos, dev)
        idx = torch.as_tensor(cols, device=dev)
        ro = np.stack([chunk_r(torch.from_numpy(np.ascontiguousarray(test_X[offsets[c]:offsets[c + 1]])).to(dev)
                               .index_select(1, idx).double(),
                               torch.from_numpy(test_Y[offsets[c]:offsets[c + 1]]).to(dev).double(), wt)
                       for c in range(len(offsets) - 1)])
        ref = np.load(os.path.join(OLD, 'fits', safe_name(fid) + '.npz'))['r_chunk_elec']
        err = float(np.abs(ro - ref).max())
        assert err < 1e-6, f'{fid}: old-test r differs from varpart-v20 by {err}'
        W[fid], cols_of[fid], ab_of[fid], rold_of[fid] = wt, idx, ab, ro
        print(f'refit {fid[:70]:70s} old-test max|dr|={err:.1e} {time.time() - t0:.0f}s', flush=True)
        torch.cuda.empty_cache()
    del tune_X, tune_Y

    emb = L.build_embedder(device=dev, interact_set='v3e')
    cand = json.load(open(os.path.join(NT, 'test_candidates.json')))
    prim = set(json.load(open(os.path.join(NT, 'test_primary.json'))))
    rnew = {fid: [] for fid in W}
    kept = []
    for i, c in enumerate(cand):
        stim = {c: np.load(os.path.join(NT, 'wave_aligned', f'{c}_wave.npy')).astype(np.float32)}
        f, r = features.get_features_per_chunk(stim, data.load_responses([c]), [c], emb)
        if not f:
            continue
        X = torch.from_numpy(f[0]).to(dev).double(); Y = torch.from_numpy(r[0]).to(dev).double()
        assert X.shape[1] == G.RAW_DIM * G.NDELAYS
        for fid in W:
            rnew[fid].append(chunk_r(X.index_select(1, cols_of[fid]), Y, W[fid]))
        kept.append(c)
        del X, Y
        if i % 50 == 0:
            print(f'  new test {i + 1}/{len(cand)} {time.time() - t0:.0f}s', flush=True)
    pm = np.array([c in prim for c in kept])
    for fid, raw in todo:
        ra = np.stack(rnew[fid])
        out = os.path.join(FITS, safe_name(fid) + '.npz')
        np.savez(out + '.tmp.npz', fit_id=fid, raw_cols=raw, r_chunk_elec=ra[pm], r_all=ra,
                 chunks=np.array(kept), r_old=rold_of[fid], alphas_best=ab_of[fid])
        os.replace(out + '.tmp.npz', out)
        m = ra[pm].mean(0)
        print(f'{fid[:80]:80s} new primary grid={m[:125].mean():.4f} all166={m.mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
