"""Phase 2: ridge fits for column subsets of LEAN-v21, from the phase-1 statistics.

Same algorithm as src/encoding.bootstrap_ridge_gpu (per-electrode alpha chosen by
mean tune-set correlation over the same 5 bootstrap splits, alphas logspace(1,4,12),
final weights from the full-train XtX eigendecomposition), restricted to the
columns of each subset. Test metric = canonical per-chunk per-electrode Pearson r
(src/eval._per_chunk_elec_corr), saved unaveraged as (n_test_chunks, 166).

usage: phase2_fit.py --task i --ntasks N      (round-robin over groups.fit_list())
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import encoding  # noqa: E402
import groups as G  # noqa: E402
from phase1_stats import acquire_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, 'stats')
FITS = os.path.join(HERE, 'fits')
ALPHAS = np.logspace(1, 4, 12)
SINGCUT = 1e-10


def safe_name(fit_id: str) -> str:
    return fit_id.split(':')[0] + '_' + hashlib.md5(fit_id.encode()).hexdigest()[:12]


def eig_good(M):
    """encoding._eig_good, falling back to a CPU eigendecomposition (same float64 math)
    if a 16 GB GPU runs out of memory."""
    try:
        return encoding._eig_good(M, SINGCUT)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print('  [eigh] GPU OOM -> CPU fallback', flush=True)
        V, Lam = encoding._eig_good(M.cpu(), SINGCUT)
        return V.to(M.device), Lam.to(M.device)


def fit_subset(cols, XtX, XtY, tune_X, tune_Y, tune_pos, test_X, test_Y, offsets, dev):
    # XtX stays in host RAM; only the subset block goes to the GPU (peak ~10 GB at full
    # size, so 16 GB V100s work). All ridge math is float64, as in the harness.
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
        V, Lam = eig_good(XtX_sub)
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
    V, Lam = eig_good(XtX_S)
    B = V.T @ XtY_S
    wt = torch.zeros(len(cols), nvox, dtype=torch.float64, device=dev)
    for ua in np.unique(alphas_best):
        sel = torch.as_tensor(np.nonzero(alphas_best == ua)[0], device=dev)
        wt[:, sel] = V @ (B[:, sel] / (Lam + float(ua) ** 2).unsqueeze(1))
    del V, Lam, B, XtX_S
    rr_all = []
    for c in range(len(offsets) - 1):
        f = torch.from_numpy(np.ascontiguousarray(test_X[offsets[c]:offsets[c + 1]])).to(dev).index_select(1, idx).double()
        r = torch.from_numpy(test_Y[offsets[c]:offsets[c + 1]]).to(dev).double()
        pred = f @ wt
        a = r - r.mean(0); b = pred - pred.mean(0)
        den = torch.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        rr = torch.where(den == 0, torch.zeros_like(den), (a * b).sum(0) / torch.where(den == 0, torch.ones_like(den), den))
        rr_all.append(torch.nan_to_num(rr).cpu().numpy())
    return np.stack(rr_all), alphas_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--ntasks', type=int, required=True)
    args = ap.parse_args()
    assert os.path.exists(os.path.join(ST, 'DONE')), 'phase 1 not finished'
    os.makedirs(FITS, exist_ok=True)
    mine = [f for i, f in enumerate(G.fit_list()) if i % args.ntasks == args.task]
    todo = [(fid, cols) for fid, cols in mine
            if not os.path.exists(os.path.join(FITS, safe_name(fid) + '.npz'))]
    print(f'task {args.task}/{args.ntasks}: {len(mine)} fits, {len(todo)} to do', flush=True)
    if not todo:
        return
    dev = acquire_device()
    t0 = time.time()
    XtX = np.load(os.path.join(ST, 'XtX.npy'))
    XtY = torch.from_numpy(np.load(os.path.join(ST, 'XtY.npy'))).to(dev)
    tune_X = torch.from_numpy(np.load(os.path.join(ST, 'tune_X.npy'))).to(dev)
    tune_Y = torch.from_numpy(np.load(os.path.join(ST, 'tune_Y.npy'))).to(dev)
    tp = np.load(os.path.join(ST, 'tune_pos.npz'))
    tune_pos = [tp[f's{b}'] for b in range(len(tp.files))]
    test_X = np.load(os.path.join(ST, 'test_X.npy'), mmap_mode='r')
    test_Y = np.load(os.path.join(ST, 'test_Y.npy'))
    offsets = np.load(os.path.join(ST, 'test_offsets.npy'))
    print(f'loaded stats {time.time() - t0:.0f}s', flush=True)
    for fid, raw in todo:
        t1 = time.time()
        cols = G.delayed(raw)
        rr, ab = fit_subset(cols, XtX, XtY, tune_X, tune_Y, tune_pos,
                            test_X, test_Y, offsets, dev)
        out = os.path.join(FITS, safe_name(fid) + '.npz')
        np.savez(out + '.tmp.npz', fit_id=fid, raw_cols=raw, r_chunk_elec=rr, alphas_best=ab)
        os.replace(out + '.tmp.npz', out)
        m = rr.mean(0)
        print(f'{fid[:90]:90s} raw={len(raw):5d} all166={m.mean():.4f} '
              f'grid={m[:125].mean():.4f} {time.time() - t1:.0f}s', flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
