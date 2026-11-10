"""Streamed re-fit of a harness encoder for the multi-subject rerun (from runs-wave/newtest).

Same math as encoding.bootstrap_ridge_gpu + eval._per_chunk_elec_corr: training features are
accumulated chunk by chunk into XtX/XtY (float64) with the bootstrap tune rows kept; per-electrode
alphas are chosen on the same 5 temporal-block splits; final weights come from the full XtX. The
test chunks are scored only after the fit, one at a time (held-out sessions have up to ~400 chunks,
too many to hold), and every electrode is kept so any subset (e.g. the grid) is scored afterwards.
With MS_SPLIT set, the training list must equal (a prefix of) the subject's frozen list and the
test list must equal its frozen test list.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from . import data, encoding, features

ALPHAS = np.logspace(1, 4, 12)
SINGCUT = 1e-10


def _aligned_len(chunk, lag, trim):
    seg_len = np.load(os.path.join(data.WAVE_DIR, f"{chunk}_wave.npy"), mmap_mode="r").shape[0]
    t_resp = np.load(os.path.join(data.WAV_DIR, f"{chunk}_ecog_hasson_highgamma.npy"),
                     mmap_mode="r").shape[1]
    n = max(0, (min(data.seg_to_T(seg_len), t_resp) - lag) - trim)
    return n if n >= 2 else 0


def _one(chunk, emb, cfg):
    f, r = features.get_features_per_chunk(data.load_stim([chunk]), data.load_responses([chunk]),
                                           [chunk], emb, ndelays=cfg.ndelays, lag=cfg.lag,
                                           trim=cfg.trim)
    return (f[0], r[0]) if f else (None, None)


def _corr(X, Y, wt, dev):
    pred = torch.from_numpy(X).to(dev).double() @ wt
    r = torch.from_numpy(Y).to(dev).double()
    a = r - r.mean(0); b = pred - pred.mean(0)
    den = torch.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    val = torch.where(den == 0, torch.zeros_like(den),
                      (a * b).sum(0) / torch.where(den == 0, torch.ones_like(den), den))
    return torch.nan_to_num(val).cpu().numpy()


def rescore(emb, cfg, out):
    t0 = time.time()
    dev = "cuda"
    train, test = data.get_chunk_names(cfg.num_train, cfg.num_test)
    if os.environ.get("MS_SPLIT"):
        frozen = json.load(open(os.environ["MS_SPLIT"]))
        assert train == frozen["train"][:len(train)], "training list is not (a prefix of) the frozen list"
        assert test == frozen["test"], "test list changed"
    lens = [_aligned_len(c, cfg.lag, cfg.trim) for c in train]
    nresp = int(sum(lens))
    splits = encoding._temporal_chunk_splits(cfg.nboots, nresp, cfg.chunklen, cfg.nchunks)
    union = np.unique(np.concatenate([np.asarray(tu, dtype=np.int64) for _, tu in splits]))
    pos_of = {int(g): i for i, g in enumerate(union)}
    XtX = XtY = tune_X = tune_Y = None
    off = 0
    for i, c in enumerate(train):
        X, Y = _one(c, emb, cfg)
        n = 0 if X is None else X.shape[0]
        assert n == lens[i], f"{c}: {n} rows != predicted {lens[i]}"
        if n == 0:
            continue
        if XtX is None:
            P, V = X.shape[1], Y.shape[1]
            XtX = torch.zeros(P, P, dtype=torch.float64, device=dev)
            XtY = torch.zeros(P, V, dtype=torch.float64, device=dev)
            tune_X = np.zeros((len(union), P), np.float32)
            tune_Y = np.zeros((len(union), V), np.float32)
        Xg = torch.from_numpy(X).to(dev, torch.float64)
        XtX += Xg.T @ Xg
        XtY += Xg.T @ torch.from_numpy(Y).to(dev, torch.float64)
        del Xg
        lo, hi = np.searchsorted(union, [off, off + n])
        for g in union[lo:hi]:
            tune_X[pos_of[int(g)]] = X[g - off]
            tune_Y[pos_of[int(g)]] = Y[g - off]
        off += n
    assert off == nresp
    t_feat = time.time() - t0

    nvox = XtY.shape[1]
    a2 = torch.tensor(ALPHAS ** 2, dtype=torch.float64, device=dev)
    tX = torch.from_numpy(tune_X).to(dev)
    tY = torch.from_numpy(tune_Y).to(dev)
    all_corrs = torch.zeros(len(ALPHAS), nvox, dtype=torch.float64, device=dev)
    for _, tu in splits:
        p = torch.as_tensor([pos_of[int(g)] for g in tu], device=dev)
        Xtu = tX.index_select(0, p).double()
        Ytu = tY.index_select(0, p).double()
        sub = Xtu.T @ Xtu
        sub.neg_().add_(XtX)
        Vv, Lam = encoding._eig_good(sub, SINGCUT)
        del sub
        A = Xtu @ Vv
        B = Vv.T @ (XtY - Xtu.T @ Ytu)
        zY = encoding._zs_t(Ytu)
        for ai in range(len(ALPHAS)):
            pred = A @ (B / (Lam + a2[ai]).unsqueeze(1))
            all_corrs[ai] += torch.nan_to_num((zY * encoding._zs_t(pred)).mean(0))
        del Xtu, Ytu, Vv, Lam, A, B, zY
    best = torch.argmax(all_corrs, dim=0).cpu().numpy()
    alphas_best = ALPHAS[best]
    Vv, Lam = encoding._eig_good(XtX, SINGCUT)
    B = Vv.T @ XtY
    wt = torch.zeros(XtX.shape[0], nvox, dtype=torch.float64, device=dev)
    for ua in np.unique(alphas_best):
        sel = torch.as_tensor(np.nonzero(alphas_best == ua)[0], device=dev)
        wt[:, sel] = Vv @ (B[:, sel] / (Lam + float(ua) ** 2).unsqueeze(1))
    P = XtX.shape[0]
    del Vv, Lam, B, XtX, tX, tY
    torch.cuda.empty_cache()

    t1 = time.time()
    rr, kept = [], []
    for c in test:
        X, Y = _one(c, emb, cfg)
        if X is not None:
            rr.append(_corr(X, Y, wt, dev)); kept.append(c)
    rr = np.stack(rr)
    np.savez(out, r_chunk_elec=rr, test_chunks=np.array(kept), alphas_best=alphas_best, P=P, nresp=nresp,
             seconds_features=t_feat, seconds_test=time.time() - t1, seconds_total=time.time() - t0)
    m = rr.mean(0)
    ng = int(json.load(open(os.environ["MS_SPLIT"]))["n_grid"]) if os.environ.get("MS_SPLIT") else 125
    print(f"RESCORE P={P} n_test={len(kept)} all={m.mean():.5f} grid={m[:ng].mean():.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)
