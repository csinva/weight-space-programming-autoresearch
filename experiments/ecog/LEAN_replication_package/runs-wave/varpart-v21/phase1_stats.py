"""Phase 1: extract LEAN-v21 features once and save the sufficient
statistics every variance-partitioning fit needs.

Same chunks and pipeline as runs-wave/lean-whisper/diag_v20.py (same
features.get_features_per_chunk), LEAN-v21 encoder, but streams chunk by chunk
instead of materializing the 1.75M x 15408 training matrix:

  stats/XtX.npy     (P, P)  float64  sum over train chunks of X_c^T X_c
  stats/XtY.npy     (P, 166) float64
  stats/tune_X.npy  rows of the training matrix used by the 5 bootstrap tune
  stats/tune_Y.npy  splits of encoding.bootstrap_ridge_gpu (same seed/indices)
  stats/tune_pos.npz   per-split positions into tune_X
  stats/test_X.npy  (N_test_rows, P) float32, test_Y, test_offsets (23 chunks)

Any column subset S of the ridge fit is then XtX[S,S], XtY[S], tune_X[:, S],
test_X[:, S] -- identical math to fitting that subset from scratch.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data, features, encoding  # noqa: E402
import lean_v21_enc as L  # noqa: E402
import groups as G  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'stats')
NBOOTS, CHUNKLEN, NCHUNKS = 5, 40, 20  # encoding.bootstrap_ridge defaults


def acquire_device(retries=8, wait=5, max_requeue=15):
    """Self-requeue on a100 shared-node GPU collisions (see train_conformer_transfer.py)."""
    last = None
    for k in range(retries):
        try:
            t = torch.zeros(1, device='cuda'); torch.cuda.synchronize(); del t
            return 'cuda'
        except Exception as e:  # noqa: BLE001
            last = e
            print(f'[gpu] busy/unavailable, attempt {k + 1}/{retries}: {e}', flush=True)
            time.sleep(wait)
    jid = os.environ.get('SLURM_JOB_ID')
    n_re = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    if jid and n_re < max_requeue:
        print(f'[gpu] requeuing {jid} (restart {n_re}->{n_re + 1})', flush=True)
        os.system(f'scontrol requeue {jid}')
        time.sleep(20)
    raise RuntimeError(f'could not acquire CUDA: {last}')


def aligned_len(chunk: str) -> int:
    """Rows this chunk contributes after features._align (0 if dropped)."""
    seg_len = np.load(os.path.join(data.WAVE_DIR, f'{chunk}_wave.npy'), mmap_mode='r').shape[0]
    t_resp = np.load(os.path.join(data.WAV_DIR, f'{chunk}_ecog_hasson_highgamma.npy'),
                     mmap_mode='r').shape[1]
    T = min(data.seg_to_T(seg_len), t_resp)
    n = max(0, (T - features.LAG_FRAMES) - features.TRIM_FRAMES)
    return n if n >= 2 else 0


def one_chunk(chunk, emb):
    stim = data.load_stim([chunk]); resp = data.load_responses([chunk])
    f, r = features.get_features_per_chunk(stim, resp, [chunk], emb)
    return (f[0], r[0]) if f else (None, None)


def main():
    t0 = time.time()
    dev = acquire_device()
    os.makedirs(OUT, exist_ok=True)
    train, test = data.get_chunk_names(500, 100)
    print(f'train={len(train)} test={len(test)}', flush=True)

    emb = L.build_embedder(device=dev)  # v21 default = scored 'v3e' config
    recorded = []
    orig_concat = emb._concat_groups

    def _rec(groups):
        if not recorded:
            recorded.extend((n, int(t.shape[1])) for n, t in groups)
        return orig_concat(groups)
    emb._concat_groups = _rec

    # Bootstrap tune rows: global row indices into vstack(train features), same as harness.
    lens = [aligned_len(c) for c in train]
    nresp = int(sum(lens))
    splits = encoding._temporal_chunk_splits(NBOOTS, nresp, CHUNKLEN, NCHUNKS)
    tune_union = np.unique(np.concatenate([np.asarray(tu, dtype=np.int64) for _, tu in splits]))
    pos_of = {int(g): i for i, g in enumerate(tune_union)}
    print(f'nresp={nresp} tune_union={len(tune_union)}', flush=True)

    P = G.RAW_DIM * G.NDELAYS
    XtX = torch.zeros(P, P, dtype=torch.float64, device=dev)
    XtY = torch.zeros(P, data.N_ELECTRODES, dtype=torch.float64, device=dev)
    tune_X = np.zeros((len(tune_union), P), dtype=np.float32)
    tune_Y = np.zeros((len(tune_union), data.N_ELECTRODES), dtype=np.float32)
    off = 0
    verified = False
    for i, c in enumerate(train):
        X, Y = one_chunk(c, emb)
        if not verified and X is not None:
            exp = [(g, d) for g, d, _ in G.GROUPS if g != 'scattering']
            assert recorded == exp, f'group layout mismatch:\n got {recorded}\n exp {exp}'
            assert X.shape[1] == P, (X.shape, P)
            verified = True
            print('group layout verified against embedder output', flush=True)
        n = 0 if X is None else X.shape[0]
        assert n == lens[i], f'{c}: rows {n} != predicted {lens[i]}'
        if n == 0:
            continue
        Xg = torch.from_numpy(X).to(dev, torch.float64)
        Yg = torch.from_numpy(Y).to(dev, torch.float64)
        XtX += Xg.T @ Xg
        XtY += Xg.T @ Yg
        del Xg, Yg
        lo, hi = np.searchsorted(tune_union, [off, off + n])
        for g in tune_union[lo:hi]:
            tune_X[pos_of[int(g)]] = X[g - off]
            tune_Y[pos_of[int(g)]] = Y[g - off]
        off += n
        if i % 25 == 0:
            print(f'  train {i + 1}/{len(train)} rows={off} {time.time() - t0:.0f}s', flush=True)
    assert off == nresp

    test_X, test_Y, offsets = [], [], [0]
    for c in test:
        X, Y = one_chunk(c, emb)
        if X is None:
            continue
        test_X.append(X); test_Y.append(Y); offsets.append(offsets[-1] + X.shape[0])
    print(f'test chunks kept={len(test_X)} rows={offsets[-1]}', flush=True)

    np.save(os.path.join(OUT, 'XtX.npy'), XtX.cpu().numpy())
    np.save(os.path.join(OUT, 'XtY.npy'), XtY.cpu().numpy())
    np.save(os.path.join(OUT, 'tune_X.npy'), tune_X)
    np.save(os.path.join(OUT, 'tune_Y.npy'), tune_Y)
    np.savez(os.path.join(OUT, 'tune_pos.npz'),
             **{f's{b}': np.array([pos_of[int(g)] for g in tu], dtype=np.int64)
                for b, (_, tu) in enumerate(splits)})
    np.save(os.path.join(OUT, 'test_X.npy'), np.vstack(test_X))
    np.save(os.path.join(OUT, 'test_Y.npy'), np.vstack(test_Y))
    np.save(os.path.join(OUT, 'test_offsets.npy'), np.asarray(offsets, dtype=np.int64))
    with open(os.path.join(OUT, 'meta.json'), 'w') as fh:
        json.dump(dict(train=train, test=test, nresp=nresp, P=P,
                       groups=recorded + [('scattering', dict((g, d) for g, d, _ in G.GROUPS)['scattering'])],
                       seconds=time.time() - t0), fh, indent=1)
    # Written last: phase 2 waits for this marker.
    open(os.path.join(OUT, 'DONE'), 'w').write(f'{time.time() - t0:.0f}s\n')
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
