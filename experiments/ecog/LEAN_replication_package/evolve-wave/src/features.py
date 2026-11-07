"""Build encoding features for the waveform track (patient1 ECoG).

Per chunk: the aligned waveform segment is framed by the FIXED `data.frame_windows`
helper into (T, win) audio windows on the 30 Hz grid, the embedder (front-end) maps
those to a per-frame embedding (T, H), and then we apply the SAME post-processing as
the mel track: FIR delays, the 100 ms neural lag (3 frames), the 30-frame (1.0 s)
trim, and per-chunk z-scoring.

The alignment crop (1 s off each end) is already baked into the pre-extracted
segments; the lag + extra 30-frame trim below match the mel-track pipeline exactly,
so wave features are paired with high-gamma identically.

The fixed pipeline lives here — do NOT edit it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import data

LAG_FRAMES = 3    # neural lag: 100 ms at 30 Hz
TRIM_FRAMES = 30  # extra trim: 1.0 s from the start (on top of the extraction crop)


def _zscore(v: np.ndarray) -> np.ndarray:
    s = v.std(0)
    s[s == 0] = 1.0
    return (v - v.mean(0)) / s


def make_delayed(stim: np.ndarray, ndelays: int = 4) -> np.ndarray:
    """Concatenate FIR-delayed copies of `stim` for delays 0..ndelays-1 frames."""
    n, d = stim.shape
    out = []
    for delay in range(ndelays):
        dstim = np.zeros((n, d), dtype=stim.dtype)
        if delay == 0:
            dstim[:] = stim
        else:
            dstim[delay:] = stim[:-delay]
        out.append(dstim)
    return np.hstack(out)


def _align(feat: np.ndarray, resp: np.ndarray,
           lag: int = LAG_FRAMES, trim: int = TRIM_FRAMES):
    if lag > 0:
        feat = feat[:-lag]
        resp = resp[lag:]
    return feat[trim:], resp[trim:]


def get_features_per_chunk(stim_dict, resp_dict, chunks: list[str],
                           embedder: Callable[[np.ndarray], np.ndarray],
                           ndelays: int = 4, lag: int = LAG_FRAMES,
                           trim: int = TRIM_FRAMES):
    """Return (feat_list, resp_list): aligned, trimmed, z-scored per chunk.

    `embedder(windows (T, win)) -> (T, H)` is the only thing that varies between
    attempts. The embedder declares its analysis window via an optional `.win`
    attribute (default 2048); the fixed framer centers every window on the same
    30 Hz grid, so alignment holds for any front-end.
    """
    win = int(getattr(embedder, "win", 2048))
    feat_list, resp_list = [], []
    for c in chunks:
        seg = stim_dict[c]
        resp = resp_dict[c]
        T = min(data.seg_to_T(len(seg)), resp.shape[0])
        windows = data.frame_windows(seg, win=win, n_frames=T)   # (T, win)
        emb = np.asarray(embedder(windows), dtype=np.float32)     # (T, H)
        emb = make_delayed(emb, ndelays=ndelays)
        feat, r = _align(emb, resp[:T], lag=lag, trim=trim)
        if feat.shape[0] < 2:
            continue
        feat_list.append(_zscore(feat).astype(np.float32))
        resp_list.append(_zscore(r).astype(np.float32))
    return feat_list, resp_list
