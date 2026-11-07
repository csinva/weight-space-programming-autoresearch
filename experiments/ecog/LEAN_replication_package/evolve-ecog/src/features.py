"""Build encoding features from per-frame audio embeddings (patient1 ECoG).

Every feature is already at 30 Hz and time-aligned, so unlike the fMRI project
there is NO Lanczos / TR downsampling. For each chunk the embedder maps the
mel-spectrogram (T, 50) to a per-frame embedding (T, H). We then:

  1. FIR-delay the embedding (a few 30 Hz taps of audio history),
  2. apply the neural lag (response lags stimulus by ~100 ms = 3 frames),
  3. trim the first 30 frames (1.0 s) to match the project preprocessing,
  4. z-score features and responses per chunk.

The fixed pipeline lives here — do NOT edit it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# Project conventions (see CLAUDE.md):
LAG_FRAMES = 3    # neural lag: 100 ms at 30 Hz
TRIM_FRAMES = 30  # preprocessing trim: 1.0 s from the start


def _zscore(v: np.ndarray) -> np.ndarray:
    s = v.std(0)
    s[s == 0] = 1.0
    return (v - v.mean(0)) / s


def make_delayed(stim: np.ndarray, ndelays: int = 4) -> np.ndarray:
    """Concatenate FIR-delayed copies of `stim` for delays 0..ndelays-1 frames.

    Delay 0 (the current frame) is included so the ridge always sees the
    instantaneous audio embedding plus a short history. With ndelays=1 this is a
    no-op passthrough.
    """
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
           lag: int = LAG_FRAMES, trim: int = TRIM_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Pair feat[t-lag] with resp[t] (response lags stimulus), then trim the start.

    The trim also discards the FIR-delay warm-up region (trim=30 >> typical
    ndelays), so no delayed-zero rows leak into the fit.
    """
    if lag > 0:
        feat = feat[:-lag]
        resp = resp[lag:]
    return feat[trim:], resp[trim:]


def get_features_per_chunk(stim_dict, resp_dict, chunks: list[str],
                           embedder: Callable[[np.ndarray], np.ndarray],
                           ndelays: int = 4, lag: int = LAG_FRAMES,
                           trim: int = TRIM_FRAMES):
    """Return (feat_list, resp_list): aligned, trimmed, z-scored per chunk.

    `embedder(melspec (T,50)) -> (T, H)` is the only thing that varies between
    attempts. Each list entry is one chunk so the per-chunk per-electrode metric
    can be computed downstream.
    """
    feat_list, resp_list = [], []
    for c in chunks:
        emb = np.asarray(embedder(stim_dict[c]), dtype=np.float32)  # (T, H)
        emb = make_delayed(emb, ndelays=ndelays)
        feat, resp = _align(emb, resp_dict[c], lag=lag, trim=trim)
        if feat.shape[0] < 2:
            continue  # skip chunks too short to z-score / correlate
        feat_list.append(_zscore(feat).astype(np.float32))
        resp_list.append(_zscore(resp).astype(np.float32))
    return feat_list, resp_list
