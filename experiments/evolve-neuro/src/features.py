"""Build features for fMRI encoding from per-ngram transformer embeddings.

For every word we build a 10-gram (the word plus the preceding words) and embed
it with the model in `interpretable_transformer.py`. Per-word vectors are then
Lanczos-downsampled onto the fMRI TR timeline, trimmed/z-scored, and expanded
with FIR delays.

Simplified from `neuro/features/{feature_spaces,feature_utils}.py` and
`neuro/data/interp_data.py`.
"""
from typing import Dict, List

import numpy as np


# ----------------------------- ngrams -----------------------------
def get_ngrams(words: List[str], ngram_size: int = 10) -> List[str]:
    """Each word -> a string of the (ngram_size) words leading up to and incl it."""
    ngrams = []
    for i in range(len(words)):
        lo = max(0, i - ngram_size)
        ngrams.append(' '.join(words[lo: i + 1]).strip())
    return ngrams


# ----------------------------- downsampling -----------------------------
def _lanczosfun(cutoff, t, window=3):
    t = t * cutoff
    val = window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi ** 2 * t ** 2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val


def lanczos_downsample(data, oldtime, newtime, window=3):
    """Interpolate rows of `data` from `oldtime` onto `newtime` (Lanczos filter)."""
    cutoff = 1 / np.mean(np.diff(newtime))
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for i in range(len(newtime)):
        sincmat[i, :] = _lanczosfun(cutoff, newtime[i] - oldtime, window)
    return np.dot(sincmat, data)


# ----------------------------- normalization / delays -----------------------------
def _zscore(v):
    s = v.std(0)
    s[s == 0] = 1.0
    return (v - v.mean(0)) / s


def trim_and_zscore(downsampled: Dict[str, np.ndarray], trim=5, extra_trim=0):
    """Trim [5+trim+extra_trim : -trim-extra_trim] per story, z-score, then stack.

    `extra_trim` drops additional TRs from the start and end of every story
    (applied identically to responses). Matches response trim.
    """
    lo = 5 + trim + extra_trim
    hi = trim + extra_trim
    feats = [_zscore(downsampled[s][lo: -hi]) for s in downsampled]
    return np.vstack(feats)


def make_delayed(stim, ndelays=4):
    """Concatenate FIR-delayed copies of `stim` (delays 1..ndelays TRs)."""
    n, d = stim.shape
    out = []
    for delay in range(1, ndelays + 1):
        dstim = np.zeros((n, d))
        dstim[delay:, :] = stim[:-delay, :]
        out.append(dstim)
    return np.hstack(out)


# ----------------------------- top-level -----------------------------
def get_features(wordseqs, stories, embedder, ngram_size=10, ndelays=4, extra_trim=0):
    """Return delayed feature matrix (sum_of_trimmed_trs, ndelays * hidden_dim).

    `extra_trim` drops additional TRs from each story's start and end (must
    match the same trim applied to responses).
    """
    downsampled = {}
    for story in stories:
        ws = wordseqs[story]
        ngrams = get_ngrams(list(ws.data), ngram_size=ngram_size)
        word_vectors = embedder(ngrams)
        downsampled[story] = lanczos_downsample(
            word_vectors, oldtime=ws.data_times, newtime=ws.tr_times
        )
    feats = trim_and_zscore(downsampled, extra_trim=extra_trim)
    return make_delayed(feats, ndelays=ndelays)
