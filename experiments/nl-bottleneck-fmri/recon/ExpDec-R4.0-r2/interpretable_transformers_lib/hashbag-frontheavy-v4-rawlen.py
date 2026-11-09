"""Hand-written text features for predicting fMRI responses (no training).

Rebuild of an untrained one-layer hashed-token transformer: current-word hashed random
embedding plus a down-weighted, front-heavy (~1/position) hashed bag of the 10-gram tokens,
plus three length-tracking dims, all passed through a final LayerNorm.
"""
import hashlib
import re
import numpy as np

MODEL_NAME = 'hashbag-frontheavy-v4-rawlen'
MODEL_DESCRIPTION = ('Current-word MD5-hashed Gaussian 1024-d vector (4096 buckets) + 0.5x pooled (uniform attn, tokens shrunk 1/pos, unnormalized so length-dependent) context: '
                     'chars (last-word vs earlier), len bucket, first+last letter, skip-bigram, stem, char 2-5grams, word; '
                     '8 heads of 128 dims with ~1/position weights and slight recency tilt; 3 raw length dims (words, chars, tokens) so final LayerNorm scale is length-dependent.')

N_BUCKETS = 4096
DIM = 1024
N_HEADS = 8
HEAD_DIM = DIM // N_HEADS
CTX_WEIGHT = 0.5
CTX_SCALE = 300.0 / np.log(300.0)  # fixed constant: ~unit weight-sum at a typical 300-token 10-gram
SUFFIXES = ['ingly', 'ings', 'ing', 'edly', 'ness', 'ment', 'tion', 'ions', 'ion', 'ies', 'ied',
            'ers', 'er', 'est', 'ed', 'ly', 'es', 's']


def _bucket(tok):
    return int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16) % N_BUCKETS


def _stem(w):
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w


def _len_bucket(n):
    return min(n, 12) if n < 8 else (8 if n < 10 else 9 if n < 13 else 10)


def _tokens(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        words = ['<empty>']
    toks = []
    earlier = ' '.join(words[:-1])
    for c in earlier:
        toks.append('ch:' + c)
    for c in words[-1]:
        toks.append('lch:' + c)
    for i, w in enumerate(words):
        toks.append('len:%d' % _len_bucket(len(w)))
        toks.append('fl:%s%s' % (w[0], w[-1]))
        if i >= 2:
            toks.append('skip:%s_%s' % (words[i - 2], w))
        toks.append('stem:' + _stem(w))
        p = '<' + w + '>'
        for n in range(2, 6):
            for j in range(len(p) - n + 1):
                toks.append('cg%d:%s' % (n, p[j:j + n]))
        toks.append('w:' + w)
    return words, toks


class Embedder:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.table = rng.standard_normal((N_BUCKETS, DIM)).astype(np.float32)
        # per-head recency tilt (small), heads differ slightly
        self.tilts = np.linspace(0.0, 0.3, N_HEADS)

    def __call__(self, texts):
        out = np.zeros((len(texts), DIM + 3), dtype=np.float32)
        for i, text in enumerate(texts):
            words, toks = _tokens(text)
            ids = np.array([_bucket(t) for t in toks])
            V = self.table[ids]  # (T, DIM)
            T = len(toks)
            pos = np.arange(1, T + 1, dtype=np.float64)
            rel = pos / T
            pooled = np.zeros(DIM, dtype=np.float64)
            for h in range(N_HEADS):
                a = np.exp(self.tilts[h] * rel)
                a /= a.sum()  # near-uniform attention, slight recency tilt
                w = a / pos * CTX_SCALE  # LN with large position signal shrinks token ~1/position
                sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                pooled[sl] = w @ V[:, sl]
            ident = V[-1].astype(np.float64)  # current-word identity
            x = np.concatenate([ident + CTX_WEIGHT * pooled,
                                [float(len(words)), float(len(text)), float(T)]])
            x = (x - x.mean()) / (x.std() + 1e-5)
            out[i] = x
        return out


def build_embedder():
    return Embedder()
