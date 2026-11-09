"""Hand-written text features for predicting fMRI responses.

Rebuild of an untrained one-layer 'transformer' over hashed tokens:
current-word hashed random embedding + down-weighted, front-heavy hashed bag of the 10-gram
(8 heads of 128 dims, ~1/position shrinkage, slight recency tilt) + 3 length dims, final LayerNorm.
"""
import hashlib
import re
import numpy as np

MODEL_NAME = 'hashbag-frontheavy-v1'
MODEL_DESCRIPTION = ('Current-word MD5-hashed Gaussian 1024-d embedding plus down-weighted pooled bag of hashed '
                     'tokens (chars, len bucket, first+last letter, skip-bigram, stem, char 2-5grams, word) '
                     'weighted ~1/position with slight per-head recency tilt; 3 length dims; final LayerNorm.')

D = 1024
NB = 4096
NH = 8
HD = D // NH
SEED = 0
CTX_GAIN = 0.35   # pooled context smaller than identity
POS0 = 1.0        # 1/(position + POS0)-type shrinkage

_SUFFIXES = ['ingly', 'ations', 'ation', 'ments', 'ment', 'ness', 'ings', 'ing', 'edly', 'ies',
             'ied', 'ers', 'est', 'ly', 'ed', 'er', 'es', 's']


def _stem(w):
    for s in _SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w


def _h(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % NB


def _len_bucket(n):
    return str(min(n, 12)) if n < 8 else ('8-11' if n < 12 else '12+')


def _word_tokens(words, i):
    w = words[i]
    toks = ['len:' + _len_bucket(len(w)), 'fl:' + w[0] + w[-1]]
    if i >= 2:
        toks.append('skip:' + words[i - 2] + '_' + w)
    toks.append('stem:' + _stem(w))
    p = '<' + w + '>'
    for n in range(2, 6):
        for j in range(len(p) - n + 1):
            toks.append('cg:' + p[j:j + n])
    toks.append('w:' + w)
    return toks


def _layernorm(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


class Embedder:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.table = rng.standard_normal((NB, D)).astype(np.float32)
        self.char_last = rng.standard_normal((128, D)).astype(np.float32)
        self.char_prev = rng.standard_normal((128, D)).astype(np.float32)
        # slight per-head recency tilt (head 0 ~ uniform, head 7 a bit more recent)
        self.tilt = np.linspace(0.0, 0.6, NH)
        self.cache = {}

    def _cid(self, c):
        o = ord(c)
        return o if o < 128 else 127

    def _one(self, text):
        t = text.lower()
        words = re.findall(r"[a-z0-9']+", t)
        if not words:
            words = ['<unk>']
        # build token vectors in order
        vecs = []
        last_start = t.rstrip().rfind(' ') + 1
        for k, c in enumerate(t):
            tab = self.char_last if k >= last_start else self.char_prev
            vecs.append(tab[self._cid(c)])
        for i in range(len(words)):
            for tok in _word_tokens(words, i):
                vecs.append(self.table[_h(tok)])
        V = np.stack(vecs)                       # (N, D)
        N = V.shape[0]
        pos = np.arange(N, dtype=np.float32)
        shrink = 1.0 / (pos + POS0)              # LayerNorm with large positional signal ~ 1/position
        rel = pos / max(N - 1, 1)
        pooled = np.empty(D, dtype=np.float32)
        for h in range(NH):
            w = np.exp(self.tilt[h] * rel)
            w = w / w.sum()                      # nearly uniform averaging
            pooled[h * HD:(h + 1) * HD] = (w * shrink) @ V[:, h * HD:(h + 1) * HD]
        # rescale pooled so it is a smaller (but length-dependent) companion of the identity
        pooled = pooled * CTX_GAIN * np.sqrt(N)
        ident = self.table[_h('w:' + words[-1])]
        lens = np.array([np.log1p(N), len(words) / 10.0, len(t) / 60.0], dtype=np.float32)
        out = np.concatenate([ident + pooled, lens])
        return _layernorm(out).astype(np.float32)

    def __call__(self, texts):
        return np.stack([self._one(x) for x in texts])


def build_embedder():
    return Embedder()
