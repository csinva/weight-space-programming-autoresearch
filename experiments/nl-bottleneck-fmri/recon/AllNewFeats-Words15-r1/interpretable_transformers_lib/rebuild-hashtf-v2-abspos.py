"""Hand-written text features for predicting fMRI responses.

One hand-set "transformer layer": the 10-gram becomes a token sequence (lowercased characters,
last-word characters using a separate embedding table, then md5-hashed word tags oldest->newest,
then the current word's unigram hash). Each token gets a fixed Gaussian 1024-d vector. Eight heads
(128 dims each) softmax-pool the vectors with position-based decay rates 0..6; because of the
pre-attention LayerNorm the effective weighting is primacy (earliest tokens favoured). Readout adds
the current word's hash vector, applies LayerNorm, and appends n and 6n^2/512 (n = sequence length).
"""
import hashlib
import re
import numpy as np

MODEL_NAME = 'rebuild-hashtf-v2-abspos'
MODEL_DESCRIPTION = ('One hand-set attention layer over char tokens (last word separate table) + md5-hashed '
                     'word tags (4096 buckets); 8 heads x128 dims primacy softmax pooling, decay rates 0..6 per token index; '
                     'readout + current-word hash vector, LayerNorm, plus n and 6n^2/512.')

D = 1024
N_HEADS = 8
HD = D // N_HEADS
N_BUCKETS = 4096
RATES = np.linspace(0.0, 6.0, N_HEADS)

FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their me him us them i you he she it we they
and or but nor so yet if then because although though while when where whether as than
of in on at by for with about against between into through during before after above below to from
up down out off over under again further once here there all any both each few more most other some
such no not only own same too very can will just should would could may might must shall do does did
is am are was were be been being have has had having what which who whom whose why how
""".split())

SUFFIXES = ['ational', 'ization', 'fulness', 'ousness', 'iveness', 'ement', 'ments', 'ness', 'ment',
            'tion', 'sion', 'ing', 'ers', 'ies', 'ied', 'est', 'ful', 'ous', 'ive', 'ize', 'ise', 'able',
            'ible', 'ally', 'ly', 'ed', 'er', 'es', 's']


def stem(w):
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w


def bucket(tag):
    return int(hashlib.md5(tag.encode('utf-8')).hexdigest(), 16) % N_BUCKETS


def word_tags(words, j):
    w = words[j]
    tags = []
    if w in FUNCTION_WORDS:
        tags.append('fw:' + w)
    tags.append('len:%d' % min(len(w), 15))
    if w:
        tags.append('first:' + w[0])
        tags.append('last:' + w[-1])
    if j >= 2:
        tags.append('skip:%s_%s' % (words[j - 2], w))
    tags.append('stem:' + stem(w))
    b = '<' + w + '>'
    for n in range(2, 6):
        for k in range(len(b) - n + 1):
            tags.append('cg%d:%s' % (n, b[k:k + n]))
    return tags


def layernorm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class Embedder:
    def __init__(self):
        rng = np.random.default_rng(0)
        s = 1.0 / np.sqrt(D)
        self.char_ctx = rng.standard_normal((256, D)).astype(np.float32) * s
        self.char_last = rng.standard_normal((256, D)).astype(np.float32) * s
        self.hash_emb = rng.standard_normal((N_BUCKETS, D)).astype(np.float32) * s

    def tokens(self, text):
        text = text.lower()
        words = [re.sub(r'\s+', '', w) for w in text.split()]
        words = [w for w in words if w]
        vecs = []
        if not words:
            return np.zeros((1, D), np.float32), None
        ctx = ' '.join(words[:-1])
        for c in ctx:
            vecs.append(self.char_ctx[ord(c) % 256])
        if ctx:
            vecs.append(self.char_ctx[ord(' ')])
        for c in words[-1]:
            vecs.append(self.char_last[ord(c) % 256])
        for j in range(len(words)):
            for t in word_tags(words, j):
                vecs.append(self.hash_emb[bucket(t)])
        cur = self.hash_emb[bucket('uni:' + words[-1])]
        vecs.append(cur)
        return np.stack(vecs), cur

    def embed_one(self, text):
        X, cur = self.tokens(text)
        n = X.shape[0]
        Xn = layernorm(X)
        pos = np.arange(n).astype(np.float64)  # token index from the earliest token
        out = np.zeros(D, np.float32)
        for h in range(N_HEADS):
            sc = -RATES[h] * pos  # primacy: earliest tokens get the largest weight
            w = np.exp(sc - sc.max()); w /= w.sum()
            out[h * HD:(h + 1) * HD] = w @ Xn[:, h * HD:(h + 1) * HD]
        if cur is not None:
            out = out + cur * np.sqrt(D)  # current word hash vector (scaled to LN units)
        out = layernorm(out)
        return np.concatenate([out, [n, 6.0 * n * n / 512.0]]).astype(np.float32)

    def __call__(self, texts):
        return np.stack([self.embed_one(t) for t in texts])


def build_embedder():
    return Embedder()
