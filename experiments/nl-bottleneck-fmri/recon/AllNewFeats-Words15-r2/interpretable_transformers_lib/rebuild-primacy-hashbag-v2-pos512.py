"""Hand-set one-layer transformer over character + hashed-tag tokens of each 10-gram.

Tokens: lowercased characters (last-word characters use a separate embedding table), then per
word (oldest to newest) md5-hashed tags into 4096 buckets, then the current word's unigram hash.
Each token has a fixed Gaussian vector (1024 d). Eight heads (128 d each) softmax-pool with
position-based rates 0..6 that (as in the original, because of a pre-attention LayerNorm) favour
the EARLIEST tokens. Readout = pooled + current-word hash vector, LayerNorm, plus n and 6n^2/512.
"""
import hashlib
import re
import numpy as np

MODEL_NAME = 'rebuild-primacy-hashbag-v2-pos512'
MODEL_DESCRIPTION = ('One hand-set attention layer: Gaussian-embedded chars (last word separate) + '
                     'md5 4096-bucket tags (function word, length, first/last letter, skip-bigram, '
                     'stem, char 2-5grams) + current unigram hash; 8 heads primacy-weighted pooling '
                     'rates 0..6 per position/512; readout pooled + current hash vec, LayerNorm, plus n, 6n^2/512.')

D = 1024
N_HEADS = 8
HD = D // N_HEADS
N_BUCKETS = 4096
RATES = np.linspace(0.0, 6.0, N_HEADS)

FUNCTION_WORDS = """the a an and or but nor so yet for of in on at to from by with about as into
onto upon over under after before between through during without within against among i me my
mine we us our you your he him his she her it its they them their this that these those who whom
whose which what when where why how is am are was were be been being have has had do does did
not no will would shall should can could may might must there here then than if because while
all some any each every just very too also only up down out off again more most such own same
other""".split()
FW_SET = set(FUNCTION_WORDS)

SUFFIXES = ['ingly', 'ations', 'ation', 'ments', 'ment', 'ness', 'ings', 'ing', 'edly', 'ies',
            'ied', 'ers', 'est', 'ly', 'ed', 'er', 'es', 's']


def stem(w):
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w


def hbucket(tag):
    return int(hashlib.md5(tag.encode('utf-8')).hexdigest(), 16) % N_BUCKETS


def clean(w):
    return re.sub(r"[^a-z0-9']", '', w)


# vocabulary layout: [0,256) chars of earlier words, [256,512) chars of last word, then hash buckets
CHAR_OFF, LAST_OFF, HASH_OFF = 0, 256, 512
VOCAB = HASH_OFF + N_BUCKETS


def word_tags(words, i):
    w = words[i]
    tags = []
    if w in FW_SET:
        tags.append('fw:' + w)
    tags.append('len:%d' % min(len(w), 15))
    if w:
        tags.append('first:' + w[0])
        tags.append('last:' + w[-1])
    prev2 = words[i - 2] if i >= 2 else '<s>'
    tags.append('skip:%s_%s' % (prev2, w))
    tags.append('stem:' + stem(w))
    b = '<' + w + '>'
    for n in range(2, 6):
        for k in range(len(b) - n + 1):
            tags.append('cg%d:%s' % (n, b[k:k + n]))
    return tags


class Embedder:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.E = (rng.standard_normal((VOCAB, D)) / np.sqrt(D)).astype(np.float32)
        self.cache = {}

    def tokens(self, text):
        text = text.lower()
        raw = text.split()
        ids = []
        for wi, w in enumerate(raw):
            off = LAST_OFF if wi == len(raw) - 1 else CHAR_OFF
            if wi > 0:
                ids.append(off + ord(' '))
            for c in w:
                ids.append(off + (ord(c) % 256))
        words = [clean(w) for w in raw]
        for i in range(len(words)):
            for t in word_tags(words, i):
                ids.append(HASH_OFF + hbucket(t))
        cur = words[-1] if words else ''
        uni = HASH_OFF + hbucket('uni:' + cur)
        ids.append(uni)
        return np.array(ids, dtype=np.int64), uni

    def __call__(self, texts):
        out = np.zeros((len(texts), D + 2), dtype=np.float32)
        for r, text in enumerate(texts):
            ids, uni = self.tokens(text)
            n = len(ids)
            X = self.E[ids]  # (n, D)
            pos = np.arange(n) / 512.0  # 0 = earliest; fixed 512-token position scale
            pooled = np.empty(D, dtype=np.float32)
            for h in range(N_HEADS):
                s = -RATES[h] * pos  # primacy: earliest tokens get the largest weight
                w = np.exp(s - s.max()); w /= w.sum()
                pooled[h * HD:(h + 1) * HD] = w @ X[:, h * HD:(h + 1) * HD]
            y = pooled + self.E[uni]
            y = (y - y.mean()) / np.sqrt(y.var() + 1e-5)
            out[r, :D] = y
            out[r, D] = n
            out[r, D + 1] = 6.0 * n * n / 512.0
        return out


def build_embedder():
    return Embedder()
