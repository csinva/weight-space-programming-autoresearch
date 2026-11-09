"""Hashed multi-scale subword + function-word-tag token stream, pooled with 8 exponential recency
decays into eight 64-dim blocks (512 dims), plus current-word identity, layer-normalized with a raw
position value. No training; all random vectors are fixed by seed."""
import hashlib
import numpy as np

MODEL_NAME = 'rec-msub-fwtags-v2-shared64'
MODEL_DESCRIPTION = ('Token stream of tagged chars, function-word tags, padded char 3/2/4/5-grams, word ids; '
                     'MD5-hashed to 4096 Gaussian buckets; 64-d vectors shared across 8 exp-recency pooled blocks + current word id; '
                     'layernorm with raw position value.')

N_BUCKETS = 4096
DIM = 512
N_BLOCKS = 8
BLOCK = DIM // N_BLOCKS
RATES = [0.0, 0.15, 0.3, 0.6, 1.0, 1.7, 3.0, 6.0]
MAX_TOK = 512
NGRAM_SPECS = [(3, 14), (2, 10), (4, 10), (5, 8)]

FUNCTION_WORDS = set("""
the a an this that these those my your his her its our their him me us them i you he she it we they
myself himself herself itself themselves who whom whose which what
of in on at by for with from to into onto upon about over under after before between through during
without within against among around across behind beyond toward towards near off out up down
and or but nor so yet if because although though while when where whether than as then
is am are was were be been being have has had do does did will would shall should can could may might must
not no there here all some any each every both either neither such very just also only
""".split())


def _bucket(tok):
    return int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16) % N_BUCKETS


class Embedder:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.table = rng.standard_normal((N_BUCKETS, BLOCK)).astype(np.float32)
        self.cache = {}

    def _id(self, tok):
        b = self.cache.get(tok)
        if b is None:
            b = self.cache[tok] = _bucket(tok)
        return b

    def tokens(self, text):
        words = text.lower().split()
        if not words:
            return [], None
        toks = []
        joined = ' '.join(words)
        last_start = len(joined) - len(words[-1])
        # characters, tagged as inside the last word or earlier
        for i, ch in enumerate(joined):
            toks.append(('cL:' if i >= last_start else 'cE:') + ch)
        # function-word tags, oldest first
        for w in words:
            if w in FUNCTION_WORDS:
                toks.append('fw:' + w)
        # boundary-padded char n-grams per word, oldest to newest
        for w in words:
            p = '<' + w + '>'
            for n, cap in NGRAM_SPECS:
                grams = [p[j:j + n] for j in range(len(p) - n + 1)][:cap]
                toks.extend('g%d:%s' % (n, g) for g in grams)
        # whole-word identities, current word last
        for w in words:
            toks.append('w:' + w)
        toks = toks[:MAX_TOK]
        return [self._id(t) for t in toks], self._id('w:' + words[-1])

    def __call__(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        rates = np.array(RATES)[:, None]
        for i, text in enumerate(texts):
            ids, cur = self.tokens(text)
            if not ids:
                continue
            E = self.table[ids]  # (L, DIM)
            L = len(ids)
            dist = (L - 1 - np.arange(L))[None, :]
            W = np.exp(-rates * dist)
            W /= W.sum(1, keepdims=True)  # (8, L)
            # each 64-dim block pools the same shared 64-dim vectors at its own decay rate
            v = (W @ E.astype(np.float64)) + self.table[cur][None, :]  # (8, 64)
            v = v.ravel()
            x = np.concatenate([v, [float(L)]])  # raw position value inside the layer norm
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
            out[i] = x[:DIM]
        return out


def build_embedder():
    return Embedder()
