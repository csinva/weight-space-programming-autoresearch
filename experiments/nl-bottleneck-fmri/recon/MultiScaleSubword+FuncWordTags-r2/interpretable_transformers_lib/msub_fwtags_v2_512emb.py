"""Hand-written text features for predicting fMRI responses.

Multi-scale recency pooling over a hashed token stream (characters, function-word tags,
character n-grams, word identities). No training; all embeddings are fixed random Gaussians.
"""
import hashlib
import numpy as np
import torch

MODEL_NAME = 'msub_fwtags_v2_512emb'
MODEL_DESCRIPTION = ('512-dim: hashed (MD5, 4096 buckets) random Gaussian 512-d embeddings (64-d slice per block) of a token stream '
                     '(chars tagged last/earlier, function-word tags, per-word char 3/2/4/5-grams, word ids), '
                     'pooled in 8 blocks with exponential recency decays 0..6, plus current word id, layer-normed '
                     'with raw position value; truncation at 512 tokens.')

N_BUCKETS = 4096
DIM = 64
RATES = [0.0, 0.15, 0.3, 0.6, 1.0, 1.7, 3.0, 6.0]
MAX_TOKENS = 512
NGRAM_SPECS = [(3, 14), (2, 10), (4, 10), (5, 8)]

FUNCTION_WORDS = set("""
the a an this that these those my your his her its our their some any no every each all both
of in on at by for with from to into onto about over under after before through between among
against during without within along across behind beyond near upon off out up down around
and or but nor so yet if because although though while when where whether than as since until unless
i me you he him she it we us they them myself yourself himself herself itself ourselves themselves
who whom whose which what
is am are was were be been being have has had do does did will would shall should can could may might must
not there here then just very too also only
""".split())


def _bucket(tok):
    return int.from_bytes(hashlib.md5(tok.encode('utf-8')).digest()[:8], 'little') % N_BUCKETS


class Embedder:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.table = torch.randn(N_BUCKETS, DIM * len(RATES), generator=g)
        self.cache = {}

    def _b(self, tok):
        b = self.cache.get(tok)
        if b is None:
            b = self.cache[tok] = _bucket(tok)
        return b

    def tokens(self, text):
        words = text.lower().split()
        toks = []
        if not words:
            return toks, None
        prefix = ' '.join(words[:-1])
        last = words[-1]
        for c in prefix + (' ' if prefix else ''):
            toks.append('ch_prev:' + c)
        for c in last:
            toks.append('ch_last:' + c)
        for w in words:
            if w in FUNCTION_WORDS:
                toks.append('fw:' + w)
        for w in words:
            p = '<' + w + '>'
            for n, cap in NGRAM_SPECS:
                grams = [p[i:i + n] for i in range(len(p) - n + 1)][:cap]
                toks.extend(f'g{n}:' + g for g in grams)
        for w in words:
            toks.append('w:' + w)
        return toks[:MAX_TOKENS], 'w:' + last

    def __call__(self, texts):
        out = np.zeros((len(texts), DIM * len(RATES)), dtype=np.float32)
        rates = torch.tensor(RATES).unsqueeze(1)
        for i, text in enumerate(texts):
            toks, cur = self.tokens(text)
            if not toks:
                continue
            idx = torch.tensor([self._b(t) for t in toks])
            E = self.table[idx].reshape(-1, len(RATES), DIM)  # (L, 8, 64)
            L = len(toks)
            age = torch.arange(L - 1, -1, -1, dtype=torch.float32).unsqueeze(0)  # 0 for last token
            w = torch.exp(-rates * age)  # (8, L)
            w = w / w.sum(1, keepdim=True)
            pooled = torch.einsum('bl,lbd->bd', w, E)  # (8, 64)
            pooled = pooled + self.table[self._b(cur)].reshape(len(RATES), DIM)
            x = torch.cat([pooled.reshape(-1), torch.tensor([float(L)])])
            x = torch.nn.functional.layer_norm(x, x.shape)
            out[i] = x[:-1].numpy()
        return out


def build_embedder():
    return Embedder()
