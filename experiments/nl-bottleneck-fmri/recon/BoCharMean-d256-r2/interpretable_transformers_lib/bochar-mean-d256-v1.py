"""Bag-of-characters features: random projection of character proportions over the last 64
characters, plus a faint final-character signal, layer-normalized. No training."""
import numpy as np

MODEL_NAME = 'bochar-mean-d256-v1'
MODEL_DESCRIPTION = ('Bag of characters: lowercase last 64 chars, 49-symbol alphabet; fixed random Gaussian '
                     '256-d vectors (seed 0) standardized per symbol, averaged equally over window, plus final '
                     'char vector at 1/16 weight, then layer-norm.')

ALPHABET = list('abcdefghijklmnopqrstuvwxyz') + [' ', "'"] + list('0123456789') + \
    list('.,!?;:-"()') + ['<unk>']  # 49 symbols
D = 256
WINDOW = 64


class Embedder:
    def __init__(self):
        self.idx = {c: i for i, c in enumerate(ALPHABET)}
        self.unk = len(ALPHABET) - 1
        rng = np.random.default_rng(0)
        E = rng.standard_normal((len(ALPHABET), D))
        E = (E - E.mean(1, keepdims=True)) / E.std(1, keepdims=True)  # zero mean, unit var, length ~16
        self.E = E

    def __call__(self, texts):
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, text in enumerate(texts):
            s = text.lower()[-WINDOW:]
            ids = [self.idx.get(c, self.unk) for c in s]
            if not ids:
                continue
            counts = np.bincount(ids, minlength=len(ALPHABET)).astype(float)
            h = counts / counts.sum() @ self.E          # equal-weight mean over window
            h = h + self.E[ids[-1]] / 16.0              # faint final-character signal
            h = (h - h.mean()) / (h.std() + 1e-5)       # layer norm
            out[i] = h
        return out


def build_embedder():
    return Embedder()
