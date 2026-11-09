"""Bag-of-characters model: random-projected character histogram of last 64 chars + faint last char, layer-normed."""
import numpy as np

MODEL_NAME = 'rec-bochar-mean-d256-v1'
MODEL_DESCRIPTION = ('Lowercased last 64 chars; 49-symbol alphabet (a-z, space, apostrophe, digits, punctuation, unknown); '
                     'seed-0 Gaussian 256-d vectors standardized per vector; equal-weight mean over window plus final '
                     'char vector scaled 1/16; layer norm.')

ALPHABET = list('abcdefghijklmnopqrstuvwxyz') + [' ', "'"] + list('0123456789') + list('.,!?;:-"()')
IDX = {c: i for i, c in enumerate(ALPHABET)}
UNK = len(ALPHABET)  # 48 -> 49 symbols total
D = 256


class Embedder:
    def __init__(self):
        rng = np.random.default_rng(0)
        E = rng.standard_normal((UNK + 1, D))
        E = (E - E.mean(1, keepdims=True)) / E.std(1, keepdims=True)
        self.E = E

    def __call__(self, texts):
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, t in enumerate(texts):
            s = t.lower()[-64:]
            if not s:
                continue
            ids = [IDX.get(c, UNK) for c in s]
            x = self.E[ids].mean(0) + self.E[ids[-1]] / 16.0
            x = (x - x.mean()) / (x.std() + 1e-5)
            out[i] = x
        return out


def build_embedder():
    return Embedder()
