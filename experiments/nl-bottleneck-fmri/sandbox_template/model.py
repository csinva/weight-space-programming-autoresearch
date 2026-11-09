"""Hand-written text features for predicting fMRI responses. Edit this file.

build_embedder() must return an object that maps a list of 10-gram strings (all of one story's
10-grams, in order) to a float array of shape (number of 10-grams, number of features).
"""
import numpy as np

MODEL_NAME = 'template-letters'
MODEL_DESCRIPTION = 'Counts of each letter in the last word (a placeholder to replace).'


class Embedder:
    def __call__(self, texts):
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.split()
            for c in words[-1] if words else '':
                if 'a' <= c <= 'z':
                    out[i, ord(c) - 97] += 1
        return out


def build_embedder():
    return Embedder()
