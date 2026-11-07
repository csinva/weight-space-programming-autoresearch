"""Whisper-only reference encoder for the new-test rerun (the dashed line in Figure 1).

Same features as runs-wave/lean-whisper (diag_canonical.py 'Whisper-only' fit: the 1280-dim
{chunk}_whisper.npy rows, 4 FIR delays, per-chunk z-score, harness ridge), found the same way:
the harness passes only waveform windows, so the chunk is identified by the MD5 of its first
4096-sample window (evolve-wave/ling_features/fingerprint_map.json, extended here with the new
test chunks). An unknown fingerprint raises instead of returning zeros. Must reproduce the
old-test Whisper score (all-166 0.0817, grid 0.0933) before its new-test score is used.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from src import data
from src.eval import EncodingConfig, run_encoding

FP = '/path/to/ecog_data/circuit-building-autoresearch/evolve-wave/ling_features/fingerprint_map.json'
model_shorthand_name = 'Whisper-only'


class WhisperOnly:
    win = 4096

    def __init__(self):
        self.fp = json.load(open(FP))
        nt = os.environ.get('NEWTEST_DIR')
        if nt:
            for c in json.load(open(os.path.join(nt, 'test_candidates.json'))):
                seg = np.load(os.path.join(nt, 'wave_aligned', f'{c}_wave.npy'), mmap_mode='r')
                w0 = np.ascontiguousarray(np.asarray(seg[:self.win]).astype(np.float32))
                self.fp[hashlib.md5(w0.tobytes()).hexdigest()] = c

    def __call__(self, windows):
        T = windows.shape[0]
        w0 = np.ascontiguousarray(np.asarray(windows, dtype=np.float32)[0])
        c = self.fp.get(hashlib.md5(w0.tobytes()).hexdigest())
        if c is None:
            raise KeyError('waveform fingerprint not in map')
        arr = np.load(os.path.join(data.WAV_DIR, f'{c}_whisper.npy')).astype(np.float32)
        return arr[:T] if arr.shape[0] >= T else np.pad(arr, ((0, T - arr.shape[0]), (0, 0)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-train', type=int, default=500)
    ap.add_argument('--num-test', type=int, default=100)
    a = ap.parse_args()
    run_encoding(WhisperOnly(), EncodingConfig(num_train=a.num_train, num_test=a.num_test))
