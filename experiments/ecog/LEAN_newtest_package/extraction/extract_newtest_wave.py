"""Aligned waveform segments for the new test chunks, written to newtest/wave_aligned/ (NOT the
harness WAVE_DIR: adding files there would change data._valid_chunks() and reshuffle training).
Uses the harness's own build_segment; first proves it reproduces an existing segment exactly."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'evolve-wave'))
from extract_wave_aligned import build_segment, _neural_T  # noqa: E402
from src import data  # noqa: E402

OUT = os.path.join(HERE, 'wave_aligned')
task, ntasks = int(sys.argv[1]), int(sys.argv[2])
if task == 0:
    for c in ('450a36e4_0190', json.load(open(os.path.join(HERE, 'train_frozen.json')))[0]):
        ref = np.load(os.path.join(data.WAVE_DIR, f'{c}_wave.npy'))
        new = build_segment(c, _neural_T(c))
        assert ref.dtype == new.dtype and ref.shape == new.shape and np.array_equal(ref, new), c
        print(f'[check] {c} rebuilt bit-identical {new.shape}', flush=True)
cand = json.load(open(os.path.join(HERE, 'test_candidates.json')))
for c in cand[task::ntasks]:
    out = os.path.join(OUT, f'{c}_wave.npy')
    if os.path.exists(out):
        continue
    T = _neural_T(c)
    seg = build_segment(c, T)
    assert data.seg_to_T(len(seg)) == T, c
    np.save(out + '.tmp.npy', seg); os.replace(out + '.tmp.npy', out)
print(f'task {task} done', flush=True)
