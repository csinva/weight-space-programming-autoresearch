"""Aligned waveform segments for each subject's train + test chunks -> <subj>/wave_aligned/.
Same math as evolve-wave/extract_wave_aligned.build_segment (task 0 first proves bit-identity on an
existing patient1 segment), with the subject's own wav/high-gamma directory."""
import glob, json, os, sys
import numpy as np
from scipy.io import wavfile
HERE = os.path.dirname(os.path.abspath(__file__))
SR, HOP, WINMAX, CROP = 44100, 1470, 4096, 30


def build_segment(wavdir, c):
    T = int(np.load(f'{wavdir}/{c}_ecog_hasson_highgamma.npy', mmap_mode='r').shape[1])
    sr, wav = wavfile.read(f'{wavdir}/{c}.wav')
    assert sr == SR, (c, sr)
    if wav.ndim > 1:
        wav = wav[:, 1 if wav.shape[1] > 1 else 0]
    wav = wav.astype(np.float32, copy=False)
    s0 = CROP * HOP - WINMAX // 2
    seg_len = (T - 1) * HOP + WINMAX
    if s0 + seg_len > len(wav):
        wav = np.pad(wav, (0, s0 + seg_len - len(wav)), mode='reflect')
    seg = wav[s0:s0 + seg_len].astype(np.float16)
    assert (len(seg) - WINMAX) // HOP + 1 == T
    return seg


task, ntasks = int(sys.argv[1]), int(sys.argv[2])
if task == 0:
    ref = np.load('/path/to/scratch/patient1_wave_aligned/450a36e4_0190_wave.npy')
    new = build_segment('/path/to/ecog_data/wav_files', '450a36e4_0190')
    assert ref.dtype == new.dtype and np.array_equal(ref, new)
    print('[check] patient1 450a36e4_0190 rebuilt bit-identical', flush=True)
items = []
for f in sorted(glob.glob(os.path.join(HERE, 'NY[0-9]*', 'split.json'))):
    cfg = json.load(open(f))
    if cfg['subject'] == 'patient1VAL':
        continue
    items += [(cfg['data_dir'], cfg['wave_dir'], c) for c in cfg['train'] + cfg['test']]
for wavdir, out, c in items[task::ntasks]:
    p = os.path.join(out, f'{c}_wave.npy')
    if not os.path.exists(p):
        np.save(p + '.tmp.npy', build_segment(wavdir, c)); os.replace(p + '.tmp.npy', p)
print(f'task {task}: {len(items[task::ntasks])} chunks done', flush=True)
