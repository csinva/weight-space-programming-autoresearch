"""Per-subject sanity checks before the multi-subject rerun: data formats, audio alignment
(waveform framing must reproduce the saved melspec at lag 0, as verified for patient1), Whisper
alignment (rows = melspec frames), electrode counts and the grid rows."""
import json, os, sys
import numpy as np
from scipy.io import wavfile
import torch
REPO = '/path/to/ecog_data/circuit-building-autoresearch'
sys.path.insert(0, os.path.join(REPO, 'evolve-wave'))
from src import data
SUBJ = {'patient2': '236d910e', 'patient3': 'e3d31b95', 'patient4': '2eeb170d', 'patient5': '5e416143', 'patient6': '5d2842ef',
        'patient1': '450a36e4'}
HOP, WINMAX, CROP, SR = 1470, 4096, 30, 44100


def dirs(s):
    if s == 'patient1':
        return '/path/to/ecog_data/wav_files', '/path/to/ecog_data/wav_files'
    return f'/path/to/scratch/ondemand/{s}_Dayfiles/Dayfiles/wav_files', f'/path/to/ecog_data/whisper_features/{s}'


def htk_fbanks(n_freqs, f_min, f_max, n_mels, sr):
    """torchaudio.functional.melscale_fbanks(norm=None, mel_scale='htk'), in NumPy."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    imel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, sr // 2, n_freqs)
    f_pts = imel(np.linspace(mel(f_min), mel(f_max), n_mels + 2))
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up))


def verify(wavdir, c):
    T = np.load(f'{wavdir}/{c}_ecog_hasson_highgamma.npy', mmap_mode='r').shape[1]
    sr, wav = wavfile.read(f'{wavdir}/{c}.wav')
    if wav.ndim > 1:
        wav = wav[:, 1 if wav.shape[1] > 1 else 0]
    wav = wav.astype(np.float32)
    s0 = CROP * HOP - WINMAX // 2; L = (T - 1) * HOP + WINMAX
    if s0 + L > len(wav):
        wav = np.pad(wav, (0, s0 + L - len(wav)), mode='reflect')
    seg = wav[s0:s0 + L].astype(np.float16).astype(np.float32)
    win = 1024
    x = torch.from_numpy(data.frame_windows(seg, win=win, n_frames=T))
    p = torch.fft.rfft(x * torch.hann_window(win, periodic=True), n=win).abs() ** 2
    fb = torch.from_numpy(htk_fbanks(513, 20.0, 8000.0, 50, SR)).float()
    db = 10 * torch.log10(torch.clamp(p @ fb, min=1e-10)); db = torch.maximum(db, db.max() - 70).numpy()
    saved = np.load(f'{wavdir}/{c}_melspec.npy').T
    m = min(len(db), len(saved)); d = np.abs(db[:m] - saved[:m])
    a = db[:m].mean(1); b = saved[:m].mean(1); a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    lag = int(np.correlate(a, b, 'full').argmax() - (m - 1))
    return sr, T, saved.shape[0], float(d.max()), float(d.mean()), lag


for s, held in SUBJ.items():
    wavdir, whdir = dirs(s)
    root = os.path.dirname(wavdir)
    ml = json.load(open(os.path.join(root, 'master_list.json')))
    hg = np.load(f'{wavdir}/{ml[0]}_ecog_hasson_highgamma.npy', mmap_mode='r')
    held_ml = [c for c in ml if c.startswith(held)]
    have_wh = sum(os.path.exists(f'{whdir}/{c}_whisper.npy') for c in ml)
    print(f'== {s}: master {len(ml)} (held-out {held}: {len(held_ml)}), whisper for {have_wh}, HG {hg.shape} {hg.dtype}', flush=True)
    okw = lambda c: os.path.exists(f'{whdir}/{c}_whisper.npy')
    held_w = [c for c in held_ml if okw(c)]
    print(f'   held-out with whisper: {len(held_w)}', flush=True)
    for c in (held_w[len(held_w) // 2], [x for x in ml if not x.startswith(held) and okw(x)][7]):
        sr, T, Tm, dmax, dmean, lag = verify(wavdir, c)
        wh = np.load(f'{whdir}/{c}_whisper.npy', mmap_mode='r').shape
        print(f'   {c}: sr {sr} T_hg {T} T_mel {Tm} T_whisper {wh} | melspec repro max|d| {dmax:.3f} mean {dmean:.4f} lag {lag}', flush=True)
