"""Pre-trim per-chunk aligned waveform segments for the patient1 waveform track.

For each chunk we slice the 44.1 kHz wav to a segment that, when framed by
`src.data.frame_windows` (hop=1470, center semantics), yields exactly the same
30 Hz frames as `_melspec.npy` / `_ecog_hasson_highgamma.npy` — i.e. feature frame
i is centered at wav sample (i + 30)*1470 (the 1 s extraction crop is baked in),
with WINMAX//2 samples of real-audio context on each side so any win <= WINMAX is
exactly centered. Segments are saved as float16 to a scratch directory.

  python extract_wave_aligned.py --verify 2cffbdfa_0014      # prove alignment, no write
  python extract_wave_aligned.py --chunks-file <json> [--limit N]   # extract a list
  python extract_wave_aligned.py --speech-rich --num-train 700      # extract test + train pool
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import join

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data
from src.data import SR, HOP, EXTRACT_CROP_FRAMES, WINMAX, WAVE_DIR, WAV_DIR, MASTER_LIST


def _read_wav_mono(chunk: str) -> np.ndarray:
    sr, wav = wavfile.read(join(WAV_DIR, f"{chunk}.wav"))
    assert sr == SR, f"{chunk}: sr={sr} != {SR}"
    if wav.ndim > 1:                       # match original: channel 1 if stereo
        wav = wav[:, 1 if wav.shape[1] > 1 else 0]
    return wav.astype(np.float32, copy=False)


def _neural_T(chunk: str) -> int:
    return int(np.load(join(WAV_DIR, f"{chunk}_ecog_hasson_highgamma.npy")).shape[1])


def build_segment(chunk: str, T: int) -> np.ndarray:
    """Slice the aligned segment covering 30 Hz frames [0..T-1] with WINMAX//2 context."""
    wav = _read_wav_mono(chunk)
    s0 = EXTRACT_CROP_FRAMES * HOP - WINMAX // 2          # 42052 for defaults (>0)
    seg_len = (T - 1) * HOP + WINMAX
    need = s0 + seg_len
    if need > len(wav):
        wav = np.pad(wav, (0, need - len(wav)), mode="reflect")
    return wav[s0:s0 + seg_len].astype(np.float16)


def extract(chunk: str) -> str:
    T = _neural_T(chunk)
    seg = build_segment(chunk, T)
    assert data.seg_to_T(len(seg)) == T, f"{chunk}: seg_to_T mismatch"
    os.makedirs(WAVE_DIR, exist_ok=True)
    out = join(WAVE_DIR, f"{chunk}_wave.npy")
    np.save(out, seg)
    return out


def verify(chunk: str) -> None:
    """Reproduce _melspec.npy from the aligned windows -> prove the framing is aligned."""
    import torch
    import torchaudio

    T = _neural_T(chunk)
    seg = build_segment(chunk, T).astype(np.float32)
    win = 1024
    windows = data.frame_windows(seg, win=win, n_frames=T)          # (T, 1024)

    x = torch.from_numpy(windows)
    hann = torch.hann_window(win, periodic=True)
    power = (torch.fft.rfft(x * hann, n=win).abs() ** 2)            # (T, 513)
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=win // 2 + 1, f_min=20, f_max=8000, n_mels=50,
        sample_rate=SR, norm=None, mel_scale="htk")                # (513, 50)
    mel = power @ mel_fb                                            # (T, 50)
    db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
    db = torch.maximum(db, db.max() - 70.0)
    repro = db.numpy()                                             # (T, 50)

    saved = np.load(join(WAV_DIR, f"{chunk}_melspec.npy")).T[:T]   # (T, 50)
    m = min(repro.shape[0], saved.shape[0])
    diff = np.abs(repro[:m] - saved[:m])
    print(f"[verify {chunk}] T={T}  reproduced-vs-saved melspec: "
          f"max|Δ|={diff.max():.4f}  mean|Δ|={diff.mean():.4f}")

    # alignment lag check: per-frame energy cross-correlation (peak must be at lag 0)
    a = repro[:m].mean(1); b = saved[:m].mean(1)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    xc = np.correlate(a, b, mode="full")
    lag = xc.argmax() - (m - 1)
    print(f"[verify {chunk}] energy cross-corr peak lag = {lag} frames (expect 0); "
          f"corr@0 = {float((a * b).mean()):.4f}")
    if diff.max() < 0.5 and lag == 0:
        print("[verify] PASS — waveform windows are aligned to the melspec/high-gamma grid.")
    else:
        print("[verify] WARNING — misalignment or reproduction mismatch; investigate.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", metavar="CHUNK", default=None)
    ap.add_argument("--chunks-file", default=None)
    ap.add_argument("--speech-rich", action="store_true",
                    help="extract the speech-rich test set + a non-450a36e4 train pool")
    ap.add_argument("--num-train", type=int, default=700)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.verify:
        verify(args.verify)
        return

    chunks = []
    if args.speech_rich:
        test = json.load(open(join(os.path.dirname(__file__), "speech_rich_test_chunks.json")))
        all_chunks = json.load(open(MASTER_LIST))
        def ok(c):
            return (os.path.exists(join(WAV_DIR, f"{c}_melspec.npy"))
                    and os.path.exists(join(WAV_DIR, f"{c}_ecog_hasson_highgamma.npy"))
                    and os.path.exists(join(WAV_DIR, f"{c}_whisper.npy")))
        valid = [c for c in all_chunks if ok(c)]
        rng = np.random.RandomState(42); rng.shuffle(valid)
        pool = [c for c in valid if not c.startswith("450a36e4")][:args.num_train]
        chunks = list(dict.fromkeys(test + pool))   # test first, dedup, preserve order
    elif args.chunks_file:
        chunks = json.load(open(args.chunks_file))
    if args.limit:
        chunks = chunks[:args.limit]

    os.makedirs(WAVE_DIR, exist_ok=True)
    done = skipped = 0
    for i, c in enumerate(chunks):
        out = join(WAVE_DIR, f"{c}_wave.npy")
        if os.path.exists(out):
            skipped += 1
            continue
        try:
            extract(c)
            done += 1
        except Exception as e:
            print(f"  !! {c}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)} (extracted {done}, skipped {skipped})")
    print(f"done: extracted {done}, skipped {skipped}, total requested {len(chunks)} -> {WAVE_DIR}")


if __name__ == "__main__":
    main()
