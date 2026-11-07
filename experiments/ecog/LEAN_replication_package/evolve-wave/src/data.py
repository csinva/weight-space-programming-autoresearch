"""Load the patient1 RAW-WAVEFORM -> ECoG high-gamma encoding dataset.

This is the waveform variant of the project: instead of the precomputed
mel-spectrogram, the encoder starts from the raw audio. To keep everything
aligned to the existing 30 Hz high-gamma timeline, the wav is **pre-trimmed**
into per-chunk aligned segments (see ../extract_wave_aligned.py), and this module
exposes a FIXED framing helper `frame_windows` that turns a segment into one audio
window per 30 Hz frame, on exactly the same grid the mel-spectrogram used.

Alignment recipe (verified to reproduce _melspec.npy, see extract_wave_aligned.py):
  - audio: 44100 Hz mono.
  - 30 Hz frames at HOP = 44100/30 = 1470 samples; frame i centered at sample i*HOP
    (torchaudio center=True semantics; interior frames see only real audio).
  - the mel pipeline cropped EXTRACT_CROP_FRAMES (=30, i.e. 1.0 s) off both ends to
    match the neural 1 s crop. So feature frame i corresponds to full-STFT frame
    (i + EXTRACT_CROP_FRAMES), i.e. wav sample (i + 30)*HOP.
  - a chunk's aligned segment covers feature frames [0 .. T-1] with WINMAX//2 samples
    of real-audio context on each side, so any window length win <= WINMAX is exactly
    centered on the same frame centers — the agent cannot break alignment.

The downstream pipeline (features.py) then applies the SAME extra trim (30 frames)
+ neural lag (3 frames) that the mel track uses, so wave features are paired with
high-gamma identically.
"""
from __future__ import annotations

import json
import os
from os.path import join

import numpy as np

# audio / grid constants (verified against the saved mel-spectrograms)
SR = 44100              # wav sample rate
OUT_SR = 30             # feature/neural sample rate
HOP = SR // OUT_SR      # 1470 samples per 30 Hz frame
EXTRACT_CROP_FRAMES = 30  # 1.0 s cropped off each end during mel extraction
WINMAX = 4096           # max analysis window the aligned segment supports (~93 ms)

N_ELECTRODES = 166

PROJECT_ROOT = "/path/to/ecog_data"
WAV_DIR = join(PROJECT_ROOT, "wav_files")                       # *_ecog_hasson_highgamma.npy
MASTER_LIST = join(PROJECT_ROOT, "master_list.json")
WAVE_DIR = "/path/to/scratch/patient1_wave_aligned"          # pre-trimmed {chunk}_wave.npy


def seg_to_T(seg_len: int) -> int:
    """Number of 30 Hz frames an aligned segment of length `seg_len` supports."""
    return (seg_len - WINMAX) // HOP + 1


def frame_windows(seg: np.ndarray, win: int = 2048, n_frames: int | None = None) -> np.ndarray:
    """FIXED framer: aligned segment (1-D) -> (T, win) audio windows on the 30 Hz grid.

    Window for frame i is centered on the same sample as mel frame i, for ANY
    win <= WINMAX. This is the only sanctioned way to turn a segment into frames;
    it guarantees alignment regardless of the front-end design.
    """
    assert win <= WINMAX, f"win={win} exceeds WINMAX={WINMAX}"
    seg = np.asarray(seg, dtype=np.float32)
    T = seg_to_T(len(seg))
    if n_frames is not None:
        T = min(T, n_frames)
    start0 = (WINMAX - win) // 2
    sw = np.lib.stride_tricks.sliding_window_view(seg, win)  # (len-win+1, win)
    idx = start0 + np.arange(T) * HOP
    return np.ascontiguousarray(sw[idx])                     # (T, win)


def _has_seg(chunk: str) -> bool:
    return (os.path.exists(join(WAVE_DIR, f"{chunk}_wave.npy"))
            and os.path.exists(join(WAV_DIR, f"{chunk}_ecog_hasson_highgamma.npy")))


def _valid_chunks(seed: int = 42) -> list[str]:
    """Master-list chunks that have a pre-extracted aligned wave segment + high-gamma,
    deterministically shuffled."""
    with open(MASTER_LIST) as f:
        all_chunks = json.load(f)
    valid = [c for c in all_chunks if _has_seg(c)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid)
    return valid


def get_chunk_names(num_train: int | None = None, num_test: int = 20,
                    seed: int = 42) -> tuple[list[str], list[str]]:
    """Return (train_chunks, test_chunks). Same env-driven fixed-test mode as the
    mel track: if ECOG_TEST_CHUNKS_FILE is set, those chunks are the held-out test
    set and ECOG_EXCLUDE_PREFIX chunks are dropped from training."""
    valid = _valid_chunks(seed=seed)

    test_file = os.environ.get("ECOG_TEST_CHUNKS_FILE")
    if test_file:
        with open(test_file) as f:
            requested = json.load(f)
        test = [c for c in requested if _has_seg(c)]
        testset = set(test)
        excl = os.environ.get("ECOG_EXCLUDE_PREFIX", "")
        pool = [c for c in valid if c not in testset and not (excl and c.startswith(excl))]
        train = pool if num_train is None else pool[:num_train]
        return train, test

    test = valid[:num_test]
    rest = valid[num_test:]
    train = rest if num_train is None else rest[:num_train]
    return train, test


def load_stim(chunks: list[str]) -> dict[str, np.ndarray]:
    """{chunk: 1-D float32 aligned waveform segment}. The 'stimulus' is the raw audio;
    features.py frames it with `frame_windows` using the embedder's declared `win`."""
    return {c: np.load(join(WAVE_DIR, f"{c}_wave.npy")).astype(np.float32)
            for c in chunks}


def load_responses(chunks: list[str]) -> dict[str, np.ndarray]:
    """{chunk: (T, 166) float32} high-gamma ECoG responses."""
    return {c: np.load(join(WAV_DIR, f"{c}_ecog_hasson_highgamma.npy")).T.astype(np.float32)
            for c in chunks}
