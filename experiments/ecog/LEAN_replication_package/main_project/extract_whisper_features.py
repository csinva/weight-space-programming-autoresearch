#!/usr/bin/env python3
"""
Extract Whisper large-v3 encoder features using causal (right-aligned) sliding windows.

For each output frame at time t, feeds Whisper the 30s of audio ending at t,
so the encoder only sees past audio. With stride=1s, we take the last 1s
(50 encoder frames at 50Hz) from each window, giving max 1s of future
look-ahead from bidirectional attention within the window.

Output per chunk: wav_files/{id}_whisper.npy
  Shape: (T,  1280), dtype float16
  Where T matches melspec/vjepa_8x8 exactly (same trim, same alignment).

Whisper large-v3 specs:
  - 32 encoder layers, 1280-dim hidden, 128 mel bins
  - Input: 30s audio (480,000 samples @ 16kHz) -> 3000 mel frames
  - Encoder output: 1500 frames (50Hz) per 30s window
  - We extract last_hidden_state (after final LayerNorm)

Usage:
  python extract_whisper_features.py --start 0 --count 100
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel

sys.stdout.reconfigure(line_buffering=True)

WHISPER_MODEL = "openai/whisper-large-v3"
WHISPER_SR = 16000           # Whisper expects 16kHz
WHISPER_CONTEXT_SEC = 30.0   # Max context window
WHISPER_CONTEXT_SAMPLES = int(WHISPER_CONTEXT_SEC * WHISPER_SR)  # 480,000
WHISPER_HZ = 50              # Encoder output rate (1500 frames / 30s)
WHISPER_DIM = 1280           # Hidden dimension

TARGET_HZ = 30              # Output rate to match other features
TRIM_FRAMES_30HZ = 30       # 1.0s trimmed from start (and effectively end) at 30Hz

STRIDE_SEC = 1.0            # Take last 1s of each window
STRIDE_FRAMES_50HZ = int(STRIDE_SEC * WHISPER_HZ)  # 50 frames per window


def extract_chunk(wav_path, mel_T, processor, model, device, stride_sec=STRIDE_SEC):
    """Extract Whisper encoder features for one chunk using causal sliding windows.

    Args:
        wav_path: Path to WAV file (44.1kHz or any rate)
        mel_T: Target output length at 30Hz (from melspec shape)
        processor: WhisperProcessor
        model: WhisperModel (encoder only)
        device: torch device
        stride_sec: Seconds of output to take from end of each window

    Returns:
        features: (mel_T, 1280) float32 array, aligned to match melspec/vjepa
    """
    # Load and resample audio to 16kHz
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != WHISPER_SR:
        resampler = torchaudio.transforms.Resample(sr, WHISPER_SR)
        wav = resampler(wav)
    audio = wav.squeeze(0).numpy()  # (N,) at 16kHz

    total_duration = len(audio) / WHISPER_SR

    # The melspec/vjepa features cover [trim_start, trim_start + mel_T/30Hz]
    # where trim_start = TRIM_FRAMES_30HZ / 30Hz = 1.0s
    # We need Whisper features for the same time range at 50Hz first,
    # then downsample to 30Hz.

    trim_start_sec = TRIM_FRAMES_30HZ / TARGET_HZ  # 1.0s
    feature_duration_sec = mel_T / TARGET_HZ
    feature_end_sec = trim_start_sec + feature_duration_sec

    # We need 50Hz features from trim_start_sec to feature_end_sec
    T_50hz = int(round(feature_duration_sec * WHISPER_HZ))

    # Allocate output at 50Hz
    features_50hz = np.zeros((T_50hz, WHISPER_DIM), dtype=np.float32)

    stride_frames = int(stride_sec * WHISPER_HZ)

    # Right-aligned causal windows:
    # For output time t, feed audio [t - 30s, t]
    # We process from the earliest needed time forward.
    # First output time: trim_start_sec (1.0s)
    # Last output time: feature_end_sec

    # Process windows: each window ends at window_end_sec
    # From each window we take the last stride_sec of encoder output
    out_pos = 0  # Current write position in features_50hz

    # First window: may not have full 30s of history
    # Window ends at trim_start_sec + stride_sec, covering [max(0, end-30), end]
    # We might get fewer than stride_frames usable frames at the very start

    window_end_sec = trim_start_sec + stride_sec

    while out_pos < T_50hz:
        # How many frames we'll write from this window
        frames_needed = min(stride_frames, T_50hz - out_pos)

        # Audio window: [window_end_sec - 30s, window_end_sec]
        window_start_sec = max(0.0, window_end_sec - WHISPER_CONTEXT_SEC)
        window_actual_dur = window_end_sec - window_start_sec

        # Extract audio segment, pad to exactly 30s for Whisper
        start_sample = int(round(window_start_sec * WHISPER_SR))
        end_sample = int(round(window_end_sec * WHISPER_SR))
        end_sample = min(end_sample, len(audio))
        audio_segment = audio[start_sample:end_sample]

        # Right-pad with silence to 30s (480,000 samples) — Whisper encoder
        # requires exactly 3000 mel frames (= 30s). For causal windows shorter
        # than 30s (early in the chunk), the real audio is LEFT-aligned and
        # silence pads the right, which is correct: encoder sees the real audio
        # in positions 0..actual_dur, padding after.
        if len(audio_segment) < WHISPER_CONTEXT_SAMPLES:
            audio_segment = np.pad(
                audio_segment, (0, WHISPER_CONTEXT_SAMPLES - len(audio_segment)))

        inputs = processor(
            audio_segment, sampling_rate=WHISPER_SR,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)  # (1, 128, 3000)

        with torch.no_grad():
            encoder_output = model.encoder(input_features, return_dict=True)
            hidden = encoder_output.last_hidden_state.squeeze(0).cpu().numpy()
            # hidden: (1500, 1280)

        # How many frames correspond to actual audio (not padding)?
        actual_frames = int(round(window_actual_dur * WHISPER_HZ))
        actual_frames = min(actual_frames, hidden.shape[0])

        # Take the last frames_needed frames from the actual audio portion
        # These correspond to the most recent audio (right-aligned = causal)
        take_start = max(0, actual_frames - frames_needed)
        taken = hidden[take_start:actual_frames]

        # If we got fewer frames than needed (very start of audio), pad from left
        if len(taken) < frames_needed:
            pad = np.zeros((frames_needed - len(taken), WHISPER_DIM), dtype=np.float32)
            taken = np.concatenate([pad, taken], axis=0)

        features_50hz[out_pos:out_pos + frames_needed] = taken[:frames_needed]
        out_pos += frames_needed
        window_end_sec += stride_sec

    # Downsample 50Hz -> 30Hz to match melspec/vjepa
    # Use linear interpolation
    t_50 = np.arange(T_50hz) / WHISPER_HZ  # time in seconds at 50Hz
    t_30 = np.arange(mel_T) / TARGET_HZ    # time in seconds at 30Hz

    features_30hz = np.zeros((mel_T, WHISPER_DIM), dtype=np.float32)
    for d in range(WHISPER_DIM):
        features_30hz[:, d] = np.interp(t_30, t_50, features_50hz[:, d])

    return features_30hz


def try_claim(lock_path):
    """Atomically claim a chunk via exclusive file creation. Returns True if claimed."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper large-v3 encoder features")
    parser.add_argument("--data-dir", default="wav_files")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write {id}_whisper.npy (defaults to --data-dir). Use a "
                             "writable fileset when --data-dir is on a quota-exhausted filesystem.")
    parser.add_argument("--master-list", default="master_list.json")
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC,
                        help="Seconds of output per window (causal stride)")
    parser.add_argument("--lock-dir", default="wav_files/.whisper_locks",
                        help="Directory for lock files (work-stealing coordination)")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Whisper model: {WHISPER_MODEL}")
    print(f"Causal stride: {args.stride_sec}s")
    print(f"PID: {os.getpid()}")

    # Load model
    print("Loading Whisper model...")
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperModel.from_pretrained(WHISPER_MODEL).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")

    with open(args.master_list) as f:
        master_list = json.load(f)

    os.makedirs(args.lock_dir, exist_ok=True)

    # Shuffle so parallel workers don't all hit the same chunks
    import random
    order = list(range(len(master_list)))
    random.shuffle(order)

    n_done = 0
    n_skipped = 0
    total = len(master_list)

    print(f"Scanning {total} chunks for remaining work...")

    for idx in order:
        chunk_id = master_list[idx]
        out_path = os.path.join(args.out_dir, f"{chunk_id}_whisper.npy")
        wav_path = os.path.join(args.data_dir, f"{chunk_id}.wav")
        mel_path = os.path.join(args.data_dir, f"{chunk_id}_melspec.npy")
        lock_path = os.path.join(args.lock_dir, f"{chunk_id}.lock")

        # Already done
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        # Missing source files
        if not os.path.exists(wav_path) or not os.path.exists(mel_path):
            n_skipped += 1
            continue

        # Try to claim this chunk atomically
        if not try_claim(lock_path):
            continue  # Another worker got it

        # Double-check after claiming (race window)
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        t0_chunk = time.time()

        mel = np.load(mel_path)
        mel_T = mel.shape[1]

        try:
            features = extract_chunk(
                wav_path, mel_T, processor, model, device,
                stride_sec=args.stride_sec
            )

            # Write to temp file then rename for atomicity
            # Use .npy extension on tmp so np.save doesn't append another .npy
            tmp_path = out_path[:-4] + f".tmp.{os.getpid()}.npy"
            np.save(tmp_path, features.astype(np.float16))
            os.rename(tmp_path, out_path)

            elapsed = time.time() - t0_chunk
            size_mb = os.path.getsize(out_path) / 1e6
            n_done += 1

            # Count progress
            done_total = len([f for f in os.listdir(args.out_dir)
                              if f.endswith('_whisper.npy')]) if n_done % 10 == 0 else '?'

            print(f"  {chunk_id}: ({features.shape[0]}, {features.shape[1]}) "
                  f"{size_mb:.1f} MB, {elapsed:.1f}s "
                  f"[this worker: {n_done}, total: {done_total}/{total}]")

        except Exception as e:
            print(f"  {chunk_id}: ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Remove lock so another worker can retry
            try:
                os.remove(lock_path)
            except OSError:
                pass

    print(f"\nWorker done: {n_done} extracted, {n_skipped} skipped/existing")


if __name__ == "__main__":
    main()
