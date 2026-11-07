"""Load the patient1 melspec -> ECoG high-gamma encoding dataset.

The "task" is predicting held-out ECoG high-gamma cortical activity from the
audio mel-spectrogram. Unlike the fMRI sibling project there is no TR timeline:
every feature in this project is already sampled at **30 Hz** and time-aligned on
axis 1, so no Lanczos / TR downsampling is needed.

Per chunk:
    melspec     : (50, T)  mel-spectrogram, 50 bins (20-8000 Hz)
    high-gamma  : (166, T) processed high-gamma ECoG, 166 good channels

This module returns them transposed to (T, 50) / (T, 166) so time is axis 0
(matching the fMRI pipeline's (n_trs, n_voxels) convention). A chunk plays the
role a "story" played in the fMRI project.
"""
from __future__ import annotations

import json
import os
from os.path import join

import numpy as np

# Absolute paths so the pipeline works from inside any runs-ecog/<tag>/ folder.
PROJECT_ROOT = "/path/to/ecog_data"
WAV_DIR = join(PROJECT_ROOT, "wav_files")
MASTER_LIST = join(PROJECT_ROOT, "master_list.json")

N_ELECTRODES = 166      # good channels in *_ecog_hasson_highgamma.npy
N_MELS = 50             # mel bins in *_melspec.npy
SAMPLE_RATE_HZ = 30     # every feature is sampled at 30 Hz


def _has_both(chunk: str) -> bool:
    return (os.path.exists(join(WAV_DIR, f"{chunk}_melspec.npy"))
            and os.path.exists(join(WAV_DIR, f"{chunk}_ecog_hasson_highgamma.npy")))


def _valid_chunks(seed: int = 42) -> list[str]:
    """All master-list chunks that have both a melspec and a high-gamma file,
    deterministically shuffled (seed=42, matching the project's split convention)."""
    with open(MASTER_LIST) as f:
        all_chunks = json.load(f)
    valid = [c for c in all_chunks if _has_both(c)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid)
    return valid


def get_chunk_names(num_train: int | None = None, num_test: int = 20,
                    seed: int = 42) -> tuple[list[str], list[str]]:
    """Return (train_chunks, test_chunks).

    Default: deterministic shuffle, held-out test = first `num_test` chunks (fixed
    for a given (num_test, seed) regardless of num_train). `num_train=None` uses
    all remaining chunks for training.

    Fixed-test mode (env-driven, for the speech-rich eval): if
    ``ECOG_TEST_CHUNKS_FILE`` points to a JSON list of chunk IDs, those become the
    held-out test set (filtered to chunks with both files) and `num_test` is
    ignored. Training draws from the remaining valid chunks, additionally
    excluding any chunk whose ID starts with ``ECOG_EXCLUDE_PREFIX`` (set this to
    the test recording's prefix to avoid adjacency leakage).
    """
    valid = _valid_chunks(seed=seed)

    test_file = os.environ.get("ECOG_TEST_CHUNKS_FILE")
    if test_file:
        with open(test_file) as f:
            requested = json.load(f)
        test = [c for c in requested if _has_both(c)]
        testset = set(test)
        excl = os.environ.get("ECOG_EXCLUDE_PREFIX", "")
        pool = [c for c in valid
                if c not in testset and not (excl and c.startswith(excl))]
        train = pool if num_train is None else pool[:num_train]
        return train, test

    test = valid[:num_test]
    rest = valid[num_test:]
    train = rest if num_train is None else rest[:num_train]
    return train, test


def load_stim(chunks: list[str]) -> dict[str, np.ndarray]:
    """{chunk: (T, 50) float32} mel-spectrogram stimulus."""
    return {c: np.load(join(WAV_DIR, f"{c}_melspec.npy")).T.astype(np.float32)
            for c in chunks}


def load_responses(chunks: list[str]) -> dict[str, np.ndarray]:
    """{chunk: (T, 166) float32} high-gamma ECoG responses."""
    return {c: np.load(join(WAV_DIR, f"{c}_ecog_hasson_highgamma.npy")).T.astype(np.float32)
            for c in chunks}
