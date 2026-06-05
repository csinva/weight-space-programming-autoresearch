"""Fixed data layer: stimulus + neural -> aligned, split (X, Y) tensors.

Modality-agnostic. Each ``Dataset`` declares a ``stimulus`` modality and a
``neural`` modality; this loader dispatches the right NeuralSet extractor for
each, aligns them per segment, and returns a contiguous train/test split. The
autoresearch agent should NOT need to edit this file.

Stimulus extractors are OFFLINE-friendly (no model-weight downloads):
  * image -> RawPixels (flattened RGB)
  * text  -> WordLength + WordFrequency (the trigger word's [length, zipf-freq])
  * audio -> MelSpectrum (time-averaged Mel bands)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import numpy as np
import torch
import neuralset as ns
from neuralset.extractors.image import BaseClassicImageExtractor

from config import Dataset
import nsd_subj01  # registers the NsdSubj01 study so name="NsdSubj01" resolves
import li2022_subj  # registers the Li2022PetitSubj study so name="Li2022PetitSubj" resolves


class RawPixels(BaseClassicImageExtractor):
    """Flattened, [0,1]-normalised RGB pixels of each stimulus image."""

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        return (image.astype("float32") / 255.0).reshape(-1)


_NEURAL = {
    # Bel2026 raw .fif is Elekta MaxShield -> allow_maxshield or it raises.
    "meg": lambda: ns.extractors.MegExtractor(frequency=100.0, allow_maxshield=True),
    "eeg": lambda: ns.extractors.EegExtractor(frequency=100.0),
    "fmri": lambda: ns.extractors.FmriExtractor(),
}


def _stim_extractor(d: Dataset):
    """Build the stimulus extractor for the dataset's modality.

    Text/audio extractors are imported lazily so the image path needs none of
    their (optional) dependencies.
    """
    if d.stimulus == "image":
        return RawPixels(imsize=d.imsize)
    if d.stimulus in ("text", "word"):
        from neuralset.extractors.text import WordFrequency, WordLength
        from neuralset.extractors.meta import AggregatedExtractor

        # "trigger" aggregation => features of the triggering word only, so the
        # neural HRF offset (segment start) never leaks into the stimulus.
        return AggregatedExtractor(
            extractors=[
                WordLength(aggregation="trigger"),
                WordFrequency(aggregation="trigger"),
            ],
            extractor_aggregation="cat",
        )
    if d.stimulus == "audio":
        from neuralset.extractors.audio import MelSpectrum
        from neuralset.extractors.meta import TimeAggregatedExtractor

        return TimeAggregatedExtractor(extractor=MelSpectrum(), time_aggregation="mean")
    raise ValueError(f"unknown stimulus modality: {d.stimulus!r}")


def _load_workers() -> int:
    """DataLoader workers for the per-segment neural reads (env NEURO_LOAD_WORKERS).

    The per-segment fMRI/MEG reads are I/O-bound, so parallel workers cut the load
    by ~10x (e.g. the NSD subj01 load drops from ~25 min to a few min). Defaults to
    16; override via NEURO_LOAD_WORKERS.
    """
    return int(os.environ.get("NEURO_LOAD_WORKERS", "16"))


# --- on-disk split cache --------------------------------------------------------
# The (X, Y) split for a dataset is the slow thing to build (minutes of per-segment
# neural reads). It is a pure function of the dataset's identity, so the first
# load_split computes and saves it; later calls load it from disk in seconds, and
# every AlphaEvolve proposal rerun reuses it instead of repaying the load.
_CACHE_VERSION = "v1"


def _cache_dir() -> str:
    return os.environ.get(
        "NEURO_SPLIT_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "neuro-autoresearch", "splits"),
    )


def _cache_key(d: Dataset) -> str:
    """A hash over ALL Dataset fields, so changing any field invalidates the cache."""
    payload = json.dumps(dataclasses.asdict(d), sort_keys=True, default=str)
    digest = hashlib.sha256(f"{_CACHE_VERSION}:{payload}".encode()).hexdigest()[:16]
    return f"{d.study}_{digest}.pt"


def _cache_load(path: str):
    blob = torch.load(path)
    return blob["X_train"], blob["X_test"], blob["Y_train"], blob["Y_test"]


def _cache_save(path: str, split) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X_train, X_test, Y_train, Y_test = split
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(
        {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test},
        tmp,
    )
    os.replace(tmp, path)


# --- segmentation ---------------------------------------------------------------
def _trigger_key(seg) -> object:
    """Stable identity of a segment's trigger, shared across separate segmenters.

    Two segmenters run on the SAME events (and thus the same triggers); the trigger
    DataFrame index ``_index`` is the natural shared key. Fall back to
    (timeline, onset) when an index is unavailable.
    """
    t = seg.trigger
    idx = getattr(t, "_index", None)
    return idx if idx is not None else (seg.timeline, round(t.start, 6))


def _segment_batch(d: Dataset, events, extractors: dict, start: float, duration: float):
    seg = ns.Segmenter(
        extractors=extractors,
        trigger_query=d.trigger, start=start, duration=duration,
        drop_incomplete=True,
    )
    return seg.apply(events).load_all(num_workers=_load_workers())


def _stim_window(d: Dataset) -> tuple[float, float]:
    """The stimulus (start, duration); defaults to the neural window."""
    start = d.start if d.stim_start is None else d.stim_start
    duration = d.duration if d.stim_duration is None else d.stim_duration
    return start, duration


def _build_xy(d: Dataset, events):
    """Build row-aligned (X, Y, onsets): one row per trigger, onsets the sort key.

    Fast path: when the stimulus window equals the neural window, a single
    Segmenter extracts both, already row-aligned. Decoupled path (e.g. audio on
    HRF-delayed fMRI): the stimulus and neural are segmented separately and aligned
    by trigger -- ``drop_incomplete`` may drop different triggers on each side, so
    the result is the intersection, ordered by trigger onset.
    """
    stim_start, stim_duration = _stim_window(d)
    stim_ext, neural_ext = _stim_extractor(d), _NEURAL[d.neural]()

    if (stim_start, stim_duration) == (d.start, d.duration):
        batch = _segment_batch(
            d, events, dict(stim=stim_ext, neural=neural_ext), d.start, d.duration
        )
        X = batch.data["stim"].float()
        Y = batch.data["neural"].float().reshape(len(batch.segments), -1)
        onsets = np.array([s.start for s in batch.segments])
        return X, Y, onsets

    neural_batch = _segment_batch(d, events, dict(neural=neural_ext), d.start, d.duration)
    stim_batch = _segment_batch(d, events, dict(stim=stim_ext), stim_start, stim_duration)

    stim_row = {_trigger_key(s): i for i, s in enumerate(stim_batch.segments)}
    paired = [
        (i, stim_row[k])
        for i, s in enumerate(neural_batch.segments)
        if (k := _trigger_key(s)) in stim_row
    ]
    if not paired:
        raise RuntimeError("no triggers shared between the stimulus and neural windows")
    n_idx = [i for i, _ in paired]
    s_idx = [j for _, j in paired]

    X = stim_batch.data["stim"].float()[s_idx]
    Y = neural_batch.data["neural"].float().reshape(len(neural_batch.segments), -1)[n_idx]
    onsets = np.array([neural_batch.segments[i].trigger.start for i in n_idx])
    return X, Y, onsets


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test) as 2-D float tensors.

    The split is a contiguous temporal hold-out (latest ``test_frac`` of the
    triggers by onset time), not a random shuffle: for time-continuous neural
    signals (fMRI/MEG/EEG) nearby triggers share samples, so a random split
    would leak near-duplicate rows across train and test.

    The result is cached on disk (keyed by the full dataset identity) so repeated
    runs load it in seconds; bypass with NEURO_NO_CACHE, relocate with
    NEURO_SPLIT_CACHE.
    """
    use_cache = not os.environ.get("NEURO_NO_CACHE")
    cache_path = os.path.join(_cache_dir(), _cache_key(d))
    if use_cache and os.path.exists(cache_path):
        return _cache_load(cache_path)

    path = d.path if d.path is not None else (None if d.download else ns.CACHE_FOLDER)
    study = ns.Study(name=d.study, path=path)
    if d.download:
        study.download()
    try:
        events = study.run()
    except FileNotFoundError:           # cold demo cache: self-provision once
        study.download()
        events = study.run()

    X, Y, onsets = _build_xy(d, events)

    # Contiguous temporal split by trigger onset.
    order = torch.as_tensor(np.argsort(onsets, kind="stable"), dtype=torch.long)
    n = order.numel()
    n_test = max(1, int(round(n * d.test_frac)))
    tr, te = order[: n - n_test], order[n - n_test:]
    X_train, X_test, Y_train, Y_test = X[tr], X[te], Y[tr], Y[te]

    # Drop targets that are constant on the TRAIN set: they carry no signal, are
    # unpredictable, and (e.g. non-brain fMRI voxels) only dilute the mean-r and
    # waste compute. The mask uses train variance only (no test peeking).
    keep = Y_train.std(0) > 0
    split = (X_train, X_test, Y_train[:, keep], Y_test[:, keep])

    if use_cache:
        _cache_save(cache_path, split)
    return split
