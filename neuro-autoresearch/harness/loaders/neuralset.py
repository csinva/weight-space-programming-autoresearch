"""NeuralSet / Segmenter timeline loaders: nsd, text, meg, ieeg.

Each ``Dataset`` declares a ``stimulus`` modality and a ``neural`` modality; this path
dispatches the right NeuralSet extractor for each, aligns them per segment, and returns a
train/test split. Datasets whose stimuli repeat (e.g. NSD) get one sample per unique
stimulus (repeats averaged) and an image-disjoint canonical held-out test; others keep a
contiguous temporal split.

Stimulus extractors are OFFLINE-friendly (no model-weight downloads):
  * image -> RawPixels (flattened RGB)
  * text  -> WordLength + WordFrequency (the trigger word's [length, zipf-freq])
  * audio -> MelSpectrum (time-averaged Mel bands)
"""
from __future__ import annotations

import os

import numpy as np
import torch
import neuralset as ns

from harness.datasets import Dataset
from harness import studies  # noqa: F401  (registers NsdSubj01 / Li2022PetitSubj / ...)
from harness.loaders.core import (
    RawPixels, _study_path, _load_workers, _cache_dir, _build_cache_key,
    _build_cache_load, _build_cache_save, _apply_roi, _collapse_repeats,
    _image_id, _canonical_test_ids,
)


_NEURAL = {
    # Bel2026 raw .fif is Elekta MaxShield -> allow_maxshield or it raises.
    "meg": lambda d: ns.extractors.MegExtractor(frequency=100.0, allow_maxshield=True),
    # iEEG/ECoG: like MEG, a fast electrophysiology signal. IeegExtractor picks the
    # ECOG/SEEG channels (dropping ECG/MISC) and read_raw_bids marks bads from channels.tsv.
    "ieeg": lambda d: ns.extractors.IeegExtractor(frequency=100.0),
    # fMRI: shift the window forward by `neural_offset` to align the delayed (HRF)
    # BOLD response and time-aggregate it (`neural_aggregation`); the default cleaning
    # (z-score + detrend) is applied by FmriExtractor itself.
    "fmri": lambda d: ns.extractors.FmriExtractor(
        aggregation=d.neural_aggregation, offset=d.neural_offset
    ),
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


def _seg_stim_ids(segments):
    """Per-segment stimulus identity from the trigger's ``filepath`` (or ``None``).

    Datasets whose stimuli repeat (e.g. NSD: each image is shown ~3x) expose a
    per-trigger stimulus id; the loader uses it to collapse repeats and to build an
    image-disjoint split. Modalities without a per-stimulus file (text/audio word
    triggers) have no ``filepath`` -> returns ``None`` and the loader keeps its
    contiguous-temporal split unchanged.
    """
    ids = [getattr(s.trigger, "filepath", None) for s in segments]
    return np.array(ids, dtype=object) if any(i is not None for i in ids) else None


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


def _neural_2d(neural, d) -> torch.Tensor:
    """Per-segment neural tensor -> 2-D (n_segments, n_targets).

    With ``neural_aggregation == "mean"`` the per-segment time (TR) axis is averaged
    into ONE value per channel -- a denoised, betas-like response that lifts the
    encoding score markedly (for NSD it takes a trained AlexNet from ~0.28 to ~0.32).
    ``"single"`` keeps every TR as a separate target (the default; unchanged).
    """
    neural = neural.float()
    if d.neural_aggregation == "mean" and neural.dim() >= 3:
        neural = neural.mean(dim=-1)        # collapse the trailing TR/time axis
    return neural.reshape(neural.shape[0], -1)


def _build_xy(d: Dataset, events):
    """Build row-aligned (X, Y, onsets, stim_ids): one row per trigger.

    ``onsets`` is the per-row trigger time (the temporal-split sort key); ``stim_ids``
    is the per-row stimulus identity (``None`` when the modality has no per-stimulus
    file). Fast path: when the stimulus window equals the neural window, a single
    Segmenter extracts both, already row-aligned. Decoupled path (e.g. audio on
    HRF-delayed fMRI): the stimulus and neural are segmented separately and aligned
    by trigger -- ``drop_incomplete`` may drop different triggers on each side, so
    the result is the intersection, ordered by trigger onset.
    """
    stim_start, stim_duration = _stim_window(d)
    stim_ext, neural_ext = _stim_extractor(d), _NEURAL[d.neural](d)

    if (stim_start, stim_duration) == (d.start, d.duration):
        batch = _segment_batch(
            d, events, dict(stim=stim_ext, neural=neural_ext), d.start, d.duration
        )
        X = batch.data["stim"].float()
        Y = _neural_2d(batch.data["neural"], d)
        onsets = np.array([s.start for s in batch.segments])
        return X, Y, onsets, _seg_stim_ids(batch.segments)

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
    Y = _neural_2d(neural_batch.data["neural"], d)[n_idx]
    onsets = np.array([neural_batch.segments[i].trigger.start for i in n_idx])
    stim_ids = _seg_stim_ids([neural_batch.segments[i] for i in n_idx])
    return X, Y, onsets, stim_ids


def load_neuralset(d: Dataset):
    """Timeline path: build aligned (X, Y), collapse repeats, split. 2-D float tensors.

    Two split regimes, chosen by the data itself:

    * **Repeated stimuli with a canonical held-out set** (e.g. NSD: each image is
      shown ~3x and a fixed shared-1000 subset is the standard test). The repeats of
      each unique stimulus are collapsed to one sample (the neural target is averaged;
      the identical stimulus is unchanged), and the held-out test is the canonical
      shared set -- an **image-disjoint** split (asserted), no leakage.
    * **Otherwise** a contiguous temporal hold-out (latest ``test_frac`` of the
      triggers by onset): for time-continuous neural signals (fMRI/MEG/EEG) nearby
      triggers share samples, so a random split would leak near-duplicate rows.

    For volumetric studies the dataset's ``roi`` further column-subsets the neural
    target to a brain region (a fast subset of the already-loaded voxels).

    The slow ``_build_xy`` output is cached on disk keyed by every field except
    ``roi``/``test_frac`` (see ``_build_cache_key``), so a different ROI re-uses it and
    just re-subsets/re-splits in seconds -- never re-reading the timeseries. Bypass with
    NEURO_NO_CACHE, relocate with NEURO_SPLIT_CACHE.
    """
    use_cache = not os.environ.get("NEURO_NO_CACHE")
    build_path = os.path.join(_cache_dir(), _build_cache_key(d))

    study = ns.Study(name=d.study, path=_study_path(d))

    if use_cache and os.path.exists(build_path):
        X, Y, onsets, stim_ids = _build_cache_load(build_path)   # full voxels, all trials
    else:
        if d.download:
            study.download()
        try:
            events = study.run()
        except FileNotFoundError:           # cold demo cache: self-provision once
            study.download()
            events = study.run()
        X, Y, onsets, stim_ids = _build_xy(d, events)
        if use_cache:
            _build_cache_save(build_path, X, Y, onsets, stim_ids)

    n_trials = X.shape[0]
    Y = _apply_roi(d, study, Y)         # column-subset the neural target to d.roi

    # Collapse repeats of the same stimulus to one sample (no-op without repeats).
    repeated = stim_ids is not None and len(np.unique(stim_ids)) < len(stim_ids)
    if repeated:
        X, Y, onsets, stim_ids = _collapse_repeats(X, Y, onsets, stim_ids)
    n_unique = X.shape[0]

    canonical = stim_ids is not None and hasattr(study, "_get_test_image_ids")
    if canonical:                       # image-disjoint shared-1000 held-out test
        shared = _canonical_test_ids(study)
        is_test = np.array([_image_id(s) in shared for s in stim_ids])
        tr_idx, te_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
        tr = torch.as_tensor(tr_idx, dtype=torch.long)
        te = torch.as_tensor(te_idx, dtype=torch.long)
        train_ids = set(stim_ids[tr_idx].tolist())
        test_ids = set(stim_ids[te_idx].tolist())
        assert train_ids.isdisjoint(test_ids), "train/test share images -> leakage"
        assert te.numel() and tr.numel(), "empty train or test split"
    else:                               # contiguous temporal split by trigger onset
        order = torch.as_tensor(np.argsort(onsets, kind="stable"), dtype=torch.long)
        n = order.numel()
        n_test = max(1, int(round(n * d.test_frac)))
        tr, te = order[: n - n_test], order[n - n_test:]
    X_train, X_test, Y_train, Y_test = X[tr], X[te], Y[tr], Y[te]

    regime = "shared-1000 held-out" if canonical else "temporal hold-out"
    roi_note = f"; roi={d.roi}" if d.roi != "nsdgeneral" else ""
    print(f"  {n_trials} trials -> {n_unique} unique stimuli; "
          f"train/test = {X_train.shape[0]}/{X_test.shape[0]} samples ({regime}{roi_note}); "
          f"{Y_train.shape[1]} target dims")

    # Drop targets that are constant on the TRAIN set: they carry no signal, are
    # unpredictable, and (e.g. non-brain fMRI voxels) only dilute the mean-r and
    # waste compute. The mask uses train variance only (no test peeking).
    keep = Y_train.std(0) > 0
    return (X_train, X_test, Y_train[:, keep], Y_test[:, keep])
