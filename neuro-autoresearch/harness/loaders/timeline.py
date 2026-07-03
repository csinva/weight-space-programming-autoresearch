"""NeuralSet / Segmenter timeline loader: text, meg, ieeg.

Each ``Dataset`` declares a ``stimulus`` modality and a ``neural`` modality; this path
dispatches the right NeuralSet extractor for each, aligns them per segment, and returns a
train/test split. The split is always a contiguous temporal hold-out (the latest
``test_frac`` of triggers by onset): for time-continuous neural signals (fMRI/MEG/iEEG)
nearby triggers share samples, so a random split would leak near-duplicate rows.

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
import pandas as pd
import torch
import neuralset as ns

from harness.datasets import Dataset
from harness.loaders import studies  # noqa: F401  (registers Li2022PetitSubj / Zada2025Podcast)
from harness.loaders.shared import RawPixels, _study_path, _cache_dir, drop_constant_cols, processed_dir


# Studies whose continuous neural signal has a slim, raw-free data/<ds>/ tree
# (written by scripts/process_dataset/audio_neural.py). Maps the study name to the
# processed-dir key; when that dir exists the loader rebuilds events from it instead
# of touching the raw .fif/edf/BIDS. fMRI timelines (text) are not processed here.
_PROCESSED_KEY = {"Bel2026PetitListenSample": "meg", "Zada2025Podcast": "ieeg"}

# Text (Le Petit Prince fMRI). Its volumes don't reduce to a re-segmentable continuous .fif the way
# meg/ieeg do, so instead of a slim tree it ships a BAKED final (X, Y) blob per study in data/text/
# (windows fixed at build time). Runtime loads that blob directly; the build step is the only raw reader.
_TEXT_PREFIX = "Li2022PetitSubj"


def _load_text_blob(d: Dataset):
    """Load the baked (X, Y, onsets, stim_ids) for a text fMRI study from data/text/<study>.pt."""
    blob_path = os.path.join(processed_dir("text"), f"{d.study}.pt")
    if not os.path.exists(blob_path):
        suffix = d.study[len(_TEXT_PREFIX):]
        raise FileNotFoundError(
            f"{d.study} expects the in-repo blob {blob_path!r}; data/text/ is the only source -- "
            f"build it once with `python -m scripts.process_dataset.text"
            f"{(' --subject ' + suffix) if suffix else ''}`.")
    return _build_cache_load(blob_path)


def _processed_audio_events(out: str):
    """Rebuild the NeuralSet events DataFrame from a slim data/<ds>/ tree (audio neural).

    Mirrors what the raw study.run() returns for the fields _build_xy uses: one neural
    row (pointing at the preprocessed 100Hz .fif), one Audio row (the copied wav), and one
    Word row per onset. The neural extractor re-picks/re-resamples to a no-op, and the Mel
    is recomputed at load -- so window + Mel params stay flexible and (X, Y) match the raw path.
    """
    from neuralset.events.utils import standardize_events

    meta = json.load(open(os.path.join(out, "neural", "meta.json")))
    type_for = {"meg": "Meg", "ieeg": "Ieeg"}[meta["modality"]]
    fif = os.path.join(out, "neural", f"continuous_{meta['modality']}.fif")
    timeline, subject = "t0", meta["subject"]

    neural_row = {
        "type": type_for, "start": meta["start"], "frequency": meta["frequency"],
        "filepath": fif, "subject": subject, "timeline": timeline,
    }
    audio_row = {
        "type": "Audio", "start": meta["audio_start"], "duration": meta["audio_duration"],
        "timeline": timeline, "filepath": os.path.join(out, "stimuli", "audio.wav"),
    }
    words = pd.read_csv(os.path.join(out, "stimuli", "words.csv"))
    words["type"] = "Word"
    words["timeline"] = timeline
    df = pd.concat([pd.DataFrame([neural_row, audio_row]), words], ignore_index=True)
    return standardize_events(df)


def _load_workers() -> int:
    """DataLoader workers for the per-segment neural reads (env NEURO_LOAD_WORKERS).

    The per-segment fMRI/MEG reads are I/O-bound, so parallel workers cut the load
    by ~10x (e.g. the NSD subj01 load drops from ~25 min to a few min). Defaults to
    16; override via NEURO_LOAD_WORKERS.
    """
    return int(os.environ.get("NEURO_LOAD_WORKERS", "16"))


# --- on-disk build cache --------------------------------------------------------
# The slow thing is _build_xy: minutes of per-segment neural reads to assemble the
# full (X, Y) over ALL loaded voxels. That output is independent of the cheap,
# downstream choices -- the target region (``roi``, a column subset) and the split
# fraction -- so it is cached keyed by every field EXCEPT those. The first call
# builds and saves it; later calls (including a DIFFERENT roi) load it from disk in
# seconds and just re-subset/re-split, never re-reading the timeseries.
_CACHE_VERSION = "v2"

# Fields that only affect post-build (cheap) steps, so they are excluded from the
# build-cache key: ``test_frac`` only sizes a temporal tail; ``reward_roi`` only
# selects which targets the scalar reward averages over. All region loops therefore
# share ONE build (the timeseries is read only once).
_BUILD_AGNOSTIC_FIELDS = ("test_frac", "reward_roi", "cv_folds")


def _build_cache_key(d: Dataset, processed: bool = False) -> str:
    """A hash over the Dataset fields that affect _build_xy (all but roi/test_frac).

    The raw study tree and the slim ``data/<ds>/`` processed tree produce matching
    (X, Y) by design but are independent sources, so they get DISTINCT cache paths
    (``source`` in the key). Otherwise a stale raw-built cache would permanently
    shadow a freshly-generated processed tree (the cache check precedes the
    processed branch); keying by source lets each rebuild once and reuse thereafter.
    """
    fields = {k: v for k, v in dataclasses.asdict(d).items()
              if k not in _BUILD_AGNOSTIC_FIELDS}
    fields["__source__"] = "processed" if processed else "raw"
    payload = json.dumps(fields, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{_CACHE_VERSION}:{payload}".encode()).hexdigest()[:16]
    return f"{d.study}_build_{digest}.pt"


def _build_cache_load(path: str):
    blob = torch.load(path, weights_only=False)
    return blob["X"], blob["Y"], blob["onsets"], blob["stim_ids"]


def _build_cache_save(path: str, X, Y, onsets, stim_ids) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save({"X": X, "Y": Y, "onsets": onsets, "stim_ids": stim_ids}, tmp)
    os.replace(tmp, path)


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
        aggregation=d.timeline.neural_aggregation, offset=d.timeline.neural_offset
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
        trigger_query=d.timeline.trigger, start=start, duration=duration,
        drop_incomplete=True,
    )
    return seg.apply(events).load_all(num_workers=_load_workers())


def _stim_window(d: Dataset) -> tuple[float, float]:
    """The stimulus (start, duration); defaults to the neural window."""
    start = d.timeline.start if d.timeline.stim_start is None else d.timeline.stim_start
    duration = d.timeline.duration if d.timeline.stim_duration is None else d.timeline.stim_duration
    return start, duration


def _neural_2d(neural, d) -> torch.Tensor:
    """Per-segment neural tensor -> 2-D (n_segments, n_targets).

    With ``neural_aggregation == "mean"`` the per-segment time (TR) axis is averaged
    into ONE value per channel -- a denoised, betas-like response that lifts the
    encoding score markedly (for NSD it takes a trained AlexNet from ~0.28 to ~0.32).
    ``"single"`` keeps every TR as a separate target (the default; unchanged).
    """
    neural = neural.float()
    if d.timeline.neural_aggregation == "mean" and neural.dim() >= 3:
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

    if (stim_start, stim_duration) == (d.timeline.start, d.timeline.duration):
        batch = _segment_batch(
            d, events, dict(stim=stim_ext, neural=neural_ext), d.timeline.start, d.timeline.duration
        )
        X = batch.data["stim"].float()
        Y = _neural_2d(batch.data["neural"], d)
        onsets = np.array([s.start for s in batch.segments])
        return X, Y, onsets, _seg_stim_ids(batch.segments)

    neural_batch = _segment_batch(d, events, dict(neural=neural_ext), d.timeline.start, d.timeline.duration)
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


def load_timeline(d: Dataset):
    """Timeline path: build aligned (X, Y) via the Segmenter, take a contiguous temporal
    hold-out (the latest ``test_frac`` of triggers by onset), and drop train-constant target
    columns. Returns (X_train, X_test, Y_train, Y_test) as 2-D float tensors.

    For time-continuous neural signals (fMRI/MEG/iEEG) nearby triggers share samples, so the
    split is temporal (not random) to avoid leaking near-duplicate rows. The slow ``_build_xy``
    output is cached on disk keyed by every field except ``test_frac`` (see ``_build_cache_key``);
    bypass with NEURO_NO_CACHE, relocate with NEURO_SPLIT_CACHE.
    """
    use_cache = not os.environ.get("NEURO_NO_CACHE")

    if d.study.startswith(_TEXT_PREFIX):    # text fMRI: baked in-repo blob, never re-segmented at runtime
        X, Y, onsets, stim_ids = _load_text_blob(d)
    else:
        # Prefer the slim, raw-free data/<ds>/ tree when present (audio neural: meg/ieeg).
        out = processed_dir(_PROCESSED_KEY[d.study]) if d.study in _PROCESSED_KEY else None
        use_processed = out is not None and os.path.isdir(out)

        # Key the build cache by source so a raw-built cache never shadows the processed
        # tree (and vice versa); each source rebuilds at most once.
        build_path = os.path.join(_cache_dir(), _build_cache_key(d, processed=use_processed))

        if use_cache and os.path.exists(build_path):
            X, Y, onsets, stim_ids = _build_cache_load(build_path)
        elif use_processed:
            X, Y, onsets, stim_ids = _build_xy(d, _processed_audio_events(out))
            if use_cache:
                _build_cache_save(build_path, X, Y, onsets, stim_ids)
        else:
            if d.study in _PROCESSED_KEY:   # meg/ieeg: the in-repo data/<ds>/ tree is the only source
                raise FileNotFoundError(
                    f"{d.study} expects the in-repo data/{_PROCESSED_KEY[d.study]}/ tree at {out!r}; "
                    f"run `python -m scripts.process_dataset.audio_neural "
                    f"--dataset {_PROCESSED_KEY[d.study]}`.")
            study = ns.Study(name=d.study, path=_study_path(d))
            if d.download:
                study.download()
            try:
                events = study.run()
            except FileNotFoundError:       # cold demo cache: self-provision once
                study.download()
                events = study.run()
            X, Y, onsets, stim_ids = _build_xy(d, events)
            if use_cache:
                _build_cache_save(build_path, X, Y, onsets, stim_ids)

    n_trials = X.shape[0]

    # Contiguous temporal hold-out: latest test_frac of the triggers by onset.
    order = torch.as_tensor(np.argsort(onsets, kind="stable"), dtype=torch.long)
    n = order.numel()
    n_test = max(1, int(round(n * d.test_frac)))
    tr, te = order[: n - n_test], order[n - n_test:]
    X_train, X_test, Y_train, Y_test = X[tr], X[te], Y[tr], Y[te]

    Y_train, Y_test = drop_constant_cols(Y_train, Y_test)
    print(f"  {n_trials} trials; train/test = {X_train.shape[0]}/{X_test.shape[0]} samples "
          f"(temporal hold-out); {Y_train.shape[1]} target dims")
    return (X_train, X_test, Y_train, Y_test)
