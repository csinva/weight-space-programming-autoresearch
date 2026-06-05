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

import numpy as np
import torch
import neuralset as ns
from neuralset.extractors.image import BaseClassicImageExtractor

from config import Dataset


class RawPixels(BaseClassicImageExtractor):
    """Flattened, [0,1]-normalised RGB pixels of each stimulus image."""

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        return (image.astype("float32") / 255.0).reshape(-1)


_NEURAL = {
    "meg": lambda: ns.extractors.MegExtractor(frequency=100.0),
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


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test) as 2-D float tensors.

    The split is a contiguous temporal hold-out (latest ``test_frac`` of the
    triggers by onset time), not a random shuffle: for time-continuous neural
    signals (fMRI/MEG/EEG) nearby triggers share samples, so a random split
    would leak near-duplicate rows across train and test.
    """
    path = d.path if d.path is not None else (None if d.download else ns.CACHE_FOLDER)
    study = ns.Study(name=d.study, path=path)
    if d.download:
        study.download()
    try:
        events = study.run()
    except FileNotFoundError:           # cold demo cache: self-provision once
        study.download()
        events = study.run()

    seg = ns.Segmenter(
        extractors=dict(stim=_stim_extractor(d), neural=_NEURAL[d.neural]()),
        trigger_query=d.trigger, start=d.start, duration=d.duration,
        drop_incomplete=True,
    )
    batch = seg.apply(events).load_all()
    X = batch.data["stim"].float()
    Y = batch.data["neural"].float().reshape(batch.data["neural"].shape[0], -1)

    # Contiguous temporal split by trigger onset.
    onsets = np.array([s.start for s in batch.segments])
    order = torch.as_tensor(np.argsort(onsets, kind="stable"), dtype=torch.long)
    n = order.numel()
    n_test = max(1, int(round(n * d.test_frac)))
    tr, te = order[: n - n_test], order[n - n_test:]
    X_train, X_test, Y_train, Y_test = X[tr], X[te], Y[tr], Y[te]

    # Drop targets that are constant on the TRAIN set: they carry no signal, are
    # unpredictable, and (e.g. non-brain fMRI voxels) only dilute the mean-r and
    # waste compute. The mask uses train variance only (no test peeking).
    keep = Y_train.std(0) > 0
    return X_train, X_test, Y_train[:, keep], Y_test[:, keep]
