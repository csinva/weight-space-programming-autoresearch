"""The dataset this proof-of-concept runs on, selected WITHOUT editing files.

A dataset declares both its **stimulus modality** (``stimulus``: image | text |
audio) and its **neural modality** (``neural``: meg | eeg | fmri); the harness
(``data.py``) dispatches the right NeuralSet extractor for each. The model and
the reward stay modality-agnostic, so the same loop runs across datasets and
modalities.

Three real datasets are pre-registered in ``PRESETS``. Pick one at run time via
the ``NEURO_DATASET`` environment variable (default ``"text"``) -- no source edit
needed, so parallel runs never collide::

    NEURO_DATASET=meg python evolve/model.py

For another study, name one from ``neuralset.Study.catalog()``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    study: str = "Fake2025Meg"       # any name in neuralset.Study.catalog()
    trigger: str = "type=='Image'"   # which events are the stimuli
    neural: str = "meg"              # neural modality: meg | eeg | fmri
    stimulus: str = "image"          # stimulus modality: image | text | audio
    imsize: int = 16                 # image stimuli resized to imsize x imsize
    start: float = 0.0               # segment start, seconds after trigger
    duration: float = 0.3            # segment duration, seconds
    download: bool = False           # download the study before running?
    path: str | None = None          # pre-downloaded data dir; None => shared cache
    test_frac: float = 0.2           # held-out fraction (contiguous temporal tail)


# Three real datasets spanning both neural modalities (fMRI + MEG) and stimulus
# modalities (image + text). Select one with NEURO_DATASET=<key>.
PRESETS: dict[str, Dataset] = {
    # STORY (Le Petit Prince), fMRI + word stimulus -- the FULL Li2022Petit study.
    # Each spoken Word triggers a segment; the stimulus is the word's
    # [length, frequency] and the target is the fMRI volume. start=2.0/duration=4.0
    # captures the delayed (HRF) BOLD response.
    "text": Dataset(
        study="Li2022Petit",
        trigger="type=='Word'",
        neural="fmri",
        stimulus="text",
        start=2.0,
        duration=4.0,
        download=False,
        path="/home/t-ymehta/Research/neuralset/data/li2022petit_full",
    ),
    # Natural Scenes Dataset (NSD), fMRI + image stimulus, restricted to subject 1
    # (the NsdSubj01 subclass in harness/nsd_subj01.py). Each shown Image triggers a
    # segment; the stimulus is the resized image and the target is the fMRI volume.
    "nsd": Dataset(
        study="NsdSubj01",
        trigger="type=='Image'",
        neural="fmri",
        stimulus="image",
        imsize=16,
        start=0.0,
        duration=3.0,
        download=False,
        path="/home/t-ymehta/Research/neuralset/data/nsd_subj01_full",
    ),
    # Le Petit Prince MEG (LPP-Listen, OpenNeuro ds007523), MEG + word stimulus --
    # the loop's first electrophysiological modality. The raw .fif is Elekta
    # MaxShield (handled in data.py via allow_maxshield=True). Each spoken Word
    # triggers a 0.5 s MEG epoch.
    "meg": Dataset(
        study="Bel2026PetitListenSample",
        trigger="type=='Word'",
        neural="meg",
        stimulus="text",
        start=0.0,
        duration=0.5,
        download=False,
        path="/home/t-ymehta/Research/neuralset/data/bel2026petitlisten_sample",
    ),
}

_key = os.environ.get("NEURO_DATASET", "text")
if _key not in PRESETS:
    raise ValueError(
        f"NEURO_DATASET={_key!r} is not a known dataset. "
        f"Valid keys: {sorted(PRESETS)}"
    )

# The one active dataset, chosen by NEURO_DATASET (default "text").
DATASET = PRESETS[_key]
