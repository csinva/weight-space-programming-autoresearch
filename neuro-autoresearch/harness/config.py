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
    start: float = 0.0               # neural segment start, seconds after trigger
    duration: float = 0.3            # neural segment duration, seconds
    stim_start: float | None = None  # stimulus window start; None => reuse `start`
    stim_duration: float | None = None  # stimulus window duration; None => reuse `duration`
    download: bool = False           # download the study before running?
    path: str | None = None          # pre-downloaded data dir; None => shared cache
    test_frac: float = 0.2           # held-out fraction (contiguous temporal tail)
    roi: str = "nsdgeneral"          # neural-target region (volumetric studies, e.g. NSD):
                                     # "nsdgeneral" (default = all loaded voxels) or a
                                     # sub-ROI ("V1"/"V2"/"V3"/"hV4"/"FFA"), selected as a
                                     # fast column-subset of the already-loaded voxels.


# Three real datasets spanning both neural modalities (fMRI + MEG) and stimulus
# modalities (image + text). Select one with NEURO_DATASET=<key>.
PRESETS: dict[str, Dataset] = {
    # STORY (Le Petit Prince), fMRI + audio stimulus -- one real full subject of the
    # Li2022Petit study (English EN057, all 9 story sections) via the Li2022PetitSubj
    # subclass in harness/li2022_subj.py. Each spoken Word triggers a segment; the
    # stimulus is the Mel spectrum of the audio AT THE WORD and the target is the
    # fMRI volume. The neural window start=2.0/duration=4.0 captures the delayed
    # (HRF) BOLD response, but the audio response is near-instant, so the stimulus
    # is read from a short window AT THE WORD (stim_start=0.0/stim_duration=0.5) --
    # decoupled from the HRF-delayed neural window (see data.py).
    "text": Dataset(
        study="Li2022PetitSubj",
        trigger="type=='Word'",
        neural="fmri",
        stimulus="audio",
        start=2.0,
        duration=4.0,
        stim_start=0.0,
        stim_duration=0.5,
        download=False,
        path="/home/t-ymehta/Research/neuralset/data/li2022petit_en057",
    ),
    # Natural Scenes Dataset (NSD), fMRI + image stimulus, restricted to subject 1
    # (the NsdSubj01 subclass in harness/nsd_subj01.py). Each shown Image triggers a
    # segment; the stimulus is the resized image and the target is the fMRI volume.
    # Subject 1 saw 10,000 unique images, each repeated ~3x (30,000 trials): the
    # harness collapses the repeats to one sample per unique image and holds out the
    # canonical NSD shared-1000 images as an image-disjoint test set (see data.py).
    # `roi` picks the neural-target region as a fast column-subset of the loaded
    # nsdgeneral voxels (default "nsdgeneral" = all; e.g. "V1" or "FFA" for fewer).
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
        roi="nsdgeneral",
    ),
    # Le Petit Prince MEG (LPP-Listen, OpenNeuro ds007523), MEG + audio stimulus --
    # the loop's first electrophysiological modality. The raw .fif is Elekta
    # MaxShield (handled in data.py via allow_maxshield=True). Each spoken Word
    # triggers a 0.5 s MEG epoch; the stimulus is the Mel spectrum of the audio in
    # that same window. MEG is near-instant, so the stimulus and neural windows
    # coincide ([0, 0.5] s) -- no decoupling needed.
    "meg": Dataset(
        study="Bel2026PetitListenSample",
        trigger="type=='Word'",
        neural="meg",
        stimulus="audio",
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
