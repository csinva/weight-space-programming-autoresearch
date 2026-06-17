"""The dataset this proof-of-concept runs on, selected WITHOUT editing files.

A dataset declares both its **stimulus modality** (``stimulus``: image | text |
audio) and its **neural modality** (``neural``: fmri | meg | ieeg | spikes); the
harness (``data.py``) dispatches the right NeuralSet extractor for each. The model
and the reward stay modality-agnostic, so the same loop runs across datasets and
modalities.

Presets are pre-registered in ``PRESETS``. Pick one at run time via the
``NEURO_DATASET`` environment variable (default ``"text"``) -- no source edit
needed, so parallel runs never collide::

    NEURO_DATASET=meg python -m harness.run

For another study, name one from ``neuralset.Study.catalog()``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Dataset:
    study: str = "Fake2025Meg"       # any name in neuralset.Study.catalog()
    trigger: str = "type=='Image'"   # which events are the stimuli
    neural: str = "meg"              # neural modality: fmri | meg | ieeg | spikes
    stimulus: str = "image"          # stimulus modality: image | text | audio
    imsize: int = 16                 # image stimuli resized to imsize x imsize
    start: float = 0.0               # neural segment start, seconds after trigger
    duration: float = 0.3            # neural segment duration, seconds
    neural_offset: float = 0.0       # seconds to shift the neural window forward to
                                     # align the delayed (HRF) BOLD response (fMRI)
    neural_aggregation: str = "single"  # time-aggregation: "single" (keep TRs) or
                                     # "mean" (one value per channel/voxel)
    stim_start: float | None = None  # stimulus window start; None => reuse `start`
    stim_duration: float | None = None  # stimulus window duration; None => reuse `duration`
    download: bool = False           # download the study before running?
    path: str | None = None          # pre-downloaded data dir; None => shared cache
    test_frac: float = 0.2           # held-out fraction (contiguous temporal tail)
    roi: str = "nsdgeneral"          # neural-target region as a column-subset of loaded
                                     # voxels: "nsdgeneral" (all) or a sub-ROI from the
                                     # fixed menu (V1/hV4/FFA/PPA/EBA/VWFA/early/ventral).
    reward_roi: str = "nsdgeneral"   # region the scalar reward (test_corr) narrows to;
                                     # readout still fits ALL voxels and logs every region
                                     # (corr_<name>). Set per run via NEURO_ROI. See CLAUDE.md.
    run_tag: str = ""                # per-run id (set by NEURO_RUN): suffixes the results
                                     # CSV so each run gets its own. Empty => shared.
    loader: str = "segmenter"        # "segmenter" (continuous timeline -> ns.Segmenter,
                                     # default for fmri/meg/ieeg) or "table" (pre-averaged
                                     # [electrodes x stimuli] .mat, e.g. TVSD spiking).
    subject: str = ""                # subject/animal for a multi-subject table dataset
                                     # (TVSD: "F" or "N"); unused by the segmenter path.


# Real datasets across neural modalities (fmri/meg/ieeg/spikes) and stimulus
# modalities (image/text/audio). Select one with NEURO_DATASET=<key>.
PRESETS: dict[str, Dataset] = {
    # Li2022Petit (Le Petit Prince), fMRI + audio: one full English subject (EN057)
    # via the Li2022PetitSubj subclass (harness/li2022_subj.py). Each spoken Word
    # triggers a segment. The neural window (start=2.0/duration=4.0) captures the
    # HRF-delayed BOLD, but audio response is near-instant so the stimulus is read
    # from a short decoupled window at the word (stim_start=0.0/stim_duration=0.5).
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
    # NSD (Natural Scenes Dataset), fMRI + image, subject 1 (NsdSubj01 subclass in
    # harness/nsd_subj01.py). Repeats of the 10k unique images collapse to one sample
    # each; the canonical shared-1000 images are held out as an image-disjoint test set.
    # Target = HRF-PEAK response: neural_offset=5.0s shifts the window to the ~5s BOLD
    # peak and the TRs in a 4s window are averaged to one value/voxel ("mean"); this
    # offset is what makes the signal real (offset=0 -> trained AlexNet ~0.006 vs ~0.32).
    # Pick a sub-region per run via NEURO_ROI (reward_roi). See docs/datasets/ & CLAUDE.md.
    "nsd": Dataset(
        study="NsdSubj01",
        trigger="type=='Image'",
        neural="fmri",
        stimulus="image",
        imsize=16,
        start=0.0,
        duration=4.0,
        neural_offset=5.0,
        neural_aggregation="mean",
        download=False,
        path="/home/t-ymehta/Research/neuralset/data/nsd_subj01_full",
        roi="nsdgeneral",
        reward_roi="ventral",
    ),
    # Bel2026PetitListen (LPP-Listen MEG, OpenNeuro ds007523), MEG + audio. Each
    # spoken Word triggers a 0.5s epoch; MEG is near-instant so stimulus and neural
    # windows coincide ([0,0.5]s, no HRF decoupling). GOTCHA: the raw .fif is Elekta
    # MaxShield, loaded in data.py via allow_maxshield=True. See docs/datasets/.
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
    # Zada2025Podcast ("Podcast" ECoG, OpenNeuro ds005574, Hasson lab), iEEG + audio:
    # epilepsy patients listening to a 30-min podcast. Single-subject by necessity
    # (each patient has different electrodes; default subject 01). Each transcript Word
    # triggers a 0.5s epoch; iEEG is near-instant so windows coincide (as for meg).
    "ieeg": Dataset(
        study="Zada2025Podcast",
        trigger="type=='Word'",
        neural="ieeg",
        stimulus="audio",
        start=0.0,
        duration=0.5,
        download=False,
        path="/home/t-ymehta/Research/neuralset/data/ds005574_podcast",
    ),
    # TVSD (THINGS Ventral-stream Spiking Dataset, Papale et al. 2025), image + macaque
    # spiking. NOT a NeuralSet study: ships a PRE-AVERAGED normMUA table (1024 electrodes
    # x stimuli), so loader="table" (data._load_tvsd) with the dataset's own image-disjoint
    # split. reward_roi picks cortical area V1/V4/IT; monkey F/N via NEURO_SUBJECT. GOTCHA:
    # the THINGS stimulus images are gated (not bundled) -- stage them once (see
    # data._tvsd_images_root). See docs/datasets/.
    "tvsd": Dataset(
        study="TvsdMonkeyF",
        loader="table",
        neural="spikes",
        stimulus="image",
        imsize=16,
        subject="F",
        download=False,
        path="/home/t-ymehta/Research/neuralset/data/tvsd_normmua",
        reward_roi="IT",
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

# Trivial single-field overrides: NEURO_ROI picks the reward region (one loop per
# brain region) and NEURO_RUN tags the results CSV (one per run). Both no-source-edit.
for _env, _field in (("NEURO_ROI", "reward_roi"), ("NEURO_RUN", "run_tag")):
    _val = os.environ.get(_env)
    if _val:
        DATASET = replace(DATASET, **{_field: _val})

# Input-resolution override (NEURO_IMSIZE): the operator-chosen pixel size for image
# stimuli (RawPixels imsize). It only enriches the model INPUT X -- the neural target
# Y, the readout, the split and the reward region are untouched -- so the reward stays
# trustworthy; it just stops starving untrained models of resolution (AlexNet reads
# full-res). Changes the build-cache key -> a one-time re-build at the new resolution.
_imsize = os.environ.get("NEURO_IMSIZE")
if _imsize:
    DATASET = replace(DATASET, imsize=int(_imsize))

# Optional subject override (NEURO_SUBJECT). For the table dataset TVSD it picks the
# monkey (F/N); for the `text` dataset (Le Petit Prince fMRI) it picks one of the 49
# English subjects (e.g. EN058), switching to that subject's catalog class and the
# all-subjects staging dir so each subject writes its OWN results CSV. The single-subject
# `text` preset (EN057) is otherwise unchanged.
_subject = os.environ.get("NEURO_SUBJECT")
if _subject:
    if _key == "text":
        _en_all = os.path.join(
            os.path.dirname(DATASET.path), "li2022petit_en_all", "Li2022PetitSubj")
        DATASET = replace(DATASET, study=f"Li2022PetitSubj{_subject}", path=_en_all)
    else:
        DATASET = replace(DATASET, subject=_subject, study=f"TvsdMonkey{_subject.upper()}")
