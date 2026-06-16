"""The dataset this proof-of-concept runs on, selected WITHOUT editing files.

A dataset declares both its **stimulus modality** (``stimulus``: image | text |
audio) and its **neural modality** (``neural``: meg | eeg | fmri); the harness
(``data.py``) dispatches the right NeuralSet extractor for each. The model and
the reward stay modality-agnostic, so the same loop runs across datasets and
modalities.

Three real datasets are pre-registered in ``PRESETS``. Pick one at run time via
the ``NEURO_DATASET`` environment variable (default ``"text"``) -- no source edit
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
    neural: str = "meg"              # neural modality: meg | eeg | fmri
    stimulus: str = "image"          # stimulus modality: image | text | audio
    imsize: int = 16                 # image stimuli resized to imsize x imsize
    start: float = 0.0               # neural segment start, seconds after trigger
    duration: float = 0.3            # neural segment duration, seconds
    neural_offset: float = 0.0       # seconds to shift the neural window forward to
                                     # align the delayed (HRF) BOLD response (fMRI)
    neural_aggregation: str = "single"  # neural time-aggregation: "single" (keep TRs)
                                     # or "mean" (one value per channel/voxel)
    stim_start: float | None = None  # stimulus window start; None => reuse `start`
    stim_duration: float | None = None  # stimulus window duration; None => reuse `duration`
    download: bool = False           # download the study before running?
    path: str | None = None          # pre-downloaded data dir; None => shared cache
    test_frac: float = 0.2           # held-out fraction (contiguous temporal tail)
    roi: str = "nsdgeneral"          # neural-target region (volumetric studies, e.g. NSD):
                                     # "nsdgeneral" (default = all loaded voxels) or a
                                     # sub-ROI from the fixed menu (V1/hV4/FFA/PPA/EBA/
                                     # VWFA/early/ventral), selected as a fast column-
                                     # subset of the already-loaded voxels.
    reward_roi: str = "nsdgeneral"   # region the scalar reward (test_corr) is computed
                                     # over. The readout still fits ALL voxels and every
                                     # region's predictivity is still logged (corr_<name>);
                                     # only the headline reward narrows to this region.
                                     # "nsdgeneral" = whole target set (the default for
                                     # datasets without ROIs, e.g. text/meg). For NSD pick
                                     # a famous region or visual stream: V1/hV4/FFA/PPA/
                                     # EBA/VWFA or early/ventral. Choose per run with
                                     # NEURO_ROI (e.g. NEURO_ROI=PPA) so parallel
                                     # per-region loops never collide -- each writes its
                                     # own results CSV.
    run_tag: str = ""                # optional per-run id (set by NEURO_RUN). When set,
                                     # the results file is suffixed with it so each run
                                     # gets its OWN unique CSV (history + best_so_far are
                                     # not shared across runs). Empty => shared per-region.
    loader: str = "segmenter"        # data path: "segmenter" (continuous timeline ->
                                     # ns.Segmenter, the default for fMRI/MEG/EEG) or "table"
                                     # (a pre-averaged [electrodes x stimuli] .mat already
                                     # aligned to its stimuli, e.g. TVSD macaque spiking).
    subject: str = ""                # which animal/subject for a multi-subject table dataset
                                     # (TVSD: "F" or "N"); unused by the segmenter path.


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
    # The neural target is the HRF-PEAK response: with TR=1.333s the BOLD response to
    # the image peaks ~5s after onset, so the FmriExtractor window is shifted forward
    # by neural_offset=5.0s and the peak TRs in a 4s window are averaged into ONE
    # value per voxel (neural_aggregation="mean"). (offset=0/"single" over 0-3s samples
    # pre-response baseline -- even a trained AlexNet then scores ~0.006; this
    # HRF-aligned, TR-averaged response lifts that to ~0.32, with ventral-stream ~0.28.)
    # `roi` picks the neural-target region as a fast column-subset of the loaded
    # nsdgeneral voxels (default "nsdgeneral" = all; e.g. "V1" or "FFA" for fewer).
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
    # The "Podcast" ECoG dataset (OpenNeuro ds005574, Zada et al. / Hasson lab), iEEG +
    # audio stimulus -- 9 epilepsy patients with 1,330 intracranial ECoG electrodes total
    # listened to a 30-minute spoken podcast (>5,000 words). Single-subject by necessity
    # (each patient has different electrodes), via the Zada2025Podcast study (default
    # subject 01). Each transcript Word triggers a 0.5 s iEEG epoch; the stimulus is the
    # Mel spectrum of the podcast audio in that window. iEEG is near-instant, so the
    # stimulus and neural windows coincide ([0, 0.5] s) -- no HRF decoupling (as for meg).
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
    # TVSD -- THINGS Ventral-stream Spiking Dataset (Papale et al., Neuron 2025), image +
    # macaque spiking. NOT a NeuralSet study: the release ships a PRE-AVERAGED normalized-MUA
    # table (`THINGS_normMUA.mat`, 1024 electrodes x stimuli), so it uses the `loader="table"`
    # path (`data._load_tvsd`) instead of the Segmenter. Stimulus = the THINGS image shown
    # (RawPixels, like NSD); target = each electrode's time-windowed normalized MUA. The
    # train/test split is the dataset's own (22248 train images / 100 repeated test images,
    # image-disjoint by construction). `reward_roi` picks the cortical area V1/V4/IT (one
    # area = a column-subset of the 1024 electrodes, labels hardcoded per monkey from the
    # release's norm_MUA.m); the readout fits and scores that area's electrodes. Monkey F vs
    # N is chosen with `subject` (NEURO_SUBJECT). Stimuli are the gated THINGS images (not
    # bundled): stage them once (see data._tvsd_images_root) before running.
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

# The reward region, chosen by NEURO_ROI (default = the preset's reward_roi). Lets
# you run one auto-research loop per brain region without editing any source.
_roi = os.environ.get("NEURO_ROI")
if _roi:
    DATASET = replace(DATASET, reward_roi=_roi)

# Optional per-run id (NEURO_ROI-style), so each run writes its own unique results CSV.
_run = os.environ.get("NEURO_RUN")
if _run:
    DATASET = replace(DATASET, run_tag=_run)

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
