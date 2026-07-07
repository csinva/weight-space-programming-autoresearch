"""Runtime dataset selection for the fixed evaluation harness.

A ``Dataset`` declares the stimulus + neural modality, split policy, reward region, and which
of two loader paths to take. ``harness.loaders.load_split`` is the single entry point:
  * ``loader="timeline"`` (text/meg/ieeg) -- a continuous NeuralSet study segmented on the fly by
    the ``Segmenter``; its segmentation params live in the dataset's ``timeline=Timeline(...)`` block.
  * ``loader="nsd"|"tvsd"|"nnn"`` (vision) -- a pre-aligned response table read directly by a
    standalone ``loaders.vision`` reader; ``timeline`` is ``None``.
The model and reward stay modality-agnostic, so the same search loop runs across all of them.

Every loader reads ONLY the in-repo ``data/<name>/`` tree at runtime. Presets are registered in
``PRESETS``; pick one via ``NEURO_DATASET`` (default ``"text"``), without editing source::

    NEURO_DATASET=meg python -m harness.run
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Timeline:
    """NeuralSet Segmenter params -- set ONLY for timeline datasets (``loader="timeline"``: text/meg/
    ieeg). Vision table datasets (nsd/tvsd/nnn) leave ``Dataset.timeline=None``."""
    trigger: str = "type=='Word'"       # NeuralSet trigger query (which events segment the timeline)
    start: float = 0.0                  # neural segment start, seconds after the trigger
    duration: float = 0.3               # neural segment duration, seconds
    neural_offset: float = 0.0          # shift the neural window forward to align the HRF BOLD (fMRI)
    neural_aggregation: str = "single"  # time-aggregation: "single" (keep TRs) | "mean" (one/channel)
    stim_start: float | None = None     # stimulus window start; None => reuse `start`
    stim_duration: float | None = None  # stimulus window duration; None => reuse `duration`
    stim_ngram: int = 10                # "chars" stimulus only: preceding words of context per
                                        # n-gram (trigger word + stim_ngram prior words; matches
                                        # circuit-building get_ngrams -> stim_ngram+1 tokens)
    stim_max_chars: int = 96            # "chars" stimulus only: fixed char-id width L (the LAST L
                                        # characters of the n-gram; shorter n-grams are right-padded)


@dataclass(frozen=True)
class Dataset:
    study: str = "Fake2025Meg"       # NeuralSet study name or descriptive standalone tag
    neural: str = "meg"              # neural modality: fmri | meg | ieeg | spikes
    stimulus: str = "image"          # stimulus modality: image | text | audio | chars
    imsize: int = 16                 # image stimuli resized to imsize x imsize
    download: bool = False           # download the study before running?
    path: str | None = None          # UNUSED by active presets (runtime reads ONLY data/<name>/);
                                     # kept for the download-from-scratch path (_study_path).
    test_frac: float = 0.2           # NeuralSet temporal hold-out, or standalone split input
    cv_folds: int = 0                # >1 => reward = rotating K-fold temporal CV (no single window
                                     # to overfit); 0 = the single train/test split. Fixed, not a knob.
    reward_roi: str = "nsdgeneral"   # the region the scalar reward (test_corr) covers; a fixed
                                     # per-dataset value, NOT an operator knob.
    loader: str = "timeline"         # "timeline" (Segmenter: text/meg/ieeg) | "nsd" | "tvsd" | "nnn"
    subject: str = ""                # preset subject/animal when supported (text, NSD, TVSD)
    timeline: "Timeline | None" = None  # Segmenter params; set ONLY for loader="timeline" datasets


# Active dataset presets across neural modalities (fMRI/MEG/iEEG/spikes) and
# stimulus modalities (image/audio). Select one with NEURO_DATASET=<key>.
PRESETS: dict[str, Dataset] = {
    # Li2022Petit (Le Petit Prince), fMRI + audio: one full English subject (EN057)
    # via the Li2022PetitSubj subclass (harness/loaders/studies.py). Each spoken Word
    # triggers a segment. The neural window (start=2.0/duration=4.0) captures the
    # HRF-delayed BOLD, but audio response is near-instant so the stimulus is read
    # from a short decoupled window at the word (stim_start=0.0/stim_duration=0.5).
    "text": Dataset(
        study="Li2022PetitSubj",
        neural="fmri",
        stimulus="audio",
        timeline=Timeline(trigger="type=='Word'", start=2.0, duration=4.0,
                          stim_start=0.0, stim_duration=0.5),
    ),
    # NSD (Natural Scenes Dataset), fMRI + image, subject 1 of 8. Uses the PREPARED single-trial
    # betas (betas_fithrf_GLMdenoise_RR, Algonauts-2023 style: z-scored/session, 3-rep averaged),
    # NOT the coarse raw-timeseries HRF-peak average. Y is the per-ROI betas from
    # nsd_betas/per_subject/subjNN.pkl; X is the NSD images (shared-1000 image-disjoint hold-out).
    # Reward = the joint EARLY+VENTRAL stream (~14.5k voxels); roi_corr breaks it down per ROI
    # (early, ventral, V1/V2/V3/hV4/FFA/PPA/VWFA). Subject via NEURO_SUBJECT (1-8).
    "nsd": Dataset(
        study="NsdBetasSubj01",
        loader="nsd",
        neural="fmri",
        stimulus="image",
        imsize=128,
        download=False,
        reward_roi="early+ventral",
        subject="01",
    ),
    # Bel2026PetitListen (LPP-Listen MEG, OpenNeuro ds007523), MEG + audio. Each
    # spoken Word triggers a 0.5s epoch; MEG is near-instant so stimulus and neural
    # windows coincide ([0,0.5]s, no HRF decoupling). GOTCHA: the raw .fif is Elekta
    # MaxShield, loaded by the NeuralSet loader with allow_maxshield=True.
    "meg": Dataset(
        study="Bel2026PetitListenSample",
        neural="meg",
        stimulus="audio",
        timeline=Timeline(trigger="type=='Word'", start=0.0, duration=0.5),
    ),
    # Zada2025Podcast ("Podcast" ECoG, OpenNeuro ds005574, Hasson lab), iEEG + text:
    # epilepsy patients listening to a 30-min podcast (~5.1k transcript words). Single-subject
    # by necessity (each patient has different electrodes; default subject 01). Each transcript
    # Word triggers a 0.5s epoch; the target is the HIGH-GAMMA (70-200 Hz) envelope -- the
    # standard ECoG cognition signal, staged by scripts/process_dataset/audio_neural.py --
    # word-averaged (neural_aggregation="mean") to one power value per electrode per word.
    # STIMULUS is "chars": each word maps to the char-id encoding of its N-gram (the word + the
    # preceding stim_ngram-1 words), so a hand-written char-transformer model.py must build any
    # linguistic structure itself -- the language-encoding analogue of the vision datasets (no
    # acoustic/Mel features; the neural side and CV reward are unchanged).
    "ieeg": Dataset(
        study="Zada2025Podcast",
        neural="ieeg",
        stimulus="chars",
        reward_roi="all",
        subject="01",   # default patient; the search reward means over several (NEURO_SUBJECT) so a
                        # model must generalize ACROSS brains -- no single subject's noise to overfit.
        cv_folds=5,   # reward = 5-fold temporal CV: a single fixed hold-out (~5.1k words, 99 raw-r
                      # channels, no noise ceiling) is trivially overfit by a search that grep's it.
        timeline=Timeline(trigger="type=='Word'", start=0.0, duration=0.5,
                          neural_aggregation="mean", stim_ngram=10, stim_max_chars=96),
    ),
    # TVSD (THINGS Ventral-stream Spiking Dataset, Papale et al. 2025), image + macaque
    # spiking. NOT a NeuralSet study: ships a PRE-AVERAGED normMUA table (1024 electrodes
    # x stimuli), so loader="tvsd" (loaders.vision.tvsd) with the dataset's own image-disjoint
    # split. Reward = ALL 1024 electrodes; roi_corr breaks it down per area (V1/V4/IT). Monkey
    # F/N via NEURO_SUBJECT. Runtime reads only the in-repo data/tvsd/ tree (native JPEGs + a .pt of
    # the MUA); the gated THINGS images + raw .mat are needed ONLY by the build step
    # (scripts.process_dataset.tvsd, see its RAW + THINGS_IMAGES).
    "tvsd": Dataset(
        study="TvsdMonkeyF",
        loader="tvsd",
        neural="spikes",
        stimulus="image",
        imsize=16,
        subject="F",
        download=False,
        reward_roi="all",
    ),
    # Triple-N / NNN (Li et al., ScienceDB 10.57760/sciencedb.33556), image + macaque
    # spiking. Like TVSD it is NOT a NeuralSet study: 90 sessions ship a PRE-AVERAGED
    # Processed table (`response_best` [units x 1072], cols 1..1000 = the NSD shared-1000
    # images), so loader="nnn" (loaders.vision.nnn). Y pools reliable units (reliability>0.4,
    # ~35k of 47.5k) across all sessions. Reward = ALL reliable units; roi_corr breaks it down by
    # stream area (EVC/IT). The 1000 NSD images split deterministically (every 5th held out).
    # Self-contained: X is read straight from img_pool.mat (no NSD staging). See docs/datasets/.
    "nnn": Dataset(
        study="NnnV1",
        loader="nnn",
        neural="spikes",
        stimulus="image",
        imsize=16,
        download=False,
        reward_roi="all",
    ),
}

_key = os.environ.get("NEURO_DATASET", "text")
if _key not in PRESETS:
    raise ValueError(
        f"NEURO_DATASET={_key!r} is not a known dataset. "
        f"Valid keys: {sorted(PRESETS)}"
    )

# The one active dataset, chosen by NEURO_DATASET (default "text"). The reward is always the
# WHOLE-dataset test_corr (each preset's built-in reward_roi); there is no ROI knob.
DATASET = PRESETS[_key]

# Input-resolution override (NEURO_IMSIZE): the operator-chosen pixel size for image
# stimuli (RawPixels imsize). It only enriches the model INPUT X -- the neural target
# Y, the readout, the split and the reward region are untouched -- so the reward stays
# trustworthy; it just stops starving untrained models of resolution (AlexNet reads
# full-res). Changes the build-cache key -> a one-time re-build at the new resolution.
_imsize = os.environ.get("NEURO_IMSIZE")
if _imsize:
    DATASET = replace(DATASET, imsize=int(_imsize))

def with_subject(dataset: Dataset, subject: str) -> Dataset:
    """Return ``dataset`` rebound to ``subject`` (switching the data it loads). Supported:
    text (one English subject, e.g. EN058), nsd (1-8), tvsd (monkey F/N). Other presets keep the
    value in ``subject`` but do not rewrite the study name. This is the one place the per-subject
    data mapping lives, so a multi-subject reward can build variants in-process (see
    ``harness.reward.evaluate_subjects``)."""
    if dataset.loader == "nsd":
        sid = f"{int(subject):02d}"
        return replace(dataset, subject=sid, study=f"NsdBetasSubj{sid}")
    if dataset.loader == "tvsd":
        return replace(dataset, subject=subject, study=f"TvsdMonkey{subject.upper()}")
    if dataset.study.startswith("Li2022PetitSubj"):
        # Runtime reads data/text/<study>.pt by study name; the raw dir is the build step's concern.
        return replace(dataset, subject=subject, study=f"Li2022PetitSubj{subject}")
    return replace(dataset, subject=subject)


# Default subject binding from NEURO_SUBJECT (a single id, or the first of a comma list -- the
# reward loops over the full list; this just binds the default DATASET for single-split callers).
_subject = (os.environ.get("NEURO_SUBJECT") or "").split(",")[0].strip()
if _subject:
    DATASET = with_subject(DATASET, _subject)
