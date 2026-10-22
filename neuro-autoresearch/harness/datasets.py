"""Fixed dataset presets for RSA search and cross-dataset encoding evaluation.

Vision data is read by the vendored looped-networks readers, with configurable
storage paths and fixed scientific filters. Model inputs remain flattened HWC
RGB float tensors in [0,1]. Language presets retain their original loaders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

from harness.protocols import canonical_dataset, protocol_for


@dataclass(frozen=True)
class Timeline:
    """NeuralSet Segmenter params -- set ONLY for timeline datasets (``loader="timeline"``: iEEG,
    plus future language timelines). Vision table datasets (nsd/tvsd/nnn) leave ``timeline=None``."""
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
    """One dataset preset: the stimulus + neural modality, split/CV policy, reward region, and which
    loader path to take. The ``PRESETS`` entry ``NEURO_DATASET`` selects is what fixes the whole eval;
    every field here is a scientific constant of that benchmark, NOT an operator knob."""
    study: str = "Unnamed"           # NeuralSet study name or descriptive standalone tag
    neural: str = "ieeg"             # neural modality: fmri | ieeg | spikes
    stimulus: str = "image"          # stimulus modality: image | text | audio | chars
    imsize: int = 128                # image stimuli resized to imsize x imsize (hard floor MIN_IMSIZE;
                                     # 128 is the minimum -- coarser starves models of form/texture detail)
    download: bool = False           # download the study before running?
    path: str | None = None          # legacy NeuralSet download path; vision uses source-path env vars
    test_frac: float = 0.2           # NeuralSet temporal hold-out, or standalone split input
    cv_folds: int = 0                # >1 => reward = rotating K-fold temporal CV (no single window
                                     # to overfit); 0 = the single train/test split. Fixed, not a knob.
    feature_pipeline: str = ""       # optional post-forward feature transform applied between the
                                     # model and the fixed ridge (e.g. "lanczos_fir" for the Huth
                                     # fMRI-language dataset: Lanczos word->TR + z-score + FIR delays).
                                     # Empty => no transform (every other dataset). A fixed benchmark
                                     # property, NOT an operator knob.
    reward_roi: str = "nsdgeneral"   # the region the scalar reward (test_corr) covers; a fixed
                                     # per-dataset value, NOT an operator knob.
    metric: str = "encoding"         # fixed evaluation method: rsa for search, encoding for final
    stage: str = "legacy"            # search | final | legacy
    loader: str = "timeline"         # timeline | huth | laion | triplen | nsd | tvsd
    subject: str = ""                # preset subject/animal when supported (iEEG, NSD, TVSD)
    timeline: "Timeline | None" = None  # Segmenter params; set ONLY for loader="timeline" datasets


# Anatomical region coverage and thresholds live in loaders/vision/looped.py.
PRESETS: dict[str, Dataset] = {
    "laion": Dataset(study="LaionFmriSub01", loader="laion", neural="fmri",
                     reward_roi="all", subject="sub-01", metric="rsa", stage="search"),
    "triplen": Dataset(study="TripleNM1", loader="triplen", neural="spikes",
                       reward_roi="all", subject="M1", metric="rsa", stage="search"),
    "nsd": Dataset(study="NsdBetasSubj01", loader="nsd", neural="fmri",
                   reward_roi="all", subject="01", metric="encoding", stage="final"),
    "tvsd": Dataset(study="TvsdMonkeyF", loader="tvsd", neural="spikes",
                    reward_roi="all", subject="F", metric="encoding", stage="final"),
    "huth": Dataset(study="HuthLangFmriUTS03", loader="huth", neural="fmri",
                    stimulus="chars", reward_roi="all", subject="UTS03",
                    feature_pipeline="lanczos_fir"),
    "ieeg": Dataset(study="Zada2025Podcast", neural="ieeg", stimulus="chars",
                    reward_roi="all", subject="01", cv_folds=5,
                    timeline=Timeline(trigger="type=='Word'", start=0.0, duration=0.5,
                                      neural_aggregation="mean", stim_ngram=10, stim_max_chars=96)),
}
# Kept for callers that index the historical registry directly.
PRESETS["nnn"] = PRESETS["triplen"]
PRESETS["laion-fmri"] = PRESETS["laion"]
PRESETS["laion_fmri"] = PRESETS["laion"]
PRESETS["triple-n"] = PRESETS["triplen"]

MIN_IMSIZE = 128


def with_subject(dataset: Dataset, subject: str) -> Dataset:
    """Bind a public subject identifier, validating it without loading neural data."""
    if dataset.loader == "laion":
        sid = subject.strip().lower()
        if not sid.startswith("sub-"):
            sid = f"sub-{int(sid):02d}"
        if sid not in protocol_for("laion").search_subjects:
            raise ValueError(f"Unknown LAION subject {subject!r}; expected sub-01,03,05,06,07")
        return replace(dataset, subject=sid, study=f"LaionFmri{sid}")
    if dataset.loader == "nsd":
        sid = int(subject)
        if f"{sid:02d}" not in protocol_for("nsd").final_subjects:
            raise ValueError("NSD subject must be 1 through 8")
        return replace(dataset, subject=f"{sid:02d}", study=f"NsdBetasSubj{sid:02d}")
    if dataset.loader == "tvsd":
        sid = subject.strip().upper()
        if sid not in protocol_for("tvsd").final_subjects:
            raise ValueError("TVSD subject must be F or N")
        return replace(dataset, subject=sid, study=f"TvsdMonkey{sid}")
    if dataset.loader in ("triplen", "nnn"):
        sid = subject.strip().upper()
        if not sid.startswith("M"):
            sid = f"M{int(sid)}"
        if sid not in protocol_for("triplen").search_subjects:
            raise ValueError("Triple-N subject must be M1 through M5")
        return replace(dataset, subject=sid, study=f"TripleN{sid}")
    if dataset.study.startswith("HuthLangFmri"):
        sid = subject.upper() if subject.upper().startswith("UTS") else f"UTS{int(subject):02d}"
        return replace(dataset, subject=sid, study=f"HuthLangFmri{sid}")
    if dataset.study.startswith("Zada2025Podcast"):
        return replace(dataset, subject=f"{int(subject):02d}")
    return replace(dataset, subject=subject)


def dataset_key(dataset: Dataset) -> str:
    """Canonical registry name for a dataset variant."""
    return "ieeg" if dataset.loader == "timeline" else canonical_dataset(dataset.loader)


def get_dataset(name: str, subject: str | None = None) -> Dataset:
    """Resolve a preset at call time, including the current input resolution override.

    Explicit subject wins; otherwise a single default binding may use the first
    NEURO_SUBJECT entry. Protocol evaluators validate the complete fixed cohort.
    """
    dataset = PRESETS[canonical_dataset(name)]
    imsize = os.environ.get("NEURO_IMSIZE")
    if imsize:
        dataset = replace(dataset, imsize=int(imsize))
    if dataset.stimulus == "image" and dataset.imsize < MIN_IMSIZE:
        raise ValueError(f"imsize={dataset.imsize} < MIN_IMSIZE={MIN_IMSIZE}; images require at least 128px")
    if subject is None:
        subjects = [s.strip() for s in os.environ.get("NEURO_SUBJECT", "").split(",") if s.strip()]
        subject = subjects[0] if subjects else None
    return with_subject(dataset, subject) if subject else dataset


DATASET = get_dataset(os.environ.get("NEURO_DATASET", "laion"))
