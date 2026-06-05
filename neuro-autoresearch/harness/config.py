"""The single dataset this proof-of-concept runs on.

Switching datasets is a ONE-LINE edit: change ``DATASET = Dataset(...)`` below.
A dataset declares both its **stimulus modality** (``stimulus``: image | text |
audio) and its **neural modality** (``neural``: meg | eeg | fmri); the harness
(``data.py``) dispatches the right NeuralSet extractor for each. The model and
the reward stay modality-agnostic, so the same loop runs across datasets and
modalities.

For real signal, name a study from ``neuralset.Study.catalog()`` and set
``download=True`` (e.g. ``Allen2022MassiveSample`` fMRI image, ``Li2022PetitSample``
fMRI word, ``Grootswagers2022HumanSample`` EEG image).
"""
from __future__ import annotations

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


# The one active dataset. Edit this line to target another dataset.
# STORY (Le Petit Prince), fMRI + word stimulus. NeuralSet ``Li2022PetitSample``
# sample, pre-downloaded at the path below (download=False, runs offline). Each
# spoken Word triggers a segment; the stimulus is the word's [length, frequency]
# and the target is the fMRI volume. start=2.0/duration=4.0 captures the delayed
# (HRF) BOLD response; the stimulus is the trigger word, so the offset only
# shifts the neural window, not X.
DATASET = Dataset(
    study="Li2022PetitSample",
    trigger="type=='Word'",
    neural="fmri",
    stimulus="text",
    start=2.0,
    duration=4.0,
    download=False,
    path="/home/t-ymehta/Research/neuralset/data/li2022petit_sample",
    test_frac=0.2,
)

# Switch back to the Natural Scenes Dataset (NSD), fMRI + image stimulus, by
# replacing the DATASET above with this one (sample pre-downloaded, offline):
#
# DATASET = Dataset(
#     study="Allen2022MassiveSample", trigger="type=='Image'",
#     neural="fmri", stimulus="image", imsize=16, start=0.0, duration=3.0,
#     download=False, path="/home/t-ymehta/Research/neuralset/data/nsd_sample",
#     test_frac=0.2,
# )
