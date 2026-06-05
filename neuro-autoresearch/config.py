"""The single dataset this proof-of-concept runs on.

Switching datasets is a ONE-LINE edit: change ``DATASET = Dataset(...)`` below.
The default is the offline ``Fake2025Meg`` image study (no download needed).
For real signal, name a study from ``neuralset.Study.catalog()`` and set
``download=True`` (e.g. ``Allen2022MassiveSample`` fMRI, ``Grootswagers2022HumanSample`` EEG).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    study: str = "Fake2025Meg"       # any name in neuralset.Study.catalog()
    trigger: str = "type=='Image'"   # which events are the stimuli
    neural: str = "meg"              # meg | eeg | fmri
    imsize: int = 16                 # stimulus images resized to imsize x imsize
    start: float = 0.0               # segment start, seconds after trigger
    duration: float = 0.3            # segment duration, seconds
    download: bool = False           # download the study before running?
    test_frac: float = 0.2           # held-out fraction
    seed: int = 0                    # split seed


# The one active dataset. Edit this line to target another dataset.
DATASET = Dataset()
