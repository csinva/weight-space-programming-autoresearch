"""The one dataset this proof-of-concept runs on. Change it freely.

A dataset is fully specified by four fields:
  * ``study``    — any name in ``neuralset.Study.catalog()`` (NeuralFetch registers
                   ~130 real datasets; the defaults below need no download).
  * ``trigger``  — which events to lock onto (the stimuli), e.g. ``"type=='Image'"``.
  * ``neural``   — the neural modality extractor name (``meg`` / ``eeg`` / ``fmri``).
  * ``imsize``   — stimulus images are resized to ``imsize × imsize`` before the model.

Swap ``STUDY`` for a real dataset (and set ``download=True``) when you want real
signal, e.g. ``Allen2022MassiveSample`` (fMRI) or ``Grootswagers2022HumanSample`` (EEG).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    study: str = "Fake2025Meg"      # offline demo (no download)
    trigger: str = "type=='Image'"
    neural: str = "meg"             # meg | eeg | fmri
    imsize: int = 16
    start: float = 0.0
    duration: float = 0.3
    download: bool = False
    test_frac: float = 0.2
    seed: int = 0


# The single active dataset for this PoC. Edit this line to switch datasets.
DATASET = Dataset()
