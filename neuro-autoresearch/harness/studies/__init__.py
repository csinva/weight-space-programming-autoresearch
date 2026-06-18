"""Per-study NeuralSet subclasses.

Importing this package registers every custom study in
``neuralset.Study.catalog()`` so a ``Dataset.study`` name (e.g. ``"NsdSubj01"``)
resolves. ``harness.loaders.neuralset`` imports it for that side effect.
"""
from . import (  # noqa: F401  (import = registration)
    li2022_subj,
    nsd_subjects,
    zada2025podcast,
)
