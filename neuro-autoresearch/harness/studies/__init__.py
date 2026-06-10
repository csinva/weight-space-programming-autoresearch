"""Per-study NeuralSet subclasses.

Importing this package registers every custom study in
``neuralset.Study.catalog()`` so a ``Dataset.study`` name (e.g. ``"NsdSubj01"``)
resolves. ``harness.data`` imports it for that side effect.
"""
from . import li2022_subj, nsd_subj01  # noqa: F401  (import = registration)
