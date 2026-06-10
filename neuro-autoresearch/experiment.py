"""Editable run-config (the subprofile) for the autoresearch loop.

This is the small set of knobs you tune *about a run* rather than about the model
itself -- kept out of ``model.py`` so that file stays purely the model. The active
dataset is chosen separately via the ``NEURO_DATASET`` env var (see
``harness/datasets.py``).
"""
from __future__ import annotations

# How many evaluation passes ``python -m harness.run`` performs, each appending one
# row to ``results/<study>.csv``. The default 1 is the right autoresearch cadence
# (one row per model edit); raise it only once the reward is stochastic
# (e.g. multi-seed), where repeated passes measure its variance.
ITERATIONS = 1

# Slices to log alongside the scalar reward (per-region feedback). Each name becomes
# a ``corr_<name>`` column in ``results/<study>.csv``: the mean per-target
# correlation over that slice's targets. The columns are FROZEN by the first row
# written to a study's CSV, so declare the set you want ONCE and keep it stable
# (later additions are ignored with a warning). Default is none (matches the default
# ``text`` dataset, which has no slice resolver); for NSD set nsdgeneral sub-ROIs
# (e.g. ["V1", "V2", "V3", "hV4", "FFA", "PPA"]).
LOG_SLICES: list[str] = []
