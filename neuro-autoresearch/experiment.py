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

# Extra per-region feedback columns (``corr_<name>``) to log ALONGSIDE the scalar
# reward, for studies with no built-in ROI menu (text/meg -> usually ``[]``). For NSD
# this is IGNORED: the harness logs its own FIXED region menu (V1/hV4/FFA/PPA/EBA/VWFA
# + early/ventral) automatically, so the per-region feedback is guaranteed without
# being declared here. The single region the reward optimises is chosen with NEURO_ROI.
LOG_SLICES: list[str] = []
