"""Per-dataset generalization splits -- the ONLY dataset-specific knowledge the runner needs.

A dataset generalizes along ONE *axis*: the harness env var that selects an independent unit of data
(the subject, for the table datasets). The search optimizes its reward over the ``search`` units; the
generalization test re-evaluates the winning model, UNCHANGED, on the ``heldout`` units it never saw.
Add a dataset by declaring its axis + unit lists here -- ``generalize.py`` is otherwise dataset-agnostic.
"""
from __future__ import annotations

# dataset -> {axis: env var selecting a unit, search: optimized units, heldout: generalization units}
GEN_SPLITS = {
    # NSD: search reward is the subj 1,2 mean (search/search.sh default); 5,7 are staged but held out.
    "nsd": {"axis": "NEURO_SUBJECT", "search": ["1", "2"], "heldout": ["5", "7"]},
    # iEEG Podcast: reward = the cross-subject mean over patients 01-03 (search/search.sh default);
    # 04-05 are staged but held out, so generalization = predicting an UNSEEN patient's high-gamma.
    "ieeg": {"axis": "NEURO_SUBJECT", "search": ["01", "02", "03"], "heldout": ["04", "05"]},
    # NNN: reward = the mean over search monkeys M1,M2,M3 (search/search.sh default);
    # M4,M5 held out, so generalization = predicting an UNSEEN animal's IT/EVC spiking.
    "nnn": {"axis": "NEURO_SUBJECT", "search": ["1", "2", "3"], "heldout": ["4", "5"]},
    # Future datasets that switch data by subject (tvsd) drop in the same shape.
}


def split_for(dataset: str) -> dict:
    """Return a dataset's {axis, search, heldout} split, or raise naming this file if it's undeclared."""
    if dataset not in GEN_SPLITS:
        raise SystemExit(f"no generalization split for dataset {dataset!r}; add one to "
                         f"scripts/generalization/splits.py (have: {list(GEN_SPLITS)})")
    return GEN_SPLITS[dataset]
