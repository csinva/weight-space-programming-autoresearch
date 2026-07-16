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
    # Huth fMRI-language: reward = the mean over the three deeply-sampled reference subjects
    # UTS01/02/03. heldout is TBD (the shallow subjects are not staged yet), so generalize.py has no
    # held-out subjects for huth for now -- but the SEARCH subjects live here so every launcher reads
    # the same default.
    "huth": {"axis": "NEURO_SUBJECT", "search": ["UTS01", "UTS02"], "heldout": ["UTS03"]},
    # Future datasets that switch data by subject (tvsd) drop in the same shape.
}


def split_for(dataset: str) -> dict:
    """Return a dataset's {axis, search, heldout} split, or raise naming this file if it's undeclared."""
    if dataset not in GEN_SPLITS:
        raise SystemExit(f"no generalization split for dataset {dataset!r}; add one to "
                         f"scripts/eval/splits.py (have: {list(GEN_SPLITS)})")
    return GEN_SPLITS[dataset]


def search_subjects_csv(dataset: str) -> str:
    """The dataset's SEARCH subjects as a comma string (e.g. ``"1,2"``), or ``""`` if the dataset has
    no declared split (the caller then falls back to the preset's default subject). This is the ONE
    source the search and simplify launchers read for the per-dataset subject default, so the
    search reward and the held-out generalization split can never drift apart. Never raises."""
    sp = GEN_SPLITS.get(dataset)
    return ",".join(sp["search"]) if sp else ""
