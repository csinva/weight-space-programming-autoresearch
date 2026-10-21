"""The five evaluation datasets and their search and held-out subject cohorts.

This is the single source of truth for evaluation cohorts. The search launcher reads the same
``search`` subjects that the evaluation runner later compares with ``heldout`` subjects.
"""
from __future__ import annotations

VISION_DATASETS = ("nsd", "tvsd", "nnn")
LANGUAGE_DATASETS = ("huth", "ieeg")

DATASETS = {
    "nsd": {
        "domain": "vision",
        "metric": "noise_normalized_r2",
        "axis": "NEURO_SUBJECT",
        "search": ["1", "2"],
        "heldout": ["5", "7"],
        "rois": [
            "early", "ventral", "V1", "V2", "V3", "hV4", "FFA", "PPA", "VWFA",
        ],
    },
    "tvsd": {
        "domain": "vision",
        "metric": "mean_pearson_r",
        "axis": "NEURO_SUBJECT",
        "search": ["F"],
        "heldout": ["N"],
    },
    "nnn": {
        "domain": "vision",
        "metric": "noise_normalized_r2",
        "axis": "NEURO_SUBJECT",
        "search": ["1", "2", "3"],
        "heldout": ["4", "5"],
    },
    "huth": {
        "domain": "language",
        "metric": "mean_pearson_r",
        "axis": "NEURO_SUBJECT",
        "search": ["UTS01", "UTS02"],
        "heldout": ["UTS03"],
    },
    "ieeg": {
        "domain": "language",
        "metric": "mean_pearson_r",
        "axis": "NEURO_SUBJECT",
        "search": ["01", "02", "03"],
        "heldout": ["04", "05"],
    },
}


def evaluation_for(dataset: str) -> dict:
    """Return one dataset's domain and subject cohorts."""
    if dataset not in DATASETS:
        raise SystemExit(f"no evaluation definition for dataset {dataset!r}; add one to "
                         f"experiments/evaluation/datasets.py (have: {list(DATASETS)})")
    return DATASETS[dataset]


def search_subjects_csv(dataset: str) -> str:
    """Return the search cohort as a comma-separated string, or empty for an unknown dataset."""
    config = DATASETS.get(dataset)
    return ",".join(config["search"]) if config else ""
