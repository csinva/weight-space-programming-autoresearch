"""Multi-subject helpers for the vision baselines.

Mirror ``harness.reward.evaluate_subjects``: a comma ``NEURO_SUBJECT`` / ``--subject`` list scores
EACH subject on its OWN units (a separate ``with_subject`` split) and averages the per-subject rows
with EQUAL weight per subject -- so the baseline matches the search reward's multi-subject mean
exactly, instead of pooling every subject's units into one unit-weighted readout. A single token (or
none) is the plain single-split path.
"""
from __future__ import annotations

import numpy as np


def subject_tokens(cli_subject=None) -> list[str]:
    """Ordered, de-whitespaced subject tokens from ``--subject`` (a str, possibly comma-separated).
    Empty / None -> ``[]`` = the dataset's default subject (the single-split path)."""
    return [t.strip() for t in str(cli_subject or "").split(",") if t.strip()]


def common_roi_keys(roi_dicts: list[dict]) -> set:
    """ROIs present for EVERY subject (like ``reward._mean_rows``): a reliability cut can leave an ROI
    empty on some subjects, and a partial mean would be misleading, so average only the shared ones."""
    return set.intersection(*(set(d or {}) for d in roi_dicts)) if roi_dicts else set()


def mean_scalar(vals) -> float:
    """Equal-weight mean of per-subject scalars, rounded to 4 dp (the harness reward's precision)."""
    return round(float(np.mean(list(vals))), 4)
