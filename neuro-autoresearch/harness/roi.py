"""Per-ROI correlation breakdown -- advisory feedback only (the reward stays one scalar over the
whole region). Given the per-target Pearson r the readout already computed, average it within each
ROI. The agent should NOT edit this file."""
from __future__ import annotations

import numpy as np


def roi_breakdown(roi_masks: dict, train_corr, test_corr) -> dict:
    """``{roi: {"train": mean r, "test": mean r}}`` over each ROI's columns.

    ``roi_masks`` maps an ROI name to a boolean mask aligned to the scored targets (empty -> ``{}``).
    """
    tr, te = np.asarray(train_corr), np.asarray(test_corr)
    out = {}
    for name, m in roi_masks.items():
        m = np.asarray(m, dtype=bool)
        if m.any():
            out[name] = {"train": round(float(tr[m].mean()), 5),
                         "test": round(float(te[m].mean()), 5)}
    return out
