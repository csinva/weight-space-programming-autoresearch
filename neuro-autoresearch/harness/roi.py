"""Target regions for the reward -- which columns of Y get scored, and the per-ROI breakdown.

Two related, agent-non-editable concerns kept together:
  * SELECTING the reward region's target columns (``reward_region_mask`` + ``select_targets``): the
    reward narrows ``Y`` to the dataset's region before the readout fits. Every loader bakes its
    region into ``Y`` at load time and hands back ``roi_masks`` (the named sub-regions over the scored
    columns), so selection is a single lookup -- the dataset's ``reward_roi`` either names one of those
    sub-ROIs (narrow to it) or is a whole-region tag (score every loaded target).
  * The per-ROI BREAKDOWN (``roi_breakdown``): advisory feedback only -- the reward stays one scalar
    over the whole region; this just averages the per-target r the readout already computed within
    each named sub-ROI.
"""
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
            out[name] = {"train": round(float(tr[m].mean()), 4),
                         "test": round(float(te[m].mean()), 4)}
    return out


def select_targets(Y, mask):
    """Select the reward-region columns of the neural targets ``Y`` (or all if ``mask`` is None)."""
    if mask is None:
        return Y
    return Y[:, mask]


def reward_region_mask(roi_masks: dict, reward_roi: str):
    """Boolean mask over the scored targets for the dataset's reward region, or ``None`` = all targets.

    Loaders bake their region into ``Y`` and return ``roi_masks`` (the named sub-regions over those
    columns). ``reward_roi`` may narrow the reward to ONE named sub-ROI -- a key of ``roi_masks`` (e.g.
    ``"IT"`` on nnn/tvsd, ``"FFA"`` on nsd). A whole-region tag (anything that is not an ``roi_masks``
    key, e.g. ``"all"`` or NSD's ``"early+ventral"``) returns ``None`` -> the readout fits every loaded
    target. Timeline datasets (iEEG) hand back an empty ``roi_masks`` -> always ``None``.
    """
    m = roi_masks.get(reward_roi)
    return None if m is None else np.asarray(m, dtype=bool)
