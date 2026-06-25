"""Target regions for the reward -- which columns of Y get scored, and the per-ROI breakdown.

Two related, agent-non-editable concerns kept together:
  * SELECTING the reward region's target columns (``_study_for`` + ``_slice_target_mask`` +
    ``_select_targets``): the reward narrows ``Y`` to the dataset's region (e.g. NSD ventral) before
    the readout fits.
  * The per-ROI BREAKDOWN (``roi_breakdown``): advisory feedback only -- the reward stays one scalar
    over the whole region; this just averages the per-target r the readout already computed within
    each named sub-ROI.
"""
from __future__ import annotations

import numpy as np

from harness.datasets import Dataset


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


def _select_targets(Y, mask):
    """Select the reward-region columns of the neural targets ``Y`` (or all if ``mask`` is None)."""
    if mask is None:
        return Y
    return Y[:, mask]


def _standalone_target_mask(roi_masks: dict, reward_roi: str):
    """Reward-region mask for the standalone loaders (nsd/tvsd/nnn), which have no ns.Study resolver.

    These loaders bake their region into ``Y`` at load time and hand back ``roi_masks`` (the named
    sub-regions over the scored columns). ``reward_roi`` may narrow the reward to ONE of those named
    sub-ROIs -- a key of ``roi_masks`` (e.g. ``"IT"`` on nnn/tvsd, ``"FFA"`` on nsd), the analog of
    the NeuralSet ROI menu. A whole-region value (anything that is not an ``roi_masks`` key, e.g.
    ``"all"`` or NSD's ``"early+ventral"``) returns ``None`` -> the readout fits every loaded target.
    """
    m = roi_masks.get(reward_roi)
    return None if m is None else np.asarray(m, dtype=bool)


def _study_for(d: Dataset):
    """Construct the (no-download) NeuralSet study used to resolve the reward region.

    Table-loader and prepared-betas datasets are not NeuralSet studies and carry their region
    labels in the loader (TVSD) or as a per-ROI betas key (NSD-betas), not an ns.Study ROI
    resolver -- return None so the reward averages over the targets the loader already selected.
    """
    if getattr(d, "loader", "neuralset") != "neuralset":
        return None
    import neuralset as ns
    from harness.loaders.shared import _study_path
    return ns.Study(name=d.study, path=_study_path(d))


def _slice_target_mask(study, name: str, n_targets: int):
    """Boolean mask (length ``n_targets``) selecting region ``name``'s target columns.

    NSD: ``name`` is an ROI resolved by ``neuralset.roi._roi_voxel_keep`` to a per-voxel keep mask over the
    nsdgeneral voxels; the eval targets are those voxels (voxel-major if multi-TR), so the per-voxel
    mask is repeated across the TRs. Returns ``None`` (the clean extension point) when the study has
    no ROI resolver, the name is unknown, or the targets do not map cleanly onto the voxels.
    """
    if not hasattr(study, "_roi_path"):      # non-volumetric studies (text/meg): no resolver
        return None
    from harness.loaders.neuralset.roi import _roi_voxel_keep

    try:
        keep, n_voxels = _roi_voxel_keep(study, name)
    except (ValueError, FileNotFoundError):
        return None
    if keep is None:                          # whole-region name (e.g. nsdgeneral)
        return None
    if n_targets % n_voxels:                  # targets are not a clean multiple of the voxels
        return None
    return np.repeat(keep, n_targets // n_voxels)
