"""Optional reference baseline for the generalization test: a frozen ImageNet-AlexNet scored through
the SAME harness reward, per unit -- so you can see whether the champion's held-out scores beat a
trained-net oracle on the same subjects. Reuses the shared ``scripts/benchmark/alexnet_baseline.py``;
this module owns no model, just the comparison glue.

Score = the brain-score oracle: each ROI keeps its BEST AlexNet layer, then the stream ROIs that
partition the reward region are voxel-weighted into one number comparable to the champion's reward.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark")
sys.path.insert(0, _BENCH)                       # for alexnet_baseline + its vendored SRP
from alexnet_baseline import AlexNetSRP, _LAYERS, _readout

STREAMS = ("early", "ventral")                   # the ROIs that partition the NSD reward region


def _roi_counts(roi_masks, Ytr, DATASET, R):
    """Reliable-voxel count per ROI within the reward region (same slicing as ``_readout``)."""
    study = R._study_for(DATASET)
    mask = (R._standalone_target_mask(roi_masks, DATASET.reward_roi) if study is None
            else R._slice_target_mask(study, DATASET.reward_roi, int(Ytr.shape[1])))
    rm = ({k: np.asarray(v, bool)[mask] for k, v in roi_masks.items()} if mask is not None
          else {k: np.asarray(v, bool) for k, v in roi_masks.items()})
    return {k: int(v.sum()) for k, v in rm.items()}


def alex_oracle(split, DATASET, R, roi_breakdown, layers=tuple(_LAYERS)):
    """Best-layer-per-ROI frozen-AlexNet reward on this split (voxel-weighted over the stream ROIs)."""
    Xtr, Xte, Ytr, Yte, roi_masks, nc = split
    device = R._readout_device()
    model = AlexNetSRP(DATASET.imsize, list(layers))
    if torch.is_tensor(Xtr):
        Xtr, Xte = Xtr.to(device), Xte.to(device)
    ftr, fte = model.forward_layers(Xtr), model.forward_layers(Xte)
    per = {}                                                          # layer -> {roi: test score}
    for L in layers:
        rec = _readout(torch.from_numpy(ftr[L]).to(device), torch.from_numpy(fte[L]).to(device),
                       Ytr, Yte, dict(roi_masks), nc, DATASET, R, roi_breakdown, device)
        per[L] = {roi: v["test"] for roi, v in rec["roi_corr"].items()}
    counts = _roi_counts(roi_masks, Ytr, DATASET, R)
    best = {roi: max(layers, key=lambda L: per[L].get(roi, -9.0)) for roi in per[next(iter(per))]}
    rois = {roi: {"alex_best": round(per[best[roi]][roi], 5), "layer": best[roi],
                  "voxels": counts.get(roi)} for roi in best}
    streams = [s for s in STREAMS if counts.get(s)]
    wsum = sum(counts[s] for s in streams) or 1
    overall = sum(counts[s] * rois[s]["alex_best"] for s in streams) / wsum
    return {"overall": round(overall, 5), "rois": rois}
