"""Optional reference baseline for the generalization test: a frozen ImageNet-AlexNet scored through
the SAME harness reward, per unit -- so you can see whether the champion's held-out scores beat a
trained net on the same subjects. Reuses the shared ``scripts/benchmark/alexnet_baseline.py``;
this module owns no model, just the comparison glue.

Score = the SAME protocol as every AlexNet comparison in this repo: every layer (conv1-5 +
fc6/fc7/fc8) sparse-random-projected to 2048 and CONCATENATED into one representation, then ONE ridge
readout on the reward region -- NOT a best-layer cherry-pick. ``overall`` is that single concat
``test_corr`` (directly comparable to the champion's reward); ``rois`` is its per-ROI breakdown.
"""
from __future__ import annotations

import os
import sys

import torch

_BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark")
sys.path.insert(0, _BENCH)                       # for alexnet_baseline + its vendored SRP
from alexnet_baseline import AlexNetSRP, _LAYERS, _readout

import numpy as np


def alex_baseline(split, DATASET, R, roi_breakdown, layers=tuple(_LAYERS), dim=2048):
    """Concat-all-layers frozen-AlexNet reward on this split (every tap SRP->``dim``, one readout)."""
    Xtr, Xte, Ytr, Yte, roi_masks, nc = split
    device = R._readout_device()
    model = AlexNetSRP(DATASET.imsize, list(layers), dim=dim)
    if torch.is_tensor(Xtr):
        Xtr, Xte = Xtr.to(device), Xte.to(device)
    ftr, fte = model.forward_layers(Xtr), model.forward_layers(Xte)
    cat_tr = np.concatenate([ftr[L] for L in layers], axis=1)
    cat_te = np.concatenate([fte[L] for L in layers], axis=1)
    rec = _readout(torch.from_numpy(cat_tr).to(device), torch.from_numpy(cat_te).to(device),
                   Ytr, Yte, dict(roi_masks), nc, DATASET, R, roi_breakdown, device)
    rois = {roi: {"alex": round(v["test"], 5)} for roi, v in rec["roi_corr"].items()}
    return {"overall": round(rec["test_corr"], 5), "d_out": rec["d_out"], "rois": rois}
