"""Reference benchmark: a TRAINED AlexNet encoding model on NSD (subject 1).

This is a STANDALONE reference, NOT a loop proposal: it deliberately breaks the
"never trained" rule to get an upper-bound encoding score for the hand-designed
models in ``model.py`` to aim at. It does not modify the fixed harness or
``model.py`` -- it only reuses harness helpers (the split build cache, the ridge
readout, the ROI resolver) so the number is comparable to the loop's reward.

What it does
------------
1. Loads NSD subject-1 neural targets from the harness build cache (all 15724
   nsdgeneral voxels x 2 TRs = 31448 targets), collapsing image repeats exactly
   as ``harness.data.load_split`` does, with the canonical shared-1000 held-out
   split.
2. Reads each unique stimulus image THROUGH NeuralSet (``event.read()``), runs it
   through a frozen ImageNet-pretrained AlexNet, and takes the **fc7 penultimate**
   features (4096-d) as the model activations.
3. Reads those activations out with the EXACT harness ridge readout
   (``harness.reward``) and reports ``test_corr`` overall, per-ROI, and over the
   **ventral visual stream** (NSD ``streams`` atlas: midventral+ventral), with the
   readout still fit jointly to ALL voxels (every ROI kept and scored separately).
4. Cross-checks the encoding score against the himalaya RidgeCV readout on the
   identical features, to confirm the harness reward number is trustworthy.

Run from the repo root::

    python3 scripts/ref_alexnet_nsd.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neuralset as ns
from harness import studies  # noqa: F401  registers NsdSubj01
from harness.datasets import PRESETS
from harness.data import (
    _build_cache_key,
    _build_cache_load,
    _cache_dir,
    _collapse_repeats,
    _roi_voxel_keep,
)
from harness.reward import _ALPHAS, _corr_per_column_t, _readout_device

from _nsd_common import (
    ROIS,
    alexnet_fc7,
    canonical_split,
    feature_matrix,
    fit_readout,
    region_masks,
)


def main():
    device = _readout_device()
    print(f"readout device: {device}")
    d = PRESETS["nsd"]
    study = ns.Study(name=d.study, path=d.path)

    # 1. neural targets (all voxels) from the harness build cache + canonical split.
    blob = os.path.join(_cache_dir(), _build_cache_key(d))
    if not os.path.exists(blob):
        sys.exit(f"build cache missing: {blob}\nrun `NEURO_DATASET=nsd python3 -m harness.run` first")
    X16, Y, onsets, stim_ids = _build_cache_load(blob)
    n_voxels = int(_roi_voxel_keep(study, "nsdgeneral")[1])
    n_tr = Y.shape[1] // n_voxels
    print(f"  build cache: {Y.shape[0]} trials, {Y.shape[1]} targets "
          f"= {n_voxels} voxels x {n_tr} TRs")
    X16, Y, onsets, uniq = _collapse_repeats(X16, Y, onsets, stim_ids)
    print(f"  collapsed to {len(uniq)} unique images")

    tr, te = canonical_split(uniq, study)
    print(f"  split: {tr.size} train / {te.size} test")

    # 2. AlexNet fc7 features, aligned to the unique-image order.
    seg = ns.Segmenter(extractors={}, trigger_query=d.trigger,
                       start=d.start, duration=d.duration, drop_incomplete=True)
    feats = alexnet_fc7(seg.apply(study.run()).segments, device, cache=False)
    A = feature_matrix(feats, uniq)
    print(f"  feature matrix A{A.shape}")

    # 3. harness ridge readout (fit on ALL voxels jointly), per-target test corr.
    t0 = time.time()
    ro = fit_readout(A, Y, tr, te, device, n_tr)
    print(f"  targets after dropping train-constant: {int(ro.kept.sum())}/{ro.kept.size}")
    print(f"  readout fit+predict in {time.time() - t0:.1f}s")

    # 4. scores: overall, ventral stream, per-ROI.
    masks = region_masks(study)
    print("\n" + "=" * 60)
    print(f"AlexNet fc7 encoding on NSD subj01  (n_test={te.size})")
    print("=" * 60)
    print(f"  {'overall (all nsdgeneral)':<28} test_corr={ro.overall:.4f}  "
          f"train_corr={ro.train_corr:.4f}  ({ro.kept.sum()} targets)")
    vc, vn = ro.slice_corr(masks["VENTRAL"])
    print(f"  {'VENTRAL STREAM (reward)':<28} test_corr={vc:.4f}  ({vn} targets)")
    print("  per-ROI:")
    for roi in ROIS:
        rc, rn = ro.slice_corr(masks[roi])
        print(f"    {roi:<8} test_corr={rc:.4f}  ({rn} targets)")

    # 5. himalaya cross-check on the identical features.
    himalaya_check(ro, device)


def himalaya_check(ro, device):
    print("\n" + "-" * 60)
    print("Himalaya RidgeCV cross-check (same features, same alpha grid)")
    print("-" * 60)
    try:
        from himalaya.backend import set_backend
        from himalaya.ridge import RidgeCV
    except ImportError:
        print("  himalaya not installed; skipping")
        return
    set_backend("torch_cuda" if device.type == "cuda" else "torch")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    est = RidgeCV(alphas=np.asarray(_ALPHAS), fit_intercept=True, cv=5)
    t0 = time.time()
    est.fit(ro.A_tr.to(dtype), ro.Y_tr.to(dtype))
    pred = est.predict(ro.A_te.to(dtype))
    hima_vec = _corr_per_column_t(torch.as_tensor(np.asarray(pred), device=device, dtype=torch.float64),
                                  ro.Y_te).cpu().numpy()
    print(f"  himalaya fit+predict in {time.time() - t0:.1f}s")
    print(f"  mean test_corr   custom={ro.overall:.4f}  "
          f"himalaya={hima_vec.mean():.4f}  Δ={ro.overall - hima_vec.mean():+.4f}")
    print(f"  per-target maps  r(custom,himalaya)={np.corrcoef(ro.test_vec, hima_vec)[0, 1]:.5f}  "
          f"mean|Δr|={np.abs(ro.test_vec - hima_vec).mean():.3e}  "
          f"max|Δr|={np.abs(ro.test_vec - hima_vec).max():.3e}")


if __name__ == "__main__":
    main()
