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
    _canonical_test_ids,
    _collapse_repeats,
    _ensure_roi_label_file,
    _image_id,
    _nsd_subject,
    _roi_voxel_keep,
)
from harness.reward import (
    _ALPHAS,
    _as_f64,
    _corr_per_column_t,
    _readout_device,
    _ridge_gcv_coef,
)

# Ventral visual stream = the NSD "streams" atlas midventral(2)+ventral(3) labels.
_VENTRAL_LABELS = (2, 3)
# Per-ROI slices to report alongside the overall and ventral-stream scores.
_ROIS = ["V1", "hV4", "FFA", "PPA", "EBA", "VWFA"]


# --- AlexNet fc7 features -------------------------------------------------------
def alexnet_fc7_features(events_segments, device, batch_size=256):
    """fc7 (penultimate, 4096-d) features for each unique stimulus image.

    Reads images through NeuralSet (``trigger.read()``) and batches them through a
    frozen ImageNet-pretrained AlexNet on the GPU. Returns ``{filepath: np.ndarray}``.
    """
    from torchvision.models import AlexNet_Weights, alexnet

    weights = AlexNet_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()           # resize 256 -> centre-crop 224 -> normalise
    model = alexnet(weights=weights).to(device).eval()
    # Truncate the classifier to fc7: [Dropout, Linear, ReLU, Dropout, Linear, ReLU]
    # -> the 4096-d penultimate representation (post-ReLU), dropping the 1000-way head.
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:6])

    # One representative segment (hence one readable event) per unique image.
    by_path = {}
    for s in events_segments:
        fp = getattr(s.trigger, "filepath", None)
        if fp is not None and fp not in by_path:
            by_path[fp] = s.trigger
    paths = list(by_path)
    print(f"  AlexNet fc7 on {len(paths)} unique images "
          f"({type(model).__name__}, frozen, {device}) ...")

    feats: dict[str, np.ndarray] = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            imgs = torch.stack([preprocess(by_path[p].read().convert("RGB")) for p in chunk])
            out = model(imgs.to(device)).float().cpu().numpy()
            for p, f in zip(chunk, out):
                feats[p] = f
            if (i // batch_size) % 10 == 0:
                print(f"    {min(i + batch_size, len(paths))}/{len(paths)} "
                      f"({time.time() - t0:.0f}s)")
    print(f"  features ready in {time.time() - t0:.0f}s; dim={next(iter(feats.values())).shape[0]}")
    return feats


# --- masks ----------------------------------------------------------------------
def _ventral_voxel_mask(study, n_voxels):
    """Boolean keep-mask (length n_voxels) for ventral-stream voxels in nsdgeneral."""
    import nibabel
    from neuralfetch.studies.allen2022massive import _get_roi_mask

    mask = _get_roi_mask(study._roi_path(_nsd_subject(study)))
    labels = nibabel.load(_ensure_roi_label_file(study, "streams")).get_fdata()
    keep = np.isin(labels[mask], _VENTRAL_LABELS)
    assert keep.shape[0] == n_voxels
    return keep


def _target_mask(voxel_keep, n_tr, kept):
    """Voxel-level keep-mask -> mask over the surviving target columns.

    Targets are voxel-major ((voxel, tr)); repeat the per-voxel mask across the
    ``n_tr`` TRs, then restrict to the train-non-constant ``kept`` columns.
    """
    full = np.repeat(voxel_keep, n_tr)
    return full[kept]


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

    # canonical shared-1000 held-out split (image-disjoint), mirroring load_split.
    shared = _canonical_test_ids(study)
    is_test = np.array([_image_id(s) in shared for s in uniq])
    tr = np.flatnonzero(~is_test)
    te = np.flatnonzero(is_test)
    assert set(uniq[tr]).isdisjoint(set(uniq[te])), "train/test share images"
    print(f"  split: {tr.size} train / {te.size} test")

    # 2. AlexNet fc7 features, aligned to the unique-image order.
    seg = ns.Segmenter(extractors={}, trigger_query=d.trigger,
                       start=d.start, duration=d.duration, drop_incomplete=True)
    segments = seg.apply(study.run()).segments
    feats = alexnet_fc7_features(segments, device)
    missing = [p for p in uniq if p not in feats]
    if missing:
        sys.exit(f"{len(missing)} unique images had no AlexNet feature (e.g. {missing[:2]})")
    A = np.stack([feats[p] for p in uniq]).astype("float32")
    print(f"  feature matrix A{A.shape}")

    # 3. harness ridge readout (fit on ALL voxels jointly), per-target test corr.
    A_t = torch.from_numpy(A).to(device)
    Y_t = _as_f64(Y, device)
    A_tr, A_te = _as_f64(A_t[tr], device), _as_f64(A_t[te], device)
    Y_tr, Y_te = Y_t[tr], Y_t[te]

    kept = (Y_tr.std(0) > 0).cpu().numpy()          # drop train-constant targets
    Y_tr, Y_te = Y_tr[:, torch.from_numpy(kept).to(device)], Y_te[:, torch.from_numpy(kept).to(device)]
    print(f"  targets after dropping train-constant: {int(kept.sum())}/{kept.size}")

    t0 = time.time()
    x_off, y_off = A_tr.mean(0), Y_tr.mean(0)
    coef = _ridge_gcv_coef(A_tr - x_off, Y_tr - y_off)
    pred_te = (A_te - x_off) @ coef + y_off
    pred_tr = (A_tr - x_off) @ coef + y_off
    test_vec = _corr_per_column_t(pred_te, Y_te).cpu().numpy()
    train_corr = float(_corr_per_column_t(pred_tr, Y_tr).mean())
    print(f"  readout fit+predict in {time.time() - t0:.1f}s")

    # 4. scores: overall, per-ROI, ventral stream.
    def slice_corr(voxel_keep):
        m = _target_mask(voxel_keep, n_tr, kept)
        return float(test_vec[m].mean()) if m.any() else float("nan"), int(m.sum())

    print("\n" + "=" * 60)
    print(f"AlexNet fc7 encoding on NSD subj01  (n_test={te.size})")
    print("=" * 60)
    print(f"  {'overall (all nsdgeneral)':<28} test_corr={test_vec.mean():.4f}  "
          f"train_corr={train_corr:.4f}  ({kept.sum()} targets)")
    vent_keep = _ventral_voxel_mask(study, n_voxels)
    vc, vn = slice_corr(vent_keep)
    print(f"  {'VENTRAL STREAM (reward)':<28} test_corr={vc:.4f}  ({vn} targets)")
    print("  per-ROI:")
    for roi in _ROIS:
        keep_roi, nv = _roi_voxel_keep(study, roi)
        if keep_roi is None:
            continue
        rc, rn = slice_corr(keep_roi)
        print(f"    {roi:<8} test_corr={rc:.4f}  ({rn} targets)")

    # 5. himalaya cross-check on the identical features.
    himalaya_check(A_tr, A_te, Y_tr, Y_te, test_vec, device)


def himalaya_check(A_tr, A_te, Y_tr, Y_te, custom_test_vec, device):
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
    est.fit(A_tr.to(dtype), Y_tr.to(dtype))
    pred = est.predict(A_te.to(dtype))
    hima_vec = _corr_per_column_t(torch.as_tensor(np.asarray(pred), device=device, dtype=torch.float64),
                                  Y_te).cpu().numpy()
    print(f"  himalaya fit+predict in {time.time() - t0:.1f}s")
    print(f"  mean test_corr   custom={custom_test_vec.mean():.4f}  "
          f"himalaya={hima_vec.mean():.4f}  Δ={custom_test_vec.mean() - hima_vec.mean():+.4f}")
    print(f"  per-target maps  r(custom,himalaya)={np.corrcoef(custom_test_vec, hima_vec)[0, 1]:.5f}  "
          f"mean|Δr|={np.abs(custom_test_vec - hima_vec).mean():.3e}  "
          f"max|Δr|={np.abs(custom_test_vec - hima_vec).max():.3e}")


if __name__ == "__main__":
    main()
