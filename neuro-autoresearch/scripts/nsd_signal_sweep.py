"""Diagnose the NSD neural signal by maximising a TRAINED-AlexNet encoding score.

Why: with the harness default (``FmriExtractor`` offset=0, aggregation="single",
window 0-3 s) even a trained AlexNet scores ~0.006 on NSD subj01 -- far below the
literature (r~0.3+). The BOLD response peaks ~5 s after onset (TR=1.333 s), so a
0-3 s window samples mostly pre-response baseline. ``FmriExtractor`` already
exposes the standard fix: ``offset`` (shift TRs to the HRF peak) + ``aggregation``
(mean over the response window) + its default z-score/detrend cleaning.

This sweeps the HRF ``offset`` (and window) with a frozen ImageNet AlexNet (fc7)
as the readout model, repeats averaged and the canonical shared-1000 held-out
split, reporting overall + ventral-stream ``test_corr`` per offset. The AlexNet
features are cached to disk so only the neural extraction varies across the sweep.

Run from the repo root::

    python3 scripts/nsd_signal_sweep.py
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
    _canonical_test_ids,
    _ensure_roi_label_file,
    _image_id,
    _load_workers,
    _nsd_subject,
    _roi_voxel_keep,
)
from harness.reward import _as_f64, _corr_per_column_t, _readout_device, _ridge_gcv_coef

_VENTRAL_LABELS = (2, 3)        # NSD "streams" atlas: midventral + ventral
_ROIS = ["V1", "hV4", "FFA", "PPA", "EBA", "VWFA"]
_FEAT_CACHE = os.path.join(
    os.environ.get("NEURO_SPLIT_CACHE",
                   os.path.join(os.path.expanduser("~"), ".cache", "neuro-autoresearch", "splits")),
    "alexnet_fc7_nsd_subj01.npz",
)


# --- AlexNet fc7 features (cached to disk) --------------------------------------
def alexnet_fc7(segments, device, batch_size=256):
    """{filepath: fc7 4096-d} for each unique image, cached on disk across runs."""
    by_path = {}
    for s in segments:
        fp = getattr(s.trigger, "filepath", None)
        if fp is not None and fp not in by_path:
            by_path[fp] = s.trigger
    paths = sorted(by_path)

    if os.path.exists(_FEAT_CACHE):
        z = np.load(_FEAT_CACHE, allow_pickle=True)
        cached = dict(zip(z["paths"], z["feats"]))
        if set(paths).issubset(cached):
            print(f"  AlexNet fc7: loaded {len(cached)} cached features from {_FEAT_CACHE}")
            return {p: cached[p] for p in paths}

    from torchvision.models import AlexNet_Weights, alexnet
    weights = AlexNet_Weights.IMAGENET1K_V1
    pre = weights.transforms()
    model = alexnet(weights=weights).to(device).eval()
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:6])  # -> fc7 4096
    print(f"  AlexNet fc7 on {len(paths)} unique images (frozen, {device}) ...")
    feats = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            imgs = torch.stack([pre(by_path[p].read().convert("RGB")) for p in chunk]).to(device)
            for p, f in zip(chunk, model(imgs).float().cpu().numpy()):
                feats[p] = f
    print(f"  features ready in {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(_FEAT_CACHE), exist_ok=True)
    np.savez(_FEAT_CACHE, paths=np.array(paths, dtype=object),
             feats=np.stack([feats[p] for p in paths]).astype("float32"))
    print(f"  cached features -> {_FEAT_CACHE}")
    return feats


# --- neural extraction at a given HRF offset / window ---------------------------
def extract_neural(events, d, offset, start, duration, aggregation="mean"):
    """Per-trial neural response (n_seg, n_voxels) + per-trial filepaths.

    Uses ``FmriExtractor`` with the standard cleaning (z-score + detrend), shifting
    the window by ``offset`` s to align the delayed BOLD and averaging it over the
    response window (``aggregation='mean'`` -> one value per voxel).
    """
    fmri = ns.extractors.FmriExtractor(aggregation=aggregation, offset=float(offset))
    seg = ns.Segmenter(extractors=dict(neural=fmri), trigger_query=d.trigger,
                       start=start, duration=duration, drop_incomplete=True)
    batch = seg.apply(events).load_all(num_workers=_load_workers())
    Y = batch.data["neural"].float().reshape(len(batch.segments), -1)
    fps = np.array([getattr(s.trigger, "filepath", None) for s in batch.segments], dtype=object)
    return Y, fps


def _collapse_mean(Y, fps):
    """Average Y over the repeats of each unique filepath. Returns (Y_g, uniq)."""
    uniq, inv = np.unique(fps, return_inverse=True)
    inv_t = torch.as_tensor(inv.reshape(-1), dtype=torch.long)
    counts = torch.bincount(inv_t, minlength=len(uniq)).clamp(min=1).unsqueeze(1).float()
    Y_g = torch.zeros(len(uniq), Y.shape[1]).index_add_(0, inv_t, Y.float()) / counts
    return Y_g, uniq


def _ventral_mask(study, n_voxels):
    import nibabel
    from neuralfetch.studies.allen2022massive import _get_roi_mask
    mask = _get_roi_mask(study._roi_path(_nsd_subject(study)))
    labels = nibabel.load(_ensure_roi_label_file(study, "streams")).get_fdata()
    return np.isin(labels[mask], _VENTRAL_LABELS)


def score(A, Y_g, uniq, shared, device, n_tr, masks):
    """Ridge readout (fit on all voxels) -> dict of slice -> test_corr."""
    is_test = np.array([_image_id(s) in shared for s in uniq])
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    A_t = _as_f64(torch.from_numpy(A), device)
    Y_t = _as_f64(Y_g, device)
    A_tr, A_te = A_t[tr], A_t[te]
    Y_tr, Y_te = Y_t[tr], Y_t[te]
    kept = (Y_tr.std(0) > 0).cpu().numpy()
    keep_t = torch.from_numpy(kept).to(device)
    Y_tr, Y_te = Y_tr[:, keep_t], Y_te[:, keep_t]
    x_off, y_off = A_tr.mean(0), Y_tr.mean(0)
    coef = _ridge_gcv_coef(A_tr - x_off, Y_tr - y_off)
    tvec = _corr_per_column_t((A_te - x_off) @ coef + y_off, Y_te).cpu().numpy()
    trc = float(_corr_per_column_t((A_tr - x_off) @ coef + y_off, Y_tr).mean())
    out = {"overall": float(tvec.mean()), "train": trc, "n_test": int(te.size)}
    for name, vox in masks.items():
        m = np.repeat(vox, n_tr)[kept]
        out[name] = float(tvec[m].mean()) if m.any() else float("nan")
    return out


def main():
    device = _readout_device()
    print(f"device: {device}")
    d = PRESETS["nsd"]
    study = ns.Study(name=d.study, path=d.path)
    print("building events (study.run) ...")
    events = study.run()

    seg0 = ns.Segmenter(extractors={}, trigger_query=d.trigger,
                        start=0.0, duration=d.duration, drop_incomplete=True)
    feats = alexnet_fc7(seg0.apply(events).segments, device)

    n_voxels = int(_roi_voxel_keep(study, "nsdgeneral")[1])
    masks = {"VENTRAL": _ventral_mask(study, n_voxels)}
    for roi in _ROIS:
        keep, _ = _roi_voxel_keep(study, roi)
        if keep is not None:
            masks[roi] = keep
    shared = _canonical_test_ids(study)

    # Sweep the HRF offset (mean over a 4 s response window). offset 0 ~ current
    # harness behaviour; the BOLD peak should land around offset 4-6 s.
    configs = [
        ("offset=5.0 dur=2.7  2TR-sep", 5.0, 0.0, 2.7, "mean", False),
        ("offset=5.0 dur=2.7  TR-avg",  5.0, 0.0, 2.7, "mean", True),
        ("offset=5.0 dur=4    TR-avg",  5.0, 0.0, 4.0, "mean", True),
        ("offset=4.0 dur=5.3  TR-avg",  4.0, 0.0, 5.3, "mean", True),
        ("offset=5.3 dur=1.4  1TR",     5.3, 0.0, 1.4, "mean", False),
    ]
    print(f"\n{'config':<28} {'overall':>9} {'VENTRAL':>9} {'V1':>8} {'hV4':>8} "
          f"{'FFA':>8} {'PPA':>8} {'train':>8} {'sec':>6}")
    print("-" * 104)
    for label, off, st, dur, agg, avg in configs:
        t0 = time.time()
        Y, fps = extract_neural(events, d, off, st, dur, agg)
        n_tr = Y.shape[1] // n_voxels
        Y_g, uniq = _collapse_mean(Y, fps)
        if avg and n_tr > 1:                 # average the peak TRs into one response/voxel
            Y_g = Y_g.reshape(Y_g.shape[0], n_voxels, n_tr).mean(-1)
            n_tr_eff = 1
        else:
            n_tr_eff = n_tr
        feats_missing = [p for p in uniq if p not in feats]
        if feats_missing:
            sys.exit(f"missing features for {len(feats_missing)} images")
        A = np.stack([feats[p] for p in uniq]).astype("float32")
        r = score(A, Y_g, uniq, shared, device, n_tr_eff, masks)
        print(f"{label:<28} {r['overall']:>9.4f} {r['VENTRAL']:>9.4f} {r['V1']:>8.4f} "
              f"{r['hV4']:>8.4f} {r['FFA']:>8.4f} {r['PPA']:>8.4f} {r['train']:>8.4f} "
              f"{time.time() - t0:>6.1f}")


if __name__ == "__main__":
    main()
