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
from harness.data import _load_workers, _roi_voxel_keep
from harness.reward import _readout_device

from _nsd_common import (
    alexnet_fc7,
    canonical_split,
    feature_matrix,
    fit_readout,
    region_masks,
)


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
    masks = region_masks(study)

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
            n_tr = 1
        A = feature_matrix(feats, uniq)
        tr, te = canonical_split(uniq, study)
        ro = fit_readout(A, Y_g, tr, te, device, n_tr)
        vent = ro.slice_corr(masks["VENTRAL"])[0]
        cols = " ".join(f"{ro.slice_corr(masks[r])[0]:>8.4f}"
                        for r in ("V1", "hV4", "FFA", "PPA"))
        print(f"{label:<28} {ro.overall:>9.4f} {vent:>9.4f} {cols} "
              f"{ro.train_corr:>8.4f} {time.time() - t0:>6.1f}")


if __name__ == "__main__":
    main()
