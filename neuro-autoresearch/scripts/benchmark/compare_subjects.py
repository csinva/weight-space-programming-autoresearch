#!/usr/bin/env python3
"""Apples-to-apples champion-vs-AlexNet comparison, reusable across NSD subjects.

``compare()`` loads ONE harness split (per subject) and scores both models through the identical
fixed eval -- same images, same NC_r>=0.2 SNR voxel selection, same fast ridge, same noise-normalized
r^2/NC_r^2 metric, same per-ROI breakdown:

  * the search champion (any model .py defining ``MODEL``) via ``harness.reward.evaluate_on_split``;
  * a frozen ImageNet-AlexNet, scored layer-by-layer, each ROI keeping its BEST layer (the brain-score
    "oracle" protocol) -- reusing ``alexnet_baseline.AlexNetSRP`` + ``_readout``.

It returns one comparison dict per subject (per-ROI champ / alex-best / delta + the overall champion
reward and the voxel-weighted oracle-AlexNet reward over the stream ROIs that partition the region).

    python scripts/benchmark/compare_subjects.py --subjects 1 2 5          # default model = qq145
    python scripts/benchmark/compare_subjects.py --model runs/.../foo.py --subjects 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from alexnet_baseline import AlexNetSRP, _LAYERS, _readout
from eval_champion import load_model

DEFAULT_MODEL = f"{REPO}/runs/nsd__2026-06-28/models/qq145.py"
STREAMS = ("early", "ventral")          # the ROIs that partition the NSD reward region


def _roi_counts(roi_masks, Ytr, DATASET, R):
    """Reliable-voxel count per ROI within the reward region (same slicing as ``_readout``)."""
    mask = R.reward_region_mask(roi_masks, DATASET.reward_roi)
    rm = ({k: np.asarray(v, bool)[mask] for k, v in roi_masks.items()} if mask is not None
          else {k: np.asarray(v, bool) for k, v in roi_masks.items()})
    return {k: int(v.sum()) for k, v in rm.items()}


def compare(model_path=DEFAULT_MODEL, layers=tuple(_LAYERS)):
    """Score champion + per-layer AlexNet on the active dataset/subject; return the comparison dict."""
    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness.reward import evaluate_on_split
    from harness.roi import roi_breakdown
    from harness import reward as R

    split = load_split(DATASET)
    Xtr, Xte, Ytr, Yte, roi_masks, nc = split
    device = R._readout_device()

    # --- champion (the search model) ---
    champ_row = evaluate_on_split(load_model(model_path), split, DATASET)
    champ = {roi: v["test"] for roi, v in champ_row["roi_corr"].items()}

    # --- frozen AlexNet, per layer ---
    model = AlexNetSRP(DATASET.imsize, list(layers))
    if torch.is_tensor(Xtr):
        Xtr, Xte = Xtr.to(device), Xte.to(device)
    ftr, fte = model.forward_layers(Xtr), model.forward_layers(Xte)
    alex = {}                                                   # layer -> {roi: score}
    for L in layers:
        rec = _readout(torch.from_numpy(ftr[L]).to(device), torch.from_numpy(fte[L]).to(device),
                       Ytr, Yte, dict(roi_masks), nc, DATASET, R, roi_breakdown, device)
        alex[L] = {roi: v["test"] for roi, v in rec["roi_corr"].items()}

    # --- best layer per ROI (oracle) + deltas ---
    counts = _roi_counts(roi_masks, Ytr, DATASET, R)
    rois = {}
    for roi in champ:
        best = max(layers, key=lambda L: alex[L].get(roi, -9.0))
        rois[roi] = {"champ": round(champ[roi], 5), "alex_best": round(alex[best][roi], 5),
                     "alex_layer": best, "delta": round(champ[roi] - alex[best][roi], 5),
                     "voxels": counts.get(roi), "perlayer": {L: round(alex[L][roi], 5) for L in layers}}

    # overall: champion = harness reward; AlexNet oracle = voxel-weighted over the partition streams.
    streams = [s for s in STREAMS if counts.get(s)]
    wsum = sum(counts[s] for s in streams) or 1
    oracle = sum(counts[s] * rois[s]["alex_best"] for s in streams) / wsum
    return {"subject": os.environ.get("NEURO_SUBJECT", "1"), "model": os.path.basename(model_path),
            "champ_overall": round(champ_row["test_corr"], 5),
            "alex_oracle_overall": round(oracle, 5), "rois": rois, "counts": counts}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="nsd")
    p.add_argument("--subjects", nargs="+", default=["1"])
    a = p.parse_args(argv)
    os.environ["NEURO_DATASET"] = a.dataset

    results = []
    for subj in a.subjects:
        os.environ["NEURO_SUBJECT"] = str(subj)
        # the harness binds the dataset/subject at import -> reload so NEURO_SUBJECT takes effect.
        for m in [k for k in list(sys.modules) if k.startswith("harness")]:
            del sys.modules[m]
        t0 = time.time()
        res = compare(a.model)
        res["seconds"] = round(time.time() - t0, 1)
        results.append(res)
        e, v = res["rois"]["early"], res["rois"]["ventral"]
        print(f"\n=== subj{subj} | {res['model']} | {res['seconds']}s ===")
        print(f"  overall: champion {res['champ_overall']:.4f}  vs  AlexNet oracle {res['alex_oracle_overall']:.4f}")
        print(f"  early : champ {e['champ']:.4f}  alex {e['alex_best']:.4f} ({e['alex_layer']})  Δ{e['delta']:+.4f}")
        print(f"  ventral: champ {v['champ']:.4f}  alex {v['alex_best']:.4f} ({v['alex_layer']})  Δ{v['delta']:+.4f}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "results", f"compare_{a.dataset}_subj{'-'.join(a.subjects)}_{stamp}.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")
    return results


if __name__ == "__main__":
    main()
