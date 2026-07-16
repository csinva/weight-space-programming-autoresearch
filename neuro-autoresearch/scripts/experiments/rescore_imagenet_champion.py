#!/usr/bin/env python3
"""Re-score the NSD champion on NSD, UNTRAINED vs. with its ImageNet-trained ventral weights.

The champion (`qq145`) is a hand-written HMAX model whose only free weights are the two random He
conv layers `W2`/`W3` (the "ventral" branch); the Gabor V1 front-end is fixed. `train_champion_imagenet.py`
trained `W2`/`W3` on ImageNet (→ `champion_imagenet.pt`). This loads BOTH versions of the model and
scores each through the EXACT NSD harness (same loader / ridge / metric the search used), so the only
difference is the `W2`/`W3` weights. It prints the per-ROI test_corr for each and the delta, to test
whether object-classification training specifically lifts the high-level / category-selective ROIs
(FFA/VWFA/PPA/ventral) while early vision (V1–V3) — driven by the unchanged front-end — stays flat.

    NEURO_SUBJECT=1,2 python scripts/experiments/rescore_imagenet_champion.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CHAMPION = os.path.join(REPO, "runs", "nsd__2026-06-27", "models", "qq145.py")
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_imagenet.pt")

os.environ["NEURO_DATASET"] = "nsd"
os.environ.setdefault("NEURO_SUBJECT", "1,2")   # the search reward: mean over subjects 1,2
# The champion predates the 100K search cap (~1.83M params); this is an analysis re-score, not a
# search proposal, so lift the cap here (read at harness import time) to let it through.
os.environ.setdefault("NEURO_MAX_PARAMS", "100000000")


def fresh_champion():
    """A fresh champion instance (random He W2/W3, as built in the source)."""
    spec = importlib.util.spec_from_file_location("champion", CHAMPION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def score(model):
    """Score one model through the fixed NSD harness; return the mean row (test_corr + roi_corr)."""
    from harness.datasets import DATASET
    from harness.reward import evaluate_subjects
    return evaluate_subjects(model, DATASET)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== re-score champion qq145 on NSD (subjects {os.environ['NEURO_SUBJECT']}, device={device}) ===\n")

    # (a) UNTRAINED: the champion exactly as searched (random W2/W3).
    print("scoring UNTRAINED champion (random W2/W3) ...")
    untrained = score(fresh_champion())

    # (b) TRAINED: same champion, W2/W3 replaced by the ImageNet-trained weights. Move the model to
    # the device FIRST (sets _dev), THEN overwrite W2/W3 so the forward's _to() no-ops and keeps them.
    print("scoring IMAGENET-TRAINED champion (trained W2/W3) ...")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = fresh_champion()
    m._to(device)
    m.W2 = ck["W2"].to(device)
    m.W3 = ck["W3"].to(device)
    trained = score(m)

    # --- report ---------------------------------------------------------------------------------
    def roi_test(row):
        return {r: v["test"] for r, v in (row.get("roi_corr") or {}).items()}
    ru, rt = roi_test(untrained), roi_test(trained)

    # early stream first (front-end driven), then the high-level category-selective ROIs.
    order = ["early", "V1", "V2", "V3", "hV4", "ventral", "hV4", "FFA", "PPA", "VWFA"]
    seen = set()
    rois = [r for r in order if r in ru and not (r in seen or seen.add(r))]
    rois += [r for r in ru if r not in seen]

    print(f"\n{'ROI':<10}{'untrained':>11}{'trained':>10}{'Δ':>9}")
    print("-" * 40)
    for r in rois:
        u, t = ru[r], rt.get(r, float("nan"))
        print(f"{r:<10}{u:>11.4f}{t:>10.4f}{t - u:>+9.4f}")
    print("-" * 40)
    ou, ot = untrained["test_corr"], trained["test_corr"]
    print(f"{'OVERALL':<10}{ou:>11.4f}{ot:>10.4f}{ot - ou:>+9.4f}   (reward = early+ventral)")
    print(f"\nchampion recorded search reward: 0.4407 (untrained re-score above should match)")


if __name__ == "__main__":
    main()
