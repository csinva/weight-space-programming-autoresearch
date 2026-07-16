#!/usr/bin/env python3
"""Complementarity probe: is the searched model's representation complementary to AlexNet?

For NSD subject 1, in TWO brain regions (early visual, ventral visual), compare three feature
sources -- each sparse-random-projected to a common width (default 2048) for a fair comparison,
then scored through the EXACT harness readout (fast/himalaya ridge, per-voxel noise ceiling):

    early region :  AlexNet conv4   |  searched champion  |  conv4 (+) champion
    ventral region:  AlexNet fc7     |  searched champion  |  fc7   (+) champion

A single region-matched AlexNet layer is used on purpose (NOT the all-layers concat baseline): conv4
for early, fc7 (penultimate, pre-classifier -- where AlexNet's late-layer semantic selectivity lives)
for ventral. If Combined > max(AlexNet-layer, champion), the champion carries variance AlexNet does
not; the pattern ACROSS regions says what kind.

AlexNet gets its native ImageNet preprocessing (resize 224 + ImageNet mean/std) via AlexNetSRP.
Everything is projected to the same 2048 dims (champion + each AlexNet layer), so Combined is a clean
2*2048 concat -- no source wins purely by having more columns. Nothing under harness/ is modified.

    python experiments/complementarity.py                    # nsd subj1, champion qq145
    python experiments/complementarity.py --subject 1 --champion runs/.../models/qq145.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "evaluation"))

DEFAULT_CHAMPION = os.path.join(REPO, "runs", "nsd__2026-06-27", "models", "qq145.py")


def load_model(path):
    """Import a model .py and return its MODEL instance."""
    spec = importlib.util.spec_from_file_location("champ_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def srp_to(feats_np, dim, seed, cache_dir):
    """Sparse-random-project (N, D) -> (N, dim). D<=dim is returned unchanged (nothing to reduce)."""
    from srp import get_srp_transformer
    D = feats_np.shape[1]
    if D <= dim:
        return feats_np.astype(np.float32)
    srp = get_srp_transformer(D, dim, None, seed, cache_dir=os.path.join(HERE, "results", "srp_cache"))
    return srp.transform(feats_np).astype(np.float32)


def main(argv=None):
    """Score AlexNet-layer, champion, and their concat in early + ventral regions and report whether
    Combined beats the best single source (champion carries variance AlexNet misses); print + save."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="1", help="NSD subject id (default 1)")
    p.add_argument("--champion", default=DEFAULT_CHAMPION, help="searched champion model .py")
    p.add_argument("--dim", type=int, default=2048, help="common SRP width for every source (default 2048)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    os.environ["NEURO_DATASET"] = "nsd"
    os.environ["NEURO_SUBJECT"] = str(a.subject)
    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness import reward as R
    from harness.roi import roi_breakdown
    from vision_baselines import extract_alexnet
    from score import readout

    print(f"=== complementarity | nsd subj={a.subject} | champion={os.path.basename(a.champion)} "
          f"| SRP dim={a.dim} | reward_roi={DATASET.reward_roi} ===")

    split = load_split(DATASET)
    Xtr, Xte, Ytr, Yte, roi_masks, nc = split
    device = R._readout_device()
    if torch.is_tensor(Xtr):
        Xtr, Xte = Xtr.to(device), Xte.to(device)
    print(f"  n_train={Ytr.shape[0]} n_test={Yte.shape[0]} n_targets={Ytr.shape[1]}  "
          f"regions={sorted(k for k in roi_masks if np.asarray(roi_masks[k]).any())}")

    srp_cache = os.path.join(HERE, "results", "srp_cache")

    # --- AlexNet conv4 (early) and fc7 (ventral), each SRP'd to `dim` (reused eval.vision extractor) ---
    t0 = time.time()
    alex_tr = extract_alexnet(Xtr, DATASET.imsize, a.dim, a.seed, device, pretrained=True)
    alex_te = extract_alexnet(Xte, DATASET.imsize, a.dim, a.seed, device, pretrained=True)
    conv4_tr, conv4_te = alex_tr["conv4"], alex_te["conv4"]
    fc7_tr, fc7_te = alex_tr["fc7"], alex_te["fc7"]
    print(f"  AlexNet conv4/fc7 forward+SRP: {time.time()-t0:.1f}s  "
          f"(conv4->{conv4_tr.shape[1]}, fc7->{fc7_tr.shape[1]})")

    # --- searched champion, SRP'd to `dim` ---
    t0 = time.time()
    champ = load_model(a.champion)
    ch_tr = champ(Xtr).detach().float().cpu().numpy()
    ch_te = champ(Xte).detach().float().cpu().numpy()
    D_champ = ch_tr.shape[1]
    ch_tr = srp_to(ch_tr, a.dim, a.seed, srp_cache)
    ch_te = srp_to(ch_te, a.dim, a.seed, srp_cache)
    print(f"  champion forward (D_out={D_champ}) + SRP->{ch_tr.shape[1]}: {time.time()-t0:.1f}s")

    def standardize(A_tr, A_te):
        """Per-column z-score with TRAIN stats. The harness ridge centers but does NOT scale, and
        uses ONE alpha per voxel across all columns -- so without this, concatenating two blocks of
        different feature scale (AlexNet post-ReLU SRP vs the champion's RMS-balanced features) lets
        the larger-norm block dominate the shared penalty and Combined can score BELOW either single
        source. Standardizing every column to unit variance puts all sources on equal footing; applied
        uniformly to single and combined reps so the comparison stays apples-to-apples."""
        mu = A_tr.mean(0, keepdims=True)
        sd = A_tr.std(0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        return ((A_tr - mu) / sd).astype(np.float32), ((A_te - mu) / sd).astype(np.float32)

    def score(A_tr, A_te):
        """Standardize columns then run the harness readout; return the per-ROI record."""
        A_tr, A_te = standardize(A_tr, A_te)
        rec = readout(torch.from_numpy(A_tr).to(device), torch.from_numpy(A_te).to(device),
                      Ytr, Yte, dict(roi_masks), nc, DATASET, R, roi_breakdown, device)
        return rec

    # Score each source once; ridge fits each voxel independently, so the per-ROI number equals a
    # region-only readout. Combined = 2*dim concat.
    def cat(a_np, b_np):
        return np.concatenate([a_np, b_np], axis=1)

    sources = {
        "champion":     score(ch_tr, ch_te),
        "conv4":        score(conv4_tr, conv4_te),
        "fc7":          score(fc7_tr, fc7_te),
        "conv4+champ":  score(cat(conv4_tr, ch_tr), cat(conv4_te, ch_te)),
        "fc7+champ":    score(cat(fc7_tr, ch_tr), cat(fc7_te, ch_te)),
    }

    def roi_test(name, region):
        """A source's noise-normalized test score in one region (for the per-region bar plots)."""
        return round(float(sources[name]["roi_corr"][region]["test"]), 4)

    plots = {
        "early":   {"AlexNet conv4": roi_test("conv4", "early"),
                    "Champion":      roi_test("champion", "early"),
                    "Combined":      roi_test("conv4+champ", "early")},
        "ventral": {"AlexNet fc7":   roi_test("fc7", "ventral"),
                    "Champion":      roi_test("champion", "ventral"),
                    "Combined":      roi_test("fc7+champ", "ventral")},
    }

    print()
    for region, bars in plots.items():
        keys = list(bars)
        alex, champ_v, comb = bars[keys[0]], bars[keys[1]], bars[keys[2]]
        print(f"  [{region:>7}]  {keys[0]}={alex:.4f}  Champion={champ_v:.4f}  Combined={comb:.4f}   "
              f"(combined - best single = {comb - max(alex, champ_v):+.4f})")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "results", f"complementarity_nsd_subj{a.subject}_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": "nsd", "subject": a.subject,
                   "champion": os.path.basename(a.champion), "d_champ": int(D_champ),
                   "srp_dim": a.dim, "plots": plots,
                   "full_roi": {k: v["roi_corr"] for k, v in sources.items()}}, f, indent=2)
    print(f"\n  saved -> {out}")
    return plots, out


if __name__ == "__main__":
    main()
