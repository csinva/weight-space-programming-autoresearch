#!/usr/bin/env python3
"""Re-evaluate a search-DB champion through the EXACT harness reward, for an apples-to-apples
comparison against ``alexnet_baseline.py``.

Loads a model file by path (its module must define ``MODEL``), then runs
``harness.reward.evaluate_on_split`` on the same ``load_split(DATASET)`` the AlexNet baseline and the
search itself use -- same images, same shared-1000 hold-out, same early+ventral target voxels, same
per-voxel noise ceiling (``ncsnr``) reliability cut, same per-ROI breakdown. So the printed
``test_corr`` + per-ROI ``roi_corr`` are directly comparable to the AlexNet per-layer numbers.

    python scripts/benchmark/eval_champion.py runs/nsd__2026-06-28/models/qq145.py
    NEURO_SUBJECT=2 python scripts/benchmark/eval_champion.py <model.py>
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)


def load_model(path):
    spec = importlib.util.spec_from_file_location("champion_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="path to a model .py defining MODEL")
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "nsd"))
    a = p.parse_args(argv)

    os.environ["NEURO_DATASET"] = a.dataset
    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness.reward import evaluate_on_split

    subj = os.environ.get("NEURO_SUBJECT", "1")
    print(f"=== champion re-eval | {os.path.basename(a.model)} | {a.dataset} subj={subj} "
          f"| reward_roi={DATASET.reward_roi} | imsize={DATASET.imsize} ===")

    model = load_model(a.model)
    split = load_split(DATASET)
    t0 = time.time()
    row = evaluate_on_split(model, split, DATASET)
    print(f"  eval {time.time() - t0:.1f}s")
    print(f"\n  REWARD r²/NC²  test={row['test_corr']:.4f}  train={row['train_corr']:.4f}  "
          f"gap={row['train_corr'] - row['test_corr']:.4f}  "
          f"(n_voxels={row['n_targets']}, n_params={row['n_params']})")
    rc = row["roi_corr"]
    rois = list(rc)
    print("  " + "".join(f"{r:>9}" for r in rois))
    print("  r²/NC²  " + "".join(f"{rc[r]['test']:>9.4f}" for r in rois))


if __name__ == "__main__":
    main()
