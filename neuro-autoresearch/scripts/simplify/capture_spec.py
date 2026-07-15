"""Capture the input_spec the behavior-preserving `simplify` check feeds to its synthetic-batch
generator (hub._synthetic_batch). Runs ONCE at scaffold time: loads the run's real split just to read
the stimulus tensor's shape / dtype / value range, then throws the data away. The equivalence check
itself never touches the dataset -- it rebuilds a fixed synthetic batch of THIS shape/dtype/range.

    python -m scripts.simplify.capture_spec <out_spec.json> [--batch 32] [--seed 0]

Emits {stim_shape, dtype, low, high, batch, seed}. NEURO_DATASET / NEURO_SUBJECT etc. must already be
exported (simplify.sh sources the run's env.json first), so the spec matches the scored split.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch


def capture(out_path: str, batch: int = 32, seed: int = 0) -> dict:
    from harness.datasets import DATASET
    from harness.loaders import load_split

    X_train, X_test, *_ = load_split(DATASET)
    X = X_train if torch.is_tensor(X_train) else torch.as_tensor(X_train)
    n = int(X.shape[0])
    b = min(batch, n)
    sample = X[:b]
    spec = {
        "stim_shape": [int(d) for d in sample.shape[1:]],
        "dtype": str(sample.dtype).replace("torch.", ""),
        "low": float(sample.float().min()),
        "high": float(sample.float().max()),
        "batch": b,
        "seed": seed,
    }
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)
    return spec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    spec = capture(a.out, a.batch, a.seed)
    json.dump(spec, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
