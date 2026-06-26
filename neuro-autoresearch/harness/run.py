"""The fixed driver: evaluate the current model and record the reward.

Run from the repo root::

    python -m harness.run

Loads the active dataset split ONCE (the slow step), then evaluates the model in
``model.py`` exactly once and prints the scored row. A model that raises or returns a
bad shape does not crash the run: it is reported as a ``status='error'`` row (NaN
reward) and the run exits cleanly. The active dataset is chosen by ``NEURO_DATASET``
(see ``harness/datasets.py``). This is the manual eval path; the unattended search
records results into a run's ``search.db`` (see ``hub.py``), not here.
The agent should NOT edit this file.
"""
from __future__ import annotations

import time

from model import MODEL
from harness.datasets import DATASET
from harness.loaders import load_split
from harness.reward import evaluate_on_split_safe


def _print_row(row: dict) -> None:
    print(f"{row['model_name']:<18} {row['n_params']:>8}  "
          f"{row['train_corr']:>10.4f}  {row['test_corr']:>9.4f}  {row['seconds']:>5.2f}")


def main() -> None:
    print(f"Dataset: {DATASET.study}  stimulus={DATASET.stimulus}  neural={DATASET.neural}"
          f"  reward_roi={DATASET.reward_roi}")
    print("Loading split once (the slow step) ...")
    t = time.time()
    split = load_split(DATASET)
    Xtr, Xte, Ytr, Yte, _, _ = split
    print(f"  X_train{tuple(Xtr.shape)} X_test{tuple(Xte.shape)} "
          f"Y_train{tuple(Ytr.shape)} Y_test{tuple(Yte.shape)}  "
          f"(load {time.time() - t:.1f}s)\n")

    header = (f"{'model':<18} {'n_params':>8}  {'train_corr':>10}  "
              f"{'test_corr':>9}  {'sec':>5}")
    print(header)
    print("-" * len(header))
    row = evaluate_on_split_safe(MODEL, split, DATASET)
    _print_row(row)
    if row["status"] != "ok":
        print(f"\n{row['status']}\n  -> recorded as a failed attempt (NaN reward); "
              f"fix model.py and re-run.")


if __name__ == "__main__":
    main()
