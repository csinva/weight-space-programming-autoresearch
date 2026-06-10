"""The fixed driver: evaluate the current model and record the reward.

Run from the repo root::

    python -m harness.run

Loads the active dataset split ONCE (the slow step), then runs ``ITERATIONS``
evaluation passes of the model in ``model.py``, printing each and appending it to
``results/<study>.csv``. The active dataset is chosen by ``NEURO_DATASET`` (see
``harness/datasets.py``); the run-config (``ITERATIONS``, ``LOG_SLICES``) lives in
``experiment.py``. The agent should NOT edit this file.
"""
from __future__ import annotations

import time

from experiment import ITERATIONS, LOG_SLICES
from model import MODEL
from harness.datasets import DATASET
from harness.data import load_split
from harness.reward import evaluate_on_split
from harness.results import save_results


def _print_row(prefix: str, row: dict) -> None:
    print(f"{prefix}{row['model_name']:<18} {row['n_params']:>8}  "
          f"{row['train_corr']:>10.4f}  {row['test_corr']:>9.4f}  "
          f"{row['frac_above_0.1']:>8.4f}  {row['seconds']:>5.2f}")


def main() -> None:
    print(f"Dataset: {DATASET.study}  stimulus={DATASET.stimulus}  neural={DATASET.neural}"
          f"  reward_roi={DATASET.reward_roi}")
    print("Loading split once (the slow step) ...")
    t = time.time()
    split = load_split(DATASET)
    Xtr, Xte, Ytr, Yte = split
    print(f"  X_train{tuple(Xtr.shape)} X_test{tuple(Xte.shape)} "
          f"Y_train{tuple(Ytr.shape)} Y_test{tuple(Yte.shape)}  "
          f"(load {time.time() - t:.1f}s)\n")

    header = (f"{'iter':>4}  {'model':<18} {'n_params':>8}  {'train_corr':>10}  "
              f"{'test_corr':>9}  {'frac>0.1':>8}  {'sec':>5}")
    print(header)
    print("-" * len(header))
    for i in range(1, ITERATIONS + 1):
        row = evaluate_on_split(MODEL, split, DATASET, LOG_SLICES)
        save_results(row)
        _print_row(f"{i:>4}  ", row)


if __name__ == "__main__":
    main()
