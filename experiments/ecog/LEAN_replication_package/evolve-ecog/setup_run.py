"""
Create a fresh run folder for a melspec -> ECoG high-gamma autoresearch
experiment (patient patient1), seeding it with the linear ridge baseline result
only when that baseline is missing.

Usage:
    python setup_run.py <tag>                    # runs baseline iff missing
    python setup_run.py <tag> --num-train 60 --device cuda
    python setup_run.py <tag> --skip-baseline    # never run the ridge baseline
    python setup_run.py <tag> --force-baseline   # (re)run the ridge baseline

Result:
    Creates  <repo_root>/runs-ecog/<tag>/  containing:
      src/                              → symlink to evolve-ecog/src (do not edit)
      interpretable_transformer.py      → fresh local copy (the file you edit)
      program.md                        → local copy of the instructions
      results/                          → results dir; seeded with the linear ridge
                                          baseline row + plot if it was missing
      interpretable_transformers_lib/   → empty dir, for snapshots

The baseline (identity melspec embedder + ridge readout) is run through the exact
same encoding pipeline the interpretable transformer uses, so test correlations
are directly comparable. The agent then `cd`s into the run folder and iterates on
`interpretable_transformer.py`.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

SYMLINK_NAMES = ["src"]
COPY_FILES = ["interpretable_transformer.py", "program.md"]

BASELINE_NAME = "Ridge-melspec-baseline"


def setup_run(tag: str, repo_root: str | None = None) -> str:
    evolve_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_root is None:
        repo_root = os.path.dirname(evolve_dir)

    run_dir = os.path.join(repo_root, "runs-ecog", tag)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run folder already exists: {run_dir}")
    os.makedirs(run_dir)

    for name in SYMLINK_NAMES:
        src_path = os.path.join(evolve_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        os.symlink(os.path.relpath(src_path, run_dir), os.path.join(run_dir, name))

    for name in COPY_FILES:
        src_path = os.path.join(evolve_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source file: {src_path}")
        shutil.copy2(src_path, os.path.join(run_dir, name))

    os.makedirs(os.path.join(run_dir, "results"))
    os.makedirs(os.path.join(run_dir, "interpretable_transformers_lib"))
    return run_dir


def run_baseline(run_dir: str, num_train: int, num_test: int, device: str) -> None:
    """Run the linear ridge baseline and write its row into the run's results."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.baseline import MelspecIdentityEmbedder
    from src.eval import (
        EncodingConfig, run_encoding, make_result_row,
        upsert_overall_results, plot_corr_over_iterations,
    )

    print("\n=== running linear ridge baseline (melspec → ECoG) ===")
    cfg = EncodingConfig(num_train=num_train, num_test=num_test)
    embedder = MelspecIdentityEmbedder()
    r = run_encoding(embedder, cfg)

    results_dir = os.path.join(run_dir, "results")
    description = ("Identity melspec embedder + FIR-delayed ridge: linear ridge "
                   "regression from the 50 mel bins to 166 electrodes (no model "
                   "params, none trained).")
    upsert_overall_results(
        [make_result_row(r, BASELINE_NAME, embedder.n_params, description)], results_dir)
    plot_corr_over_iterations(results_dir)
    print(f"baseline test_corr: {r['test_corr']:.4f}")


def baseline_present(results_dir: str) -> bool:
    path = os.path.join(results_dir, "overall_results.csv")
    if not os.path.exists(path):
        return False
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model_shorthand_name") == BASELINE_NAME:
                return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="Run tag, e.g. 'ecog-jun01-run1'.")
    parser.add_argument("--num-train", type=int, default=60)
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="never run the ridge baseline, even if it is missing")
    parser.add_argument("--force-baseline", action="store_true",
                        help="(re)run the ridge baseline even if it is already present")
    args = parser.parse_args()

    try:
        run_dir = setup_run(args.tag)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Created run folder: {run_dir}")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available — falling back to cpu")
        args.device = "cpu"

    results_dir = os.path.join(run_dir, "results")
    if args.skip_baseline:
        print("skipping ridge baseline (--skip-baseline)")
    elif args.force_baseline or not baseline_present(results_dir):
        run_baseline(run_dir, args.num_train, args.num_test, args.device)
    else:
        print(f"ridge baseline already present in overall_results.csv — skipping")

    print()
    print("Next steps:")
    print(f"  cd {run_dir}")
    print(f"  python interpretable_transformer.py --num-train {args.num_train} "
          f"--num-test {args.num_test}  # one experiment iteration")
