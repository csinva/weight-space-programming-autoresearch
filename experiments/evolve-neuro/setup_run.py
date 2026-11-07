"""
Create a fresh run folder for an fMRI-encoding autoresearch experiment, seeding
it with the GPT-2 XL baseline result only when that baseline is missing.

Usage:
    uv run setup_run.py <tag>                       # runs baseline iff missing
    uv run setup_run.py <tag> --num-train 5 --device cuda
    uv run setup_run.py <tag> --skip-baseline       # never run GPT-2 XL
    uv run setup_run.py <tag> --force-baseline      # (re)run GPT-2 XL regardless

Result:
    Creates  <repo_root>/runs-neuro/<tag>/  containing:
      src/                                    → symlink to evolve-neuro/src
      interpretable_transformer.py            → fresh local copy (the file you edit)
      results/                                → results dir; seeded with the
                                                GPT-2 XL baseline row + plot if the
                                                baseline was missing
      interpretable_transformers_lib/         → empty dir, for snapshots

The baseline is run only if a `GPT2XL-baseline` row for the subject is not already
in `results/overall_results.csv`. It uses the GPT-2 XL embedder (`src/baseline.py`)
through the exact same encoding pipeline the interpretable transformer uses, so
the test correlations are directly comparable. The agent then `cd`s into the run
folder and iterates on `interpretable_transformer.py`.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

SYMLINK_NAMES = ["src"]
COPY_NAMES = ["interpretable_transformer.py", "results", "program.md"]


def setup_run(tag: str, repo_root: str | None = None) -> str:
    evolve_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_root is None:
        repo_root = os.path.dirname(evolve_dir)

    run_dir = os.path.join(repo_root, "runs-neuro", tag)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run folder already exists: {run_dir}")
    os.makedirs(run_dir)

    for name in SYMLINK_NAMES:
        src_path = os.path.join(evolve_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        os.symlink(os.path.relpath(src_path, run_dir), os.path.join(run_dir, name))

    for name in COPY_NAMES:
        src_path = os.path.join(evolve_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(run_dir, name))
        else:
            shutil.copy2(src_path, os.path.join(run_dir, name))

    os.makedirs(os.path.join(run_dir, "interpretable_transformers_lib"))
    return run_dir


def run_baseline(run_dir: str, subject: str, num_train: int, num_test: int,
                 layer: int, device: str) -> None:
    """Run the GPT-2 XL baseline and write its row into the run's results."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.baseline import GPT2Embedder
    from src.eval import (
        EncodingConfig, run_encoding, make_result_row,
        upsert_overall_results, plot_corr_over_iterations,
    )

    print("\n=== running GPT-2 XL baseline ===")
    cfg = EncodingConfig(subject=subject, num_train=num_train, num_test=num_test)
    embedder = GPT2Embedder(layer=layer, device=device)
    r = run_encoding(embedder, cfg)
    n_params = sum(p.numel() for p in embedder.model.parameters())

    results_dir = os.path.join(run_dir, "results")
    description = f"GPT-2 XL layer {layer} final-token 10-gram embeddings (pretrained baseline)."
    upsert_overall_results(
        [make_result_row(r, "GPT2XL-baseline", n_params, description)], results_dir)
    plot_corr_over_iterations(results_dir)
    print(f"baseline test_corr: {r['test_corr']:.4f}")


def baseline_present(results_dir: str, subject: str) -> bool:
    """True if a GPT2XL-baseline row for `subject` is already in overall_results.csv."""
    path = os.path.join(results_dir, "overall_results.csv")
    if not os.path.exists(path):
        return False
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("model_shorthand_name") == "GPT2XL-baseline"
                    and row.get("subject") == subject):
                return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="Run tag, e.g. 'fmri-may27-run1'.")
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=3)
    parser.add_argument("--layer", type=int, default=24, help="GPT-2 XL layer for the baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="never run the GPT-2 XL baseline, even if it is missing")
    parser.add_argument("--force-baseline", action="store_true",
                        help="(re)run the GPT-2 XL baseline even if it is already present")
    args = parser.parse_args()

    try:
        run_dir = setup_run(args.tag)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Created run folder: {run_dir}")

    # By default skip the (slow) baseline; only run it when it is missing from
    # the run's overall_results.csv (e.g. a fresh run folder). --force-baseline
    # re-runs it; --skip-baseline never runs it.
    results_dir = os.path.join(run_dir, "results")
    if args.skip_baseline:
        print("skipping GPT-2 XL baseline (--skip-baseline)")
    elif args.force_baseline or not baseline_present(results_dir, args.subject):
        run_baseline(run_dir, args.subject, args.num_train, args.num_test, args.layer, args.device)
    else:
        print(f"GPT-2 XL baseline already present for {args.subject} in overall_results.csv — skipping")

    print()
    print("Next steps:")
    print(f"  cd {run_dir}")
    print(f"  uv run interpretable_transformer.py --subject {args.subject} "
          f"--num-train {args.num_train}  # one experiment iteration")
