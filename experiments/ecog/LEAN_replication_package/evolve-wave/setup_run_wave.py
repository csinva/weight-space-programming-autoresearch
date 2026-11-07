"""
Create a fresh run folder for a WAVEFORM -> ECoG high-gamma autoresearch
experiment (patient1), seeding it with the waveform 'reproduce-the-melspec' floor
baseline when missing.

Usage:
    python setup_run_wave.py <tag>                 # runs floor baseline iff missing
    python setup_run_wave.py <tag> --num-train 500 --num-test 100 --device cuda
    python setup_run_wave.py <tag> --skip-baseline

Creates  <repo_root>/runs-wave/<tag>/  with:
    src/                       → symlink to evolve-wave/src (do not edit)
    wave_encoder.py            → fresh local copy (the file you edit)
    program.md                 → local copy of the instructions
    results/                   → seeded with the Wave-melspec-floor row
    wave_encoders_lib/         → empty dir, for snapshots
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

SYMLINK_NAMES = ["src"]
COPY_FILES = ["wave_encoder.py", "program.md"]
BASELINE_NAME = "Wave-melspec-floor"


def setup_run(tag: str, repo_root: str | None = None) -> str:
    evolve_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_root is None:
        repo_root = os.path.dirname(evolve_dir)
    run_dir = os.path.join(repo_root, "runs-wave", tag)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run folder already exists: {run_dir}")
    os.makedirs(run_dir)
    for name in SYMLINK_NAMES:
        src_path = os.path.join(evolve_dir, name)
        os.symlink(os.path.relpath(src_path, run_dir), os.path.join(run_dir, name))
    for name in COPY_FILES:
        shutil.copy2(os.path.join(evolve_dir, name), os.path.join(run_dir, name))
    os.makedirs(os.path.join(run_dir, "results"))
    os.makedirs(os.path.join(run_dir, "wave_encoders_lib"))
    return run_dir


def run_baseline(run_dir: str, num_train: int, num_test: int, device: str) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.baseline import WaveMelReproEmbedder
    from src.eval import (
        EncodingConfig, run_encoding, make_result_row,
        upsert_overall_results, plot_corr_over_iterations,
    )
    print("\n=== running waveform floor baseline (wave -> log-mel -> ridge) ===")
    cfg = EncodingConfig(num_train=num_train, num_test=num_test)
    embedder = WaveMelReproEmbedder(device=device)
    r = run_encoding(embedder, cfg)
    results_dir = os.path.join(run_dir, "results")
    desc = ("Reproduce a log-mel from the raw audio windows (Hann->rFFT->mel->dB) + "
            "ridge. Closed-form, no training. Waveform floor / alignment sanity check.")
    upsert_overall_results(
        [make_result_row(r, BASELINE_NAME, embedder.n_params, desc)], results_dir)
    plot_corr_over_iterations(results_dir)
    print(f"floor baseline test_corr: {r['test_corr']:.4f}")


def baseline_present(results_dir: str) -> bool:
    path = os.path.join(results_dir, "overall_results.csv")
    if not os.path.exists(path):
        return False
    with open(path, newline="") as f:
        return any(row.get("model_shorthand_name") == BASELINE_NAME for row in csv.DictReader(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag")
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--force-baseline", action="store_true")
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
        print("skipping floor baseline (--skip-baseline)")
    elif args.force_baseline or not baseline_present(results_dir):
        run_baseline(run_dir, args.num_train, args.num_test, args.device)
    else:
        print("floor baseline already present — skipping")

    print()
    print("Next steps:")
    print(f"  cd {run_dir}")
    print(f"  python wave_encoder.py --num-train {args.num_train} --num-test {args.num_test}")
