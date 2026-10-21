#!/usr/bin/env python3
"""Cross-dataset generalization matrix for the vision champions (evaluation + storage only).

Take the best searched champion of each vision dataset (``runs/best/<dataset>/model.py``) and
score every one of them on every vision dataset's HELD-OUT subjects, through the exact harness
reward. Because all three vision loaders emit ``X`` as the same ``(N, imsize*imsize*3)`` [0,1] RGB
pixel tensor (``imsize=128``), a champion searched on one dataset runs unchanged on another -- only
the ridge readout is refit per column. The result is an ``rows x cols`` matrix whose (i, j) cell is
``test_corr`` of champion-i evaluated on dataset-j held-out subjects.

The scored matrix is written once to ``runs/best/cross_vision/cross_vision.json`` so the plot never
re-runs the (minutes-long) scoring; render it with the sibling ``cross_dataset_plot.py``.

The diagonal (champion on its own held-out cohort) reproduces the held-out mean already in
``runs/best/<dataset>/evaluation.json`` -- a built-in consistency check; the off-diagonals are the
cross-dataset transfer this experiment adds.

IMPORTANT: the metric differs per column (NSD & Triple-N are noise-normalized r^2; TVSD is raw mean
Pearson r), so the matrix is comparable DOWN each column ("which champion transfers best to this
dataset"), NOT across columns -- the stored ``retention`` (mean / that column's native champion)
carries the honest cross-column-safe comparison the plot colours by.

    python experiments/evaluation/cross_dataset.py
    python experiments/evaluation/cross_dataset.py --datasets nsd,tvsd,nnn
    python experiments/evaluation/cross_dataset_plot.py          # render the stored matrix
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from common import mean_scalar, sha256_file, write_evaluation  # noqa: E402
from legacy_datasets import VISION_DATASETS, evaluation_for            # noqa: E402
from legacy_run import _NO_CAP, _free_gpu, git_commit, load_model, model_provenance  # noqa: E402


def _champion(dataset: str) -> dict:
    """Path + exporter provenance of one dataset's best searched champion."""
    path = os.path.join(REPO, "runs", "best", dataset, "model.py")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run runs/best/export_champions.py first")
    provenance = model_provenance(path)
    config = evaluation_for(dataset)
    expected = ",".join(config["search"])
    actual = ",".join(s.strip() for s in provenance["subjects"].split(","))
    if provenance["dataset"] != dataset or actual != expected:
        raise SystemExit(
            f"{path} was exported for {provenance['dataset']} subjects {actual}, but the canonical "
            f"{dataset} search cohort is {expected}; regenerate runs/best")
    return {"path": path, "provenance": provenance}


def _score_column(dataset: str, subjects: list[str], champions: dict) -> dict:
    """Score every champion on one column dataset's held-out subjects.

    Returns ``{row_dataset: {subject: test_corr}}``. Mirrors ``run.score_subject``'s per-subject
    harness-module purge so ``harness.datasets`` re-reads NEURO_DATASET/NEURO_SUBJECT on re-import.
    """
    per_row: dict[str, dict[str, float]] = {row: {} for row in champions}
    for subject in subjects:
        os.environ["NEURO_DATASET"] = dataset
        os.environ["NEURO_SUBJECT"] = str(subject)
        os.environ.pop("NEURO_SUBJECTS", None)
        os.environ["NEURO_MAX_PARAMS"] = _NO_CAP
        _free_gpu()
        _keep = {"harness.loaders.studies"}
        for module in [m for m in sys.modules if m.startswith("harness") and m not in _keep]:
            del sys.modules[module]
        from harness.datasets import DATASET
        from harness.loaders import load_split
        from harness import reward as R

        split = load_split(DATASET)
        for row, champion in champions.items():
            result = R.evaluate_on_split(load_model(champion["path"]), split, DATASET)
            per_row[row][str(subject)] = float(result["test_corr"])
    return per_row


def build_matrix(datasets: list[str]) -> dict:
    """Score the full rows x cols champion-by-dataset held-out matrix into a serialisable document."""
    for dataset in datasets:
        if dataset not in VISION_DATASETS:
            raise SystemExit(f"{dataset!r} is not a vision dataset {VISION_DATASETS}")
    champions = {dataset: _champion(dataset) for dataset in datasets}

    cohorts = {dataset: [str(s) for s in evaluation_for(dataset)["heldout"]] for dataset in datasets}
    metrics = {dataset: evaluation_for(dataset)["metric"] for dataset in datasets}

    matrix: dict[str, dict[str, dict]] = {row: {} for row in datasets}
    print(f"=== cross-dataset vision matrix | rows={datasets} cols={datasets} | held-out cohort ===\n")
    for col in datasets:
        started = time.time()
        per_row = _score_column(col, cohorts[col], champions)
        for row in datasets:
            subjects = {s: round(per_row[row][s], 4) for s in cohorts[col]}
            matrix[row][col] = {"mean": mean_scalar(subjects.values()), "subjects": subjects}
        diag = matrix[col][col]["mean"]
        print(f"  [col {col:>4}] held-out {cohorts[col]} scored in {time.time() - started:.0f}s "
              f"(native champion {diag:.4f})")

    native = {col: matrix[col][col]["mean"] for col in datasets}
    for row in datasets:
        for col in datasets:
            base = native[col]
            matrix[row][col]["retention"] = round(matrix[row][col]["mean"] / base, 4) if base else None

    def champion_record(dataset: str) -> dict:
        provenance = champions[dataset]["provenance"]
        path = champions[dataset]["path"]
        return {
            "model_id": provenance["model_id"],
            "idea": provenance["idea"],
            "source_run": os.path.join("runs", provenance["source_run"]),
            "search_subjects": [s.strip() for s in provenance["subjects"].split(",")],
            "search_reward": provenance["test_corr"],
            "n_params": provenance["n_params"],
            "model_file": os.path.relpath(path, REPO),
            "model_sha256": sha256_file(path),
        }

    return {
        "schema_version": 1,
        "experiment": "cross_dataset_generalization",
        "domain": "vision",
        "cohort": "heldout",
        "row_axis": "champion_source_dataset",
        "col_axis": "evaluation_dataset",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "readout": os.environ.get("NEURO_READOUT", "fast"),
        "datasets": {
            dataset: {
                "metric": metrics[dataset],
                "heldout_subjects": cohorts[dataset],
                "champion": champion_record(dataset),
            }
            for dataset in datasets
        },
        "native": native,
        "matrix": matrix,
    }


def print_table(document: dict) -> None:
    """Print the rows x cols matrix; rows = champion source, cols = evaluation dataset."""
    datasets = list(document["matrix"])
    metrics = {d: document["datasets"][d]["metric"] for d in datasets}
    metric_tag = {"noise_normalized_r2": "r2/NC", "mean_pearson_r": "mean r"}
    print("\n  rows = champion trained on · cols = evaluated on (held-out subjects)")
    header = "  champion\\eval".ljust(18) + "".join(f"{d:>14}" for d in datasets)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in datasets:
        cells = "".join(f"{document['matrix'][row][col]['mean']:>14.4f}" for col in datasets)
        print(f"  {row:<16}{cells}")
    print("  metric per col: " + "  ".join(
        f"{d}={metric_tag.get(metrics[d], metrics[d])}" for d in datasets))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(VISION_DATASETS),
                   help="comma-separated vision datasets for the rows AND columns")
    p.add_argument("--output", default=os.path.join(REPO, "runs", "best", "cross_vision",
                   "cross_vision.json"), help="matrix JSON path")
    a = p.parse_args(argv)

    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    document = build_matrix(datasets)
    output = os.path.abspath(a.output)
    write_evaluation(document, output)
    print_table(document)
    print(f"\n  saved matrix -> {output}")
    print(f"  plot it with:  python {os.path.relpath(os.path.join(HERE, 'cross_dataset_plot.py'), REPO)} "
          f"--input {os.path.relpath(output, REPO)}")
    return document


if __name__ == "__main__":
    raise SystemExit("This is an archived pre-migration workflow. Use experiments/evaluation/final.py for current protocols; historical reproduction requires the original repository revision.")
