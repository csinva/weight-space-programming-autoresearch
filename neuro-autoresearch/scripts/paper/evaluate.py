#!/usr/bin/env python3
"""Run the final champion-vs-baseline evaluation and write one paper-ready CSV.

Examples:
  python scripts/paper/evaluate.py runs/nsd__DATE runs/nnn__DATE runs/ieeg__DATE runs/huth__DATE
  python scripts/paper/evaluate.py runs/huth__DATE --split huth=UTS01,UTS02:UTS03

Each run contributes its DB champion. Search subjects come from the run's env.json; held-out
subjects come from scripts/eval/splits.py or --split. Completed JSON jobs are reused.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
sys.path.insert(0, str(REPO / "scripts/eval"))
from splits import GEN_SPLITS  # noqa: E402

MODALITY = {"nsd": "vision", "nnn": "vision", "ieeg": "language", "huth": "language"}
BASELINES = {
    "vision": ("scripts/eval/vision.py",
               (("alexnet", "both"), ("resnet50", "both"), ("vit", "both"),
                ("clip", "pretrained"), ("dino", "pretrained"))),
    "language": ("scripts/eval/language.py", (("gpt2-xl", "pretrained"),)),
}
METRIC = {"nsd": "sign(r)*r^2/NC_r2", "nnn": "sign(r)*r^2/NC_r2",
          "ieeg": "mean_r", "huth": "mean_r"}


def dataset_of(run: Path) -> str:
    return run.name.split("__", 1)[0]


def tokens(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def overrides(values: list[str]) -> dict[str, dict]:
    out = {}
    for value in values:
        dataset, groups = value.split("=", 1)
        search, heldout = groups.split(":", 1)
        out[dataset] = {"axis": "NEURO_SUBJECT", "search": tokens(search),
                        "heldout": tokens(heldout)}
    return out


def run_env(run: Path) -> dict:
    path = run / "env.json"
    return json.loads(path.read_text()) if path.exists() else {}


def split_for(run: Path, custom: dict) -> dict:
    dataset = dataset_of(run)
    if dataset in custom:
        split = custom[dataset]
        actual = tokens(run_env(run).get("NEURO_SUBJECT", ""))
        if actual and actual != split["search"]:
            raise SystemExit(f"{dataset}: --split search subjects {split['search']} do not match "
                             f"the run's env.json subjects {actual}")
        return split
    if dataset not in GEN_SPLITS:
        raise SystemExit(f"{dataset}: no subject split; pass --split {dataset}=SEARCH:HELDOUT")
    split = {k: list(v) if isinstance(v, list) else v for k, v in GEN_SPLITS[dataset].items()}
    actual = tokens(run_env(run).get("NEURO_SUBJECT", ""))
    if actual:
        all_units = split["search"] + split["heldout"]
        split["search"] = actual
        split["heldout"] = [u for u in all_units if u not in actual]
    if not split["heldout"]:
        raise SystemExit(f"{dataset}: the run used every declared subject; no held-out cohort remains")
    return split


def execute(cmd: list[str], env: dict, output: Path, force: bool, dry_run: bool) -> None:
    if output.exists() and not force:
        print(f"[reuse] {output}")
        return
    print("[run] " + " ".join(cmd))
    if dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="completed runs/<dataset>__<date> directories")
    p.add_argument("--split", action="append", default=[], metavar="DATASET=SEARCH:HELDOUT",
                   help="override/add a subject split; each side is comma-separated")
    p.add_argument("--output", type=Path, default=REPO / "results/paper")
    p.add_argument("--force", action="store_true", help="rerun completed jobs")
    p.add_argument("--dry-run", action="store_true", help="print commands without evaluating")
    a = p.parse_args(argv)

    custom, jobs = overrides(a.split), []
    for raw_run in a.runs:
        run = raw_run.resolve()
        if not (run / "search.db").exists():
            p.error(f"not a search run: {run}")
        dataset, split = dataset_of(run), split_for(run, custom)
        if dataset not in MODALITY:
            p.error(f"dataset {dataset!r} is not in the paper matrix")
        env = os.environ.copy()
        env.update(run_env(run))
        env["NEURO_DATASET"] = dataset
        root = a.output / run.name

        champion = root / "champion.json"
        cmd = [PYTHON, "scripts/eval/evaluate.py", "--run", str(run), "--no-baselines",
               "--search-subjects", ",".join(split["search"]),
               "--heldout-subjects", ",".join(split["heldout"]), "--output", str(champion)]
        execute(cmd, env, champion, a.force, a.dry_run)

        for group in ("search", "heldout"):
            subjects = ",".join(split[group])
            family = MODALITY[dataset]
            script, baselines = BASELINES[family]
            for model, weights in baselines:
                output = root / f"{group}_{model}_baselines.json"
                cmd = [PYTHON, script, "--dataset", dataset, "--subject", subjects,
                       "--model", model, "--weights", weights, "--output", str(output)]
                execute(cmd, env, output, a.force, a.dry_run)
                jobs.append((dataset, group, subjects, family, output))

    if a.dry_run:
        return

    rows, champions = [], {}
    for run in map(Path.resolve, a.runs):
        dataset, split = dataset_of(run), split_for(run, custom)
        result = read(a.output / run.name / "champion.json")
        for group, key in (("search", "search_mean"), ("heldout", "heldout_mean")):
            champions[dataset, group] = result[key]
            rows.append({"dataset": dataset, "group": group, "subjects": ",".join(split[group]),
                         "model": "search-champion", "family": "champion", "weights": "untrained",
                         "metric": METRIC[dataset], "test_corr": result[key],
                         "champion_minus_model": 0.0})
    for dataset, group, subjects, family, path in jobs:
        result = read(path)
        for model, record in result["records"].items():
            rows.append({"dataset": dataset, "group": group, "subjects": subjects, "model": model,
                         "family": family, "weights": record["weights"],
                         "metric": METRIC[dataset], "test_corr": record["test_corr"],
                         "champion_minus_model": round(champions[dataset, group] - record["test_corr"], 4)})

    a.output.mkdir(parents=True, exist_ok=True)
    summary = a.output / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"\nPaper results: {summary}")


if __name__ == "__main__":
    main()
