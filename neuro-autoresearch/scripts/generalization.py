"""Cross-subject generalization study (NSD ventral stream).

Takes the best never-trained model discovered on subject 1 (the ventral-stream champion of a
search run) and re-evaluates it, UNCHANGED, on subjects 1/2/3 -- re-fitting the ridge readout per
subject (the readout is part of eval, not the model). The shared-1000 test images are identical
across NSD subjects, so the same image->activation model transfers; only the neural target and
its readout differ. One row per subject is appended to ``results/generalization.csv``.

Reuses the FIXED harness wholesale (``load_split`` + ``evaluate_on_split_safe``) and reads the
champion straight from a run's ``search.db`` via ``lab``. Reward = the ventral stream.

Prereq: subject-2/3 NSD data staged locally (NSD_SUBJ02_PATH / NSD_SUBJ03_PATH, else a sibling-dir
convention). Run from the repo root::

    python scripts/generalization.py                       # latest nsd run, subjects 1/2/3
    python scripts/generalization.py --run runs/nsd__2026-06-24
    python scripts/generalization.py --subjects 1,2
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys
from dataclasses import replace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import lab  # noqa: E402  (champion lookup from a run's search.db)
import harness.studies  # noqa: F401,E402  (registers NsdSubj01/02/03)
from harness.datasets import PRESETS  # noqa: E402
from harness.data import load_split  # noqa: E402
from harness.reward import evaluate_on_split_safe  # noqa: E402

OUT = os.path.join(REPO, "results", "generalization.csv")
REWARD_ROI = "ventral"

_S1 = PRESETS["nsd"].path
SUBJECTS = {
    1: ("NsdSubj01", _S1),
    2: ("NsdSubj02", os.environ.get("NSD_SUBJ02_PATH", _S1.replace("subj01", "subj02"))),
    3: ("NsdSubj03", os.environ.get("NSD_SUBJ03_PATH", _S1.replace("subj01", "subj03"))),
}
COLS = ["subject", "study", "run", "champion_id", "test_corr", "train_corr", "gap",
        "n_targets", "n_train", "n_test", "seconds", "status"]


def latest_nsd_run() -> str:
    runs = sorted(glob.glob(os.path.join(REPO, "runs", "nsd__*")))
    if not runs:
        raise FileNotFoundError("no runs/nsd__* found; pass --run <dir>")
    return runs[-1]


def _load_model(snapshot_path: str):
    spec = importlib.util.spec_from_file_location(
        f"_gensnap_{abs(hash(snapshot_path))}", snapshot_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def append_row(row: dict) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    exists = os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})
        f.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="run dir (default: latest runs/nsd__*)")
    ap.add_argument("--subjects", default="1,2,3", help="comma-separated subject ids")
    args = ap.parse_args()

    run = os.path.abspath(args.run) if args.run else latest_nsd_run()
    subjects = [int(s) for s in args.subjects.split(",")]
    top = lab.champions(1, run=run)
    if not top:
        raise SystemExit(f"no scored (status=ok) model in {run}/search.db yet")
    champ = top[0]
    snapshot = champ["path"]
    model = _load_model(snapshot)
    print(f"champion {champ['id']} (test_corr={champ['test_corr']:.4f}) from {os.path.basename(run)}")

    base = PRESETS["nsd"]
    for subj in subjects:
        study, path = SUBJECTS[subj]
        d = replace(base, study=study, path=path, reward_roi=REWARD_ROI,
                    run_tag=f"gen-subj{subj}")
        row = evaluate_on_split_safe(model, load_split(d), d)
        append_row({
            "subject": subj, "study": study, "run": os.path.basename(run),
            "champion_id": champ["id"], "test_corr": f"{row['test_corr']:.4f}",
            "train_corr": f"{row['train_corr']:.4f}", "gap": f"{row['gap']:.4f}",
            "n_targets": row["n_targets"], "n_train": row["n_train"], "n_test": row["n_test"],
            "seconds": f"{row['seconds']:.1f}", "status": row["status"],
        })
        print(f"subj{subj} {study:10} test_corr={row['test_corr']:.4f} ({row['status']})", flush=True)
    print("GENERALIZATION DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
