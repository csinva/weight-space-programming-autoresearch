"""Cross-subject generalization study (NSD).

Takes the best never-trained model discovered for each region on subject 1 and
re-evaluates it, UNCHANGED, on subjects 1/2/3 -- re-fitting the ridge readout per
subject (the readout is part of eval, not the model). The shared-1000 test images
are identical across NSD subjects, so the same image->activation model transfers;
only the neural target and its readout differ. One row per (region, subject) is
appended to ``results/generalization.csv``.

Reuses the FIXED harness wholesale: ``load_split`` builds each subject's split and
``evaluate_on_split_safe`` scores it exactly as the autoresearch loop did. Nothing
under ``harness/`` is replicated here.

Prereq: subject-2/3 NSD data must be staged locally (subject 1 already is). Point at
it with NSD_SUBJ02_PATH / NSD_SUBJ03_PATH, else the sibling-dir convention below is
used. Run from the repo root::

    python scripts/generalization.py                 # all regions, subjects 1/2/3
    python scripts/generalization.py --subjects 1,2  # subset
    python scripts/generalization.py --regions V1,V2,V3
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness.studies  # noqa: F401,E402  (registers NsdSubj01/02/03)
from harness.datasets import PRESETS  # noqa: E402
from harness.data import load_split  # noqa: E402
from harness.reward import evaluate_on_split_safe  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "generalization.csv")

# Best run per region (the source of each region's winning model on subject 1).
REGIONS = {
    "V1": "nsd-v1-0611-0133",
    "V2": "nsd-v2-0611-1558",
    "V3": "nsd-v3-0611-1558",
    "hV4": "nsd-hv4-0611-0422",
    "FFA": "nsd-ffa-seq-0611-0636",
    "PPA": "nsd-ppa-0611-0133",
    "EBA": "nsd-eba-0611-0422",
    "VWFA": "nsd-vwfa-0611-0422",
    "ventral": "nsd-ventral-seq-0611-0636",
}

# Subject -> (study class name, staged data path). Subject 1 is already staged; the
# others default to a sibling-dir convention and can be overridden by env vars.
_S1 = PRESETS["nsd"].path
SUBJECTS = {
    1: ("NsdSubj01", _S1),
    2: ("NsdSubj02", os.environ.get("NSD_SUBJ02_PATH", _S1.replace("subj01", "subj02"))),
    3: ("NsdSubj03", os.environ.get("NSD_SUBJ03_PATH", _S1.replace("subj01", "subj03"))),
}

COLS = ["region", "subject", "study", "run", "model_name", "snapshot",
        "test_corr", "train_corr", "gap", "n_targets", "n_train", "n_test",
        "seconds", "status"]


def best_model(run_folder: str) -> tuple[str, str]:
    """(model_name, snapshot_path) of the run's best status-ok model."""
    csv_path = glob.glob(os.path.join(REPO, "runs", run_folder, "results", "*.csv"))[0]
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"]
    name = ok.loc[ok["test_corr"].idxmax(), "model_name"]
    lib = os.path.join(REPO, "runs", run_folder, "models_lib")
    hits = glob.glob(os.path.join(lib, f"*-{name}.py"))
    if not hits:                      # fall back to matching MODEL.name inside snapshots
        hits = [p for p in glob.glob(os.path.join(lib, "*.py"))
                if getattr(_load_model(p), "name", None) == name]
    if not hits:
        raise FileNotFoundError(f"no snapshot for model {name!r} in {lib}")
    return name, hits[0]


def _load_model(snapshot_path: str):
    """Import a snapshot .py in isolation and return its MODEL instance."""
    spec = importlib.util.spec_from_file_location(
        f"_gensnap_{abs(hash(snapshot_path))}", snapshot_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def load_done() -> set:
    if not os.path.exists(OUT):
        return set()
    with open(OUT, newline="") as f:
        return {(r["region"], r["subject"], r["model_name"]) for r in csv.DictReader(f)}


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
    ap.add_argument("--regions", default="", help="comma-separated subset of REGIONS")
    ap.add_argument("--subjects", default="1,2,3", help="comma-separated subject ids")
    args = ap.parse_args()

    regions = args.regions.split(",") if args.regions else list(REGIONS)
    subjects = [int(s) for s in args.subjects.split(",")]
    done = load_done()
    base = PRESETS["nsd"]

    for region in regions:
        run = REGIONS[region]
        name, snapshot = best_model(run)
        model = _load_model(snapshot)
        for subj in subjects:
            study, path = SUBJECTS[subj]
            key = (region, str(subj), name)
            if key in done:
                print(f"skip  {region:8} subj{subj} {name} (done)", flush=True)
                continue
            d = replace(base, study=study, path=path, roi="nsdgeneral",
                        reward_roi=region, run_tag=f"gen-subj{subj}")
            split = load_split(d)
            row = evaluate_on_split_safe(model, split, d)
            append_row({
                "region": region, "subject": subj, "study": study, "run": run,
                "model_name": name, "snapshot": os.path.basename(snapshot),
                "test_corr": f"{row['test_corr']:.4f}", "train_corr": f"{row['train_corr']:.4f}",
                "gap": f"{row['gap']:.4f}", "n_targets": row["n_targets"],
                "n_train": row["n_train"], "n_test": row["n_test"],
                "seconds": f"{row['seconds']:.1f}", "status": row["status"],
            })
            print(f"{region:8} subj{subj} {name:24} test_corr={row['test_corr']:.4f} "
                  f"({row['status']})", flush=True)
    print("GENERALIZATION DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
