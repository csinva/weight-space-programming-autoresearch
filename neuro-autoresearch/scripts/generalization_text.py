"""Cross-subject generalization study (Le Petit Prince fMRI / ``text``).

Takes ONE fixed never-trained model and re-evaluates it, UNCHANGED, on every English
subject of the corpus -- re-fitting the ridge readout per subject (the readout is part
of the eval, not the model). All 49 English subjects heard the SAME story, so the same
audio->activation model transfers; only the neural target and its readout differ. One
row per subject is appended to ``results/generalization_text.csv``.

This is the text-dataset analogue of ``scripts/generalization.py`` (NSD). It reuses the
FIXED harness wholesale: ``load_split`` builds each subject's split and
``evaluate_on_split_safe`` scores it exactly as the autoresearch loop does. The model
defaults to the repo's current ``model.py`` (the evolving artifact); pass ``--model`` to
score a saved snapshot instead.

Prereq: the all-subjects staging dir must exist (created when the English subjects were
downloaded). It defaults to a sibling of the single-subject ``text`` data dir and can be
overridden with ``LPP_EN_ALL_PATH``. Run from the repo root::

    python scripts/generalization_text.py                       # current model.py, all subjects
    python scripts/generalization_text.py --subjects EN057,EN058
    python scripts/generalization_text.py --model runs/<tag>/models_lib/07_foo.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness.studies  # noqa: F401,E402  (registers Li2022PetitSubj<ID> per subject)
from harness.studies.li2022_subj import EN_SUBJECTS  # noqa: E402
from harness.datasets import PRESETS  # noqa: E402
from harness.data import load_split  # noqa: E402
from harness.reward import evaluate_on_split_safe  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "generalization_text.csv")

# All-subjects staging dir (the real data folder, so any Li2022PetitSubj<ID> name
# resolves to it -- see _identify_study_subfolder). Sibling of the single-subject
# `text` data dir by convention; override with LPP_EN_ALL_PATH.
_TEXT = PRESETS["text"]
DATA = os.environ.get(
    "LPP_EN_ALL_PATH",
    os.path.join(os.path.dirname(_TEXT.path), "li2022petit_en_all", "Li2022PetitSubj"),
)

COLS = ["subject", "study", "model_name", "snapshot", "test_corr", "train_corr",
        "gap", "n_targets", "n_train", "n_test", "seconds", "status"]


def _load_model(snapshot_path: str):
    """Import a model .py in isolation and return its MODEL instance."""
    spec = importlib.util.spec_from_file_location(
        f"_gensnap_{abs(hash(snapshot_path))}", snapshot_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def load_done() -> set:
    if not os.path.exists(OUT):
        return set()
    with open(OUT, newline="") as f:
        return {(r["subject"], r["model_name"]) for r in csv.DictReader(f)}


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
    ap.add_argument("--model", default=os.path.join(REPO, "model.py"),
                    help="path to a model .py exposing MODEL (default: repo model.py)")
    ap.add_argument("--subjects", default="",
                    help="comma-separated subject ids (default: all 49 English subjects)")
    args = ap.parse_args()

    if not os.path.isdir(DATA):
        raise SystemExit(f"all-subjects data dir not found: {DATA}\n"
                         "Set LPP_EN_ALL_PATH to the staged Li2022PetitSubj folder.")

    subjects = args.subjects.split(",") if args.subjects else list(EN_SUBJECTS)
    model = _load_model(args.model)
    name = getattr(model, "name", "?")
    snapshot = os.path.basename(args.model)
    done = load_done()
    base = PRESETS["text"]

    print(f"model={name!r} ({snapshot})  subjects={len(subjects)}  data={DATA}", flush=True)
    for sub in subjects:
        key = (sub, name)
        if key in done:
            print(f"skip  {sub} {name} (done)", flush=True)
            continue
        study = f"Li2022PetitSubj{sub}"
        d = replace(base, study=study, path=DATA, run_tag=f"gen-{sub}")
        split = load_split(d)
        row = evaluate_on_split_safe(model, split, d)
        append_row({
            "subject": sub, "study": study, "model_name": name, "snapshot": snapshot,
            "test_corr": f"{row['test_corr']:.4f}", "train_corr": f"{row['train_corr']:.4f}",
            "gap": f"{row['gap']:.4f}", "n_targets": row["n_targets"],
            "n_train": row["n_train"], "n_test": row["n_test"],
            "seconds": f"{row['seconds']:.1f}", "status": row["status"],
        })
        print(f"{sub}  {name:20} test_corr={row['test_corr']:.4f} "
              f"train_corr={row['train_corr']:.4f} ({row['status']})", flush=True)
    print("GENERALIZATION-TEXT DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
