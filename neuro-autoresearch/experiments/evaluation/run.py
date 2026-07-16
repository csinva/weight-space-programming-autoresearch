#!/usr/bin/env python3
"""Evaluate one search champion and the modality's reference baselines.

Every model is scored separately on every search and held-out subject through the exact harness
reward. This is the only writer of the canonical model-first ``evaluation.json`` artifact.

    python experiments/evaluation/run.py --dataset nsd
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from common import cohort_scores, sha256_file, validate_evaluation, write_evaluation  # noqa: E402
from datasets import DATASETS, evaluation_for       # noqa: E402

BASELINE_MODEL = os.path.join(REPO, "model.py")
_NO_CAP = "1000000000000"

VISION_BASELINES = [
    {"id": "alexnet_pretrained", "name": "alexnet", "weights": "pretrained",
     "label": "AlexNet (ImageNet pretrained)"},
    {"id": "alexnet_untrained", "name": "alexnet", "weights": "untrained",
     "label": "AlexNet (untrained)"},
    {"id": "resnet50_pretrained", "name": "resnet50", "weights": "pretrained",
     "label": "ResNet-50 (ImageNet pretrained)"},
    {"id": "resnet50_untrained", "name": "resnet50", "weights": "untrained",
     "label": "ResNet-50 (untrained)"},
    {"id": "vit_b16_pretrained", "name": "vit", "weights": "pretrained",
     "label": "ViT-B/16 (ImageNet pretrained)"},
    {"id": "vit_b16_untrained", "name": "vit", "weights": "untrained",
     "label": "ViT-B/16 (untrained)"},
    {"id": "dinov2_pretrained", "name": "dino", "weights": "pretrained",
     "label": "DINOv2 (pretrained)"},
    {"id": "clip_pretrained", "name": "clip", "weights": "pretrained",
     "label": "CLIP (pretrained)"},
]
LANGUAGE_BASELINES = [
    {"id": "gpt2_xl_pretrained", "name": "gpt2-xl", "weights": "pretrained",
     "label": "GPT-2 XL (pretrained)"},
]


def load_model(path):
    """Import a model file and return its ``MODEL`` instance."""
    spec = importlib.util.spec_from_file_location("eval_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def model_provenance(path):
    """Read the structured provenance header written by ``runs/best/export_champions.py``."""
    with open(path, errors="replace") as f:
        header = "".join(f.readlines()[:8])

    def field(name):
        match = re.search(rf"(?:^# |   ){re.escape(name)}: (.*?)(?=   [^:]+: |$)",
                          header, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    provenance = {
        "dataset": field("dataset"),
        "subjects": field("subjects"),
        "test_corr": field("test_corr"),
        "n_params": field("n_params"),
        "source_run": field("source run"),
        "model_id": field("model id"),
        "idea": field("idea"),
    }
    required = ("dataset", "subjects", "test_corr", "source_run", "model_id")
    missing = [name for name in required if not provenance[name]]
    if missing:
        raise SystemExit(
            f"{path} is missing exporter provenance ({', '.join(missing)}); "
            "regenerate it with runs/best/export_champions.py")
    provenance["test_corr"] = float(provenance["test_corr"])
    if provenance["n_params"] and provenance["n_params"] != "?":
        provenance["n_params"] = int(provenance["n_params"].replace(",", ""))
    else:
        provenance["n_params"] = None
    return provenance


def baseline_specs(domain):
    return VISION_BASELINES if domain == "vision" else LANGUAGE_BASELINES


def score_subject(dataset, subject, champ_path, lm_cache):
    """Score all requested models on one subject, keyed by stable schema model IDs."""
    os.environ["NEURO_DATASET"] = dataset
    os.environ["NEURO_SUBJECT"] = str(subject)
    os.environ.pop("NEURO_SUBJECTS", None)
    os.environ["NEURO_MAX_PARAMS"] = _NO_CAP
    for module in [name for name in sys.modules if name.startswith("harness")]:
        del sys.modules[module]
    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness import reward as R

    split = load_split(DATASET)
    out = {
        "champion": R.evaluate_on_split(load_model(champ_path), split, DATASET)["test_corr"],
        "random_projection": R.evaluate_on_split(load_model(BASELINE_MODEL), split, DATASET)["test_corr"],
    }
    device = R._readout_device()
    if evaluation_for(dataset)["domain"] == "vision":
        import vision as V
        Xtr, Xte, Ytr, Yte, roi_masks, nc = split
        for spec in VISION_BASELINES:
            out[spec["id"]] = V.score_model(
                spec["name"], spec["weights"], Xtr, Xte, Ytr, Yte, roi_masks, nc,
                DATASET, R, device, dim=4096, seed=0)["test_corr"]
    else:
        import language as L
        for spec in LANGUAGE_BASELINES:
            key = (spec["name"], spec["weights"])
            lm = lm_cache.get(key) or L.ContextualLM(spec["name"], device, True)
            lm_cache[key] = lm
            out[spec["id"]] = L.score_lang_subject(lm, dataset, str(subject), R, device)["test_corr"]
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_record(spec, scores):
    return {
        "label": spec["label"],
        "role": "baseline",
        "family": spec["name"],
        "weights": spec["weights"],
        "scores": scores,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=DATASETS,
                   help="dataset whose exported best model should be evaluated")
    p.add_argument("--output", help="override runs/best/<dataset>/evaluation.json")
    a = p.parse_args(argv)

    dataset = a.dataset
    config = evaluation_for(dataset)
    cohorts = {"search": list(config["search"]), "heldout": list(config["heldout"])}
    champ_path = os.path.join(REPO, "runs", "best", dataset, "model.py")
    if not os.path.exists(champ_path):
        raise SystemExit(f"missing {champ_path}; run runs/best/export_champions.py first")
    provenance = model_provenance(champ_path)
    expected_subjects = ",".join(cohorts["search"])
    actual_subjects = ",".join(s.strip() for s in provenance["subjects"].split(","))
    if provenance["dataset"] != dataset or actual_subjects != expected_subjects:
        raise SystemExit(
            f"{champ_path} was exported for {provenance['dataset']} subjects {actual_subjects}, "
            f"but the canonical {dataset} search cohort is {expected_subjects}; regenerate runs/best")
    champ_id = provenance["model_id"]
    champ_score = provenance["test_corr"]
    out = os.path.abspath(a.output or os.path.join(os.path.dirname(champ_path), "evaluation.json"))

    print(f"=== eval | {dataset} | champion {champ_id} (search reward {champ_score:.4f}) ===")
    print(f"  {config['domain']} | search {cohorts['search']} | held-out {cohorts['heldout']}\n")

    per_subject = {"search": {}, "heldout": {}}
    lm_cache = {}
    for cohort in ("search", "heldout"):
        for subject in cohorts[cohort]:
            started = time.time()
            scores = score_subject(dataset, subject, champ_path, lm_cache)
            per_subject[cohort][str(subject)] = scores
            print(f"  [{cohort:>7}] {subject}: champion {scores['champion']:.4f}  "
                  f"({time.time() - started:.0f}s)")

    expected_models = ["champion", "random_projection"]
    expected_models += [spec["id"] for spec in baseline_specs(config["domain"])]
    scores_by_model = {
        model_id: {
            cohort: cohort_scores(
                {subject: per_subject[cohort][subject][model_id] for subject in cohorts[cohort]},
                cohorts[cohort])
            for cohort in ("search", "heldout")
        }
        for model_id in expected_models
    }

    loaded = load_model(champ_path)
    models = {
        "champion": {
            "label": "Best searched model",
            "role": "champion",
            "model_id": champ_id,
            "model_file": os.path.relpath(champ_path, os.path.dirname(out)),
            "model_sha256": sha256_file(champ_path),
            "n_params": getattr(loaded, "n_params", provenance["n_params"]),
            "idea": provenance["idea"],
            "search_reward": champ_score,
            "scores": scores_by_model["champion"],
        },
        "random_projection": {
            "label": "Random projection",
            "role": "baseline",
            "family": "random_projection",
            "weights": "fixed_random",
            "model_file": os.path.relpath(BASELINE_MODEL, os.path.dirname(out)),
            "model_sha256": sha256_file(BASELINE_MODEL),
            "scores": scores_by_model["random_projection"],
        },
    }
    for spec in baseline_specs(config["domain"]):
        models[spec["id"]] = model_record(spec, scores_by_model[spec["id"]])

    document = {
        "schema_version": 1,
        "dataset": {"id": dataset, "domain": config["domain"], "metric": config["metric"]},
        "evaluation": {
            "source_run": os.path.join("runs", provenance["source_run"]),
            "git_commit": git_commit(),
            "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "readout": os.environ.get("NEURO_READOUT", "fast"),
        },
        "cohorts": cohorts,
        "models": models,
    }
    validate_evaluation(document, expected_models, champ_path)
    write_evaluation(document, out)

    print(f"\n  {'model':<24}{'search':>10}{'held-out':>12}")
    for model_id, model in models.items():
        print(f"  {model_id:<24}{model['scores']['search']['mean']:>10.4f}"
              f"{model['scores']['heldout']['mean']:>12.4f}")
    print(f"\n  saved -> {out}")
    return document


if __name__ == "__main__":
    main()
