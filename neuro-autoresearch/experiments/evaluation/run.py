#!/usr/bin/env python3
"""Score a run's champion and reference baselines on search and held-out subjects through the
EXACT harness reward, per subject, over the subjects the search optimized AND the held-out subjects.

Every cell is the harness ``test_corr`` on the dataset's reward region: noise-normalized for NSD and
NNN, and raw mean-r for TVSD, Huth, and iEEG. Subjects are scored separately, then averaged with
equal weight inside the search and held-out cohorts.

  * single-representation models (the searched CHAMPION, the RANDOM-PROJECTION baseline model.py) go
    straight through ``harness.reward.evaluate_on_split`` -- the identical search code, no concat;
  * multi-layer nets (AlexNet/ResNet/ViT/DINO/CLIP for vision, GPT-2-XL for language) are scored
    BEST-LAYER (layer chosen on a train-only split, no test leakage) via the shared vision and
    language baseline scorers. Untrained = the same architecture with random weights.

    python experiments/evaluation/run.py --run runs/nsd__2026-06-29
    python experiments/evaluation/run.py --run runs/ieeg__2026-07-07
    python experiments/evaluation/run.py --run runs/nsd__2026-06-29 --no-baselines
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import hub                                            # noqa: E402  (needs REPO on sys.path)
from datasets import evaluation_for

RESULTS = os.path.join(HERE, "results")
BASELINE_MODEL = os.path.join(REPO, "model.py")     # the repo-root random-projection baseline
_NO_CAP = "1000000000000"                           # re-scoring fixed reference models is not a proposal -> lift the cap

# Reference baselines per modality: (name, weights). Untrained only where a random-init of the same
# architecture is meaningful (AlexNet/ResNet/ViT); DINO/CLIP are pretrained-only.
VISION_BASELINES = [("alexnet", "pretrained"), ("alexnet", "untrained"),
                    ("resnet50", "pretrained"), ("resnet50", "untrained"),
                    ("vit", "pretrained"), ("vit", "untrained"),
                    ("dino", "pretrained"), ("clip", "pretrained")]
LANGUAGE_BASELINES = [("gpt2-xl", "pretrained")]


def load_model(path):
    """Import a model .py and return its ``MODEL`` instance (a fresh module each call)."""
    spec = importlib.util.spec_from_file_location("eval_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def db_champion(run):
    """The run's highest-reward ok model -> ``{id, path, test_corr}`` (reuses ``hub.champion``)."""
    if not os.path.exists(os.path.join(run, "search.db")):
        raise SystemExit(f"no search.db in {run}")
    c = hub.champion(run)
    if c is None:
        raise SystemExit(f"no scored models in {run}")
    # the archived source is <run>/models/<id>.py (the path column's dir may be stale after a rename)
    return {"id": c["id"], "path": os.path.join(run, "models", f"{c['id']}.py"),
            "test_corr": c["test_corr"]}


def score_subject(dataset, subject, champ_path, with_baselines, lm_cache):
    """Rebind the harness to ONE subject, then score every model on it -> ``{label: test_corr}``.

    The harness binds dataset/subject at import, so its modules are reloaded per subject. ``lm_cache``
    carries the (name, weights) -> ContextualLM baselines so a big LM (GPT-2-XL) loads once, not per subject.
    """
    os.environ["NEURO_DATASET"] = dataset
    os.environ["NEURO_SUBJECT"] = str(subject)
    os.environ.pop("NEURO_SUBJECTS", None)              # single-subject reward, never the mean
    os.environ["NEURO_MAX_PARAMS"] = _NO_CAP
    for m in [k for k in sys.modules if k.startswith("harness")]:
        del sys.modules[m]
    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness import reward as R

    split = load_split(DATASET)
    out = {"champion": R.evaluate_on_split(load_model(champ_path), split, DATASET)["test_corr"],
           "random-proj": R.evaluate_on_split(load_model(BASELINE_MODEL), split, DATASET)["test_corr"]}
    if not with_baselines:
        return out

    device = R._readout_device()
    if evaluation_for(dataset)["domain"] == "vision":
        import vision as V
        Xtr, Xte, Ytr, Yte, roi_masks, nc = split
        for name, w in VISION_BASELINES:
            out[f"{name}-{w}"] = V.score_model(name, w, Xtr, Xte, Ytr, Yte, roi_masks, nc,
                                               DATASET, R, device, dim=2048, seed=0)["test_corr"]
    else:                                               # language datasets (ieeg, huth)
        import language as L
        for name, w in LANGUAGE_BASELINES:
            lm = lm_cache.get((name, w)) or L.ContextualLM(name, device, w == "pretrained")
            lm_cache[(name, w)] = lm
            out[f"{name}-{w}"] = L.score_lang_subject(lm, dataset, str(subject), R, device)["test_corr"]
    return out


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="search run dir (holds search.db + models/)")
    p.add_argument("--model", default=None, help="champion .py to test (default: the run's DB champion)")
    p.add_argument("--dataset", default=None, help="override dataset (default: inferred from run dir)")
    p.add_argument("--no-baselines", action="store_true", help="champion + random-projection only")
    p.add_argument("--search-subjects", help="comma list overriding the declared search subjects")
    p.add_argument("--heldout-subjects", help="comma list overriding the declared held-out subjects")
    p.add_argument("--output", help="write JSON to this exact path")
    a = p.parse_args(argv)

    dataset = a.dataset or os.path.basename(os.path.normpath(a.run)).split("__")[0]
    if bool(a.search_subjects) != bool(a.heldout_subjects):
        p.error("--search-subjects and --heldout-subjects must be given together")
    config = evaluation_for(dataset)
    cohorts = ({"search": a.search_subjects.split(","), "heldout": a.heldout_subjects.split(",")}
               if a.search_subjects else config)
    if a.model:
        champ_path, champ_id, champ_score = a.model, os.path.basename(a.model).removesuffix(".py"), None
    else:
        c = db_champion(a.run)
        champ_path, champ_id, champ_score = c["path"], c["id"], c["test_corr"]

    suffix = f" (search reward {champ_score:.4f})" if champ_score is not None else ""
    print(f"=== eval | {dataset} | champion {champ_id}{suffix} ===")
    print(f"  {config['domain']} | search {cohorts['search']} | held-out {cohorts['heldout']}\n")

    rows, subject_results, lm_cache = {}, {"search": {}, "heldout": {}}, {}
    for kind in ("search", "heldout"):
        for subject in cohorts[kind]:
            t0 = time.time()
            scores = score_subject(dataset, subject, champ_path, not a.no_baselines, lm_cache)
            subject_results[kind][subject] = scores
            for label, tc in scores.items():
                rows.setdefault(label, {"search": [], "heldout": []})[kind].append(tc)
            print(f"  [{kind:>7}] {subject}: champion {scores['champion']:.4f}  ({time.time()-t0:.0f}s)")

    print(f"\n  {'model':<20}{'search':>10}{'held-out':>12}")
    summary = {}
    for label, d in rows.items():
        s, h = mean(d["search"]), mean(d["heldout"])
        summary[label] = {"search": round(s, 4), "heldout": round(h, 4)}
        print(f"  {label:<20}{s:>10.4f}{h:>12.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    out = a.output or os.path.join(RESULTS, f"eval_{dataset}_{champ_id}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    champ = summary.get("champion", {})
    with open(out, "w") as f:
        json.dump({"dataset": dataset, "champion": champ_id, "champion_path": champ_path,
                   "domain": config["domain"], "search_reward": champ_score,
                   "cohorts": {k: cohorts[k] for k in ("search", "heldout")},
                   "subjects": subject_results, "summary": summary,
                   "search_mean": champ.get("search"), "heldout_mean": champ.get("heldout")}, f, indent=2)
    print(f"\n  saved -> {out}")
    return summary


if __name__ == "__main__":
    main()
