#!/usr/bin/env python3
"""Generalization test: take a search run's winning model and re-evaluate it, UNCHANGED, on the
held-out units it never saw during the search.

The search optimizes a reward over some units (NSD subjects 1,2); a model that captured real neural
computation -- not unit-specific quirks -- should still score on the held-out units (5,7). This runs
the EXACT harness reward (same loader, voxel selection, ridge, metric) per unit and reports the
search-vs-held-out table + the generalization gap. Dataset-agnostic: the per-dataset split (which
env var is the axis, which units are search vs held-out) lives in ``splits.py``.

    # auto-pick the run's DB champion, sweep nsd's search + held-out subjects
    python scripts/generalization/generalize.py --run runs/nsd__2026-06-28
    python scripts/generalization/generalize.py --run runs/nsd__2026-06-28 --model runs/.../models/foo.py
    python scripts/generalization/generalize.py --run runs/nsd__2026-06-28 --baseline   # + oracle-AlexNet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from splits import split_for

RESULTS = os.path.join(HERE, "results")


def load_model(path):
    """Import a model .py and return its ``MODEL`` instance (a fresh module each call)."""
    spec = importlib.util.spec_from_file_location("gen_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def db_champion(run):
    """Highest-scoring scored model in the run's ``search.db`` -> ``{id, path, test_corr}``."""
    db = os.path.join(run, "search.db")
    if not os.path.exists(db):
        raise SystemExit(f"no search.db in {run}")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT id, path, test_corr FROM models "
                      "WHERE status='ok' AND test_corr IS NOT NULL "
                      "ORDER BY test_corr DESC LIMIT 1").fetchone()
    con.close()
    if row is None:
        raise SystemExit(f"no scored models in {db}")
    return dict(row)


def dataset_of(run, override):
    """``runs/<dataset>__<date>`` -> ``<dataset>`` (or the explicit override)."""
    return override or os.path.basename(os.path.normpath(run)).split("__")[0]


def eval_unit(model, dataset, axis, unit, with_baseline):
    """Re-evaluate ``model`` on ONE unit through the exact harness reward (single-unit, never the mean).

    The harness binds the dataset/unit at import, so the harness modules are reloaded per unit.
    """
    os.environ["NEURO_DATASET"] = dataset
    os.environ[axis] = str(unit)
    os.environ.pop("NEURO_SUBJECTS", None)              # force the single-unit reward, not the mean
    for m in [k for k in list(sys.modules) if k.startswith("harness")]:
        del sys.modules[m]
    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness.reward import evaluate_on_split

    split = load_split(DATASET)
    row = evaluate_on_split(model, split, DATASET)
    out = {"unit": str(unit), "test_corr": round(row["test_corr"], 5),
           "roi_corr": {r: round(v["test"], 5) for r, v in row["roi_corr"].items()}}
    if with_baseline:
        from harness import reward as R
        from harness.roi import roi_breakdown
        from baseline import alex_oracle
        out["alex"] = alex_oracle(split, DATASET, R, roi_breakdown)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="search run dir (holds search.db + models/)")
    p.add_argument("--model", default=None, help="model .py to test (default: the run's DB champion)")
    p.add_argument("--dataset", default=None, help="override dataset (default: inferred from run dir)")
    p.add_argument("--baseline", action="store_true",
                   help="also score the oracle-AlexNet reference per unit")
    a = p.parse_args(argv)

    dataset = dataset_of(a.run, a.dataset)
    sp = split_for(dataset)
    axis = sp["axis"]
    if a.model:
        model_path, champ_id, champ_score = a.model, os.path.basename(a.model).removesuffix(".py"), None
    else:
        c = db_champion(a.run)
        model_path, champ_id, champ_score = c["path"], c["id"], c["test_corr"]

    head = f"=== generalization | {dataset} | model {champ_id}"
    if champ_score is not None:
        head += f" (search reward {champ_score:.4f})"
    print(head + f" | axis {axis} ===")
    print(f"  search units {sp['search']}  ->  held-out {sp['heldout']}")

    model = load_model(model_path)
    rows = []
    for kind in ("search", "heldout"):
        for unit in sp[kind]:
            t0 = time.time()
            r = eval_unit(model, dataset, axis, unit, a.baseline)
            r["kind"], r["seconds"] = kind, round(time.time() - t0, 1)
            rows.append(r)
            line = f"  [{kind:>7}] {axis}={unit}: reward {r['test_corr']:.4f}"
            if a.baseline:
                line += f"   alex-oracle {r['alex']['overall']:.4f}"
            print(line + f"   ({r['seconds']}s)")

    def mean(kind):
        v = [r["test_corr"] for r in rows if r["kind"] == kind]
        return sum(v) / len(v) if v else float("nan")

    s, h = mean("search"), mean("heldout")
    print(f"\n  search mean {s:.4f}   held-out mean {h:.4f}   gap {h - s:+.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS, f"gen_{dataset}_{champ_id}_{stamp}.json")
    with open(out, "w") as f:
        json.dump({"dataset": dataset, "model": champ_id, "model_path": model_path,
                   "search_reward": champ_score, "axis": axis,
                   "split": {"search": sp["search"], "heldout": sp["heldout"]},
                   "search_mean": round(s, 5), "heldout_mean": round(h, 5), "gap": round(h - s, 5),
                   "units": rows}, f, indent=2)
    print(f"  saved -> {out}")
    return rows


if __name__ == "__main__":
    main()
