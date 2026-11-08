"""Score baselines under the exact search protocol, or on its held-out final data.

    python -m baselines.run --dataset laion --region visual
    python -m baselines.run --dataset laion --region visual --final --only gabor,gist
    python -m baselines.run --final --only pixels --model champion=final_results/.../model.py

Baselines are fixed a priori, so scoring them on final data selects nothing; results
go to final_results/baselines/, never to search_runs/. Final NSD rows add named-area
diagnostics (each area's retained voxels within the pooled units), as in search.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "final_results" / "baselines"
AREAS = ("V1", "V2", "V3", "hV4", "FFA", "PPA")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="laion", choices=("laion", "nsd", "triplen"))
    parser.add_argument("--region", default=None, choices=("visual", "early", "ventral"))
    parser.add_argument("--final", action="store_true", help="score held-out subjects and images (TVSD for triplen)")
    parser.add_argument("--only", help="comma-separated baseline names")
    parser.add_argument("--model", action="append", default=[], metavar="[NAME=]PATH",
                        help="also score a candidate model file, sandboxed as in search (repeatable)")
    args = parser.parse_args(argv)

    import final_eval
    from baselines import load
    from neuro_eval.benchmarks import default_search_region
    from neuro_eval.datasets import get_dataset
    from neuro_eval.encoding import evaluate_subjects

    region = args.region or default_search_region(args.dataset)
    models = load(args.only.split(",") if args.only else None) if args.only or not args.model else {}
    models.update(candidates(args.model))
    os.environ["NEURO_MAX_PARAMS"] = str(10**9)          # trusted references may hold pretrained weights
    if args.final:
        selection = {"search_dataset": args.dataset, "search_region": region,
                     "search_metric": "encoding", "environment": {}}
        with final_eval.final_environment(selection) as protocol:
            os.environ["NEURO_MAX_PARAMS"] = str(10**9)
            dataset = get_dataset(protocol.final_dataset, region=region, search_metric="encoding")
            rows = score_final(models, dataset, protocol.final_subjects)
            out = OUT / f"{protocol.final_dataset}__{region}__encoding__final.json"
    else:
        os.environ.update(NEURO_DATASET=args.dataset, NEURO_REGION=region, NEURO_METRIC="encoding", NEURO_STAGE="search")
        os.environ.pop("NEURO_SUBJECT", None)
        dataset = get_dataset(args.dataset, region=region, metric="encoding")
        rows = {}
        for name, model in models.items():
            t0 = time.time()
            rows[name] = evaluate_subjects(model, dataset, stage="search")
            rows[name]["seconds"] = time.time() - t0
            print(f"{name:22s} {str(rows[name].get('status'))[:60]} {rows[name].get('test_corr')}", flush=True)
        out = OUT / f"{args.dataset}__{region}__encoding.json"
    save(models, rows, out)


def candidates(specs):
    """Sandboxed candidate files, keyed by NAME (default: the file's parent directory)."""
    from neuro_eval.model import IsolatedModel, scan_source
    models = {}
    for spec in specs:
        name, _, path = spec.rpartition("=")
        path = Path(path).resolve()
        if reason := scan_source(str(path)):
            raise ValueError(f"{path} rejected: {reason}")
        models[name or path.parent.name] = IsolatedModel(path)
    return models


def score_final(models, dataset, subjects):
    """Subject-outer loop: load each final split once, score every model on it."""
    from neuro_eval import encoding, _evaluation
    from neuro_eval.datasets import with_subject, load_split
    per_model = {name: [] for name in models}
    seconds = dict.fromkeys(models, 0.0)
    for s in subjects:
        d = with_subject(dataset, s)
        split = with_areas(load_split(d), d) if d.loader == "nsd" else load_split(d)
        for name, model in models.items():
            t0 = time.time()
            row = encoding.evaluate_on_split(model, split, d)
            seconds[name] += time.time() - t0
            per_model[name].append({**row, "subject": d.subject})
            print(f"{s} {name:22s} {row['test_corr']:.4f}", flush=True)
    return {name: {**_evaluation._mean_rows(rows), "seconds": seconds[name]} for name, rows in per_model.items()}


def with_areas(split, d):
    """Add NSD named-area masks: the area's retained voxels intersected with the pooled units."""
    from neuro_eval import readers
    from neuro_eval.benchmarks import FILTERS
    from neuro_eval.datasets import _load_native, VisionSplit
    native, subject = _load_native(d)
    pooled = np.concatenate([np.asarray(native["units"][r][subject]) for r in native["regions"]])
    index = {tuple(u): i for i, u in enumerate(pooled.tolist())}
    masks = dict(split[4])
    for area in AREAS:
        units = readers.load_all_nsd_data(dict(FILTERS["nsd"]), subjects=[subject], regions=[area])["units"][area][subject]
        hits = [index[tuple(u)] for u in np.asarray(units).tolist() if tuple(u) in index]
        if hits:
            masks[area] = np.isin(np.arange(len(pooled)), hits)
    return VisionSplit((*split[:4], masks, split[5]), source_groups=getattr(split, "source_groups", {}))


def forward_seconds(model, n=1024, size=128):
    """Wall time of one forward over n images on the GPU, after a warm-up call."""
    X = torch.rand(n, size * size * 3, device="cuda" if torch.cuda.is_available() else "cpu")
    model(X[:8])
    torch.cuda.synchronize() if X.is_cuda else None
    t0 = time.time()
    model(X)
    torch.cuda.synchronize() if X.is_cuda else None
    return time.time() - t0


def save(models, rows, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    for name, row in rows.items():
        diag = (row.get("stream_diagnostics") or {}).get("subregions", {})
        areas = {k: v.get("score") for k, v in diag.items()} or {
            k: v["test"] for k, v in (row.get("roi_corr") or {}).items()}
        results[name] = {"status": row.get("status"), "test_corr": row.get("test_corr"), "areas": areas,
                         "subject_scores": row.get("subject_scores"), "n_params": row.get("n_params"),
                         "eval_seconds": round(row["seconds"], 1),
                         "forward_seconds_1k": round(forward_seconds(models[name]), 3)}
    out.write_text(json.dumps(results, indent=2, default=str))
    out.with_suffix(".md").write_text(table(results))
    print(f"wrote {out} and {out.with_suffix('.md').name}")


def table(results):
    """Markdown table: pooled score, each area/stream, evaluation and forward times."""
    cols = [k for k in dict.fromkeys(k for r in results.values() for k in r.get("areas", {}))
            if k != "early and ventral visual streams"]                      # same as pooled
    head = ["model", "pooled", *cols, "eval s", "fwd s / 1k img"]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    rows = [" | ".join([name, fmt(r.get("test_corr")), *(fmt(r["areas"].get(c)) for c in cols),
                        f"{r['eval_seconds']:.0f}", f"{r['forward_seconds_1k']:.2f}"])
            for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("test_corr") or -1))]
    return "\n".join(f"| {line} |" for line in [" | ".join(head), " | ".join(["---"] * len(head)), *rows]) + "\n"


if __name__ == "__main__":
    main()
