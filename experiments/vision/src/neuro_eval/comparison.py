"""Compare fixed image models under the search or held-out encoding protocol.

The returned rows use the baseline artifact schema. Trusted reference names must
be explicit; generated models retain the configured candidate parameter limit.
Environment changes are scoped to the call, including failures. Calls that change
the process environment must not run concurrently in the same process.
"""
import contextlib
import os
import time

import numpy as np
import torch

from neuro_eval.benchmarks import default_search_region
from neuro_eval.environment import final_environment, preserved_environment

AREAS = ("V1", "V2", "V3", "hV4", "FFA", "PPA")


@contextlib.contextmanager
def _model_environment(name, trusted):
    previous = os.environ.get("NEURO_MAX_PARAMS")
    try:
        if name in trusted:
            os.environ["NEURO_MAX_PARAMS"] = str(10**9)
        yield
    finally:
        if previous is None:
            os.environ.pop("NEURO_MAX_PARAMS", None)
        else:
            os.environ["NEURO_MAX_PARAMS"] = previous


def compare_models(models, *, dataset="laion", region=None, final=False, trusted=()):
    """Return {name: score row} without writing artifacts.

    ``trusted`` names fixed reference models allowed to exceed the candidate
    parameter cap. All other models use NEURO_MAX_PARAMS (100,000 by default).
    Final comparisons use every final subject and only held-out shared images.
    """
    from neuro_eval.datasets import get_dataset
    from neuro_eval.encoding import evaluate_subjects

    trusted = frozenset(trusted)
    if unknown := trusted - models.keys():
        raise ValueError(f"Unknown trusted model names: {sorted(unknown)}")
    region = region or default_search_region(dataset)
    with preserved_environment():
        if final:
            selection = {"search_dataset": dataset, "search_region": region,
                         "search_metric": "encoding", "environment": {}}
            with final_environment(selection) as protocol:
                data = get_dataset(protocol.final_dataset, region=region, search_metric="encoding")
                rows = score_final(models, data, protocol.final_subjects, trusted=trusted)
                return prepare_results(models, rows, trusted=trusted)
        os.environ.update(NEURO_DATASET=dataset, NEURO_REGION=region,
                          NEURO_METRIC="encoding", NEURO_STAGE="search")
        os.environ.pop("NEURO_SUBJECT", None)
        data = get_dataset(dataset, region=region, metric="encoding")
        rows = {}
        for name, model in models.items():
            with _model_environment(name, trusted):
                t0 = time.time()
                rows[name] = evaluate_subjects(model, data, stage="search")
                rows[name]["seconds"] = time.time() - t0
                print(f"{name:22s} {str(rows[name].get('status'))[:60]} {rows[name].get('test_corr')}", flush=True)
        return prepare_results(models, rows, trusted=trusted)


def score_final(models, dataset, subjects, *, trusted=()):
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
            with _model_environment(name, trusted):
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


def prepare_results(models, rows, *, trusted=()):
    """Normalize score rows and measure the existing 1,024-image forward timing."""
    results = {}
    for name, row in rows.items():
        diag = (row.get("stream_diagnostics") or {}).get("subregions", {})
        areas = {k: v.get("score") for k, v in diag.items()} or {
            k: v["test"] for k, v in (row.get("roi_corr") or {}).items()}
        with _model_environment(name, trusted):
            elapsed = forward_seconds(models[name])
        results[name] = {"status": row.get("status"), "test_corr": row.get("test_corr"), "areas": areas,
                         "subject_scores": row.get("subject_scores"), "n_params": row.get("n_params"),
                         "eval_seconds": round(row["seconds"], 1),
                         "forward_seconds_1k": round(elapsed, 3)}
    return results


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
