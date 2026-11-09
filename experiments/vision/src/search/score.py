"""Trusted scoring of submitted candidates."""
import os
from search.queue import search_contract, find_run
from neuro_eval.model import scan_source

def score(model_file: str, dataset: str | None = None, *, region: str | None = None, metric: str | None = None) -> dict:
    """Score privately, running candidate forwards in an image-only OS sandbox.

    No candidate code is imported into this process. Isolation errors fail closed.
    """
    import contextlib, io, traceback
    from neuro_eval.benchmarks import validate_search_dataset, canonical_region, validate_search_metric, default_search_region, DEFAULT_SEARCH_METRIC
    dataset = validate_search_dataset(dataset or os.environ.get("NEURO_DATASET", "laion"))
    if os.environ.get("LAB_RUN"):
        contract = search_contract(find_run(), dataset, region, metric)
        region = region or contract["NEURO_REGION"]
        metric = metric or contract["NEURO_METRIC"]
    region = canonical_region(region or os.environ.get("NEURO_REGION") or default_search_region(dataset))
    metric = validate_search_metric(metric or os.environ.get("NEURO_METRIC", DEFAULT_SEARCH_METRIC))
    if os.environ.get("NEURO_STAGE", "search") != "search":
        raise ValueError("search.score is search-only; final evaluation goes through final_eval.py")
    os.environ.update(NEURO_DATASET=dataset, NEURO_REGION=region, NEURO_METRIC=metric)
    if os.environ.get("LAB_RUN"):
        from search.run import write_env_json
        write_env_json(find_run(), dataset, region, metric)
    os.environ.pop("NEURO_ROI", None)
    cheat = scan_source(model_file)              # static reject (also catches already-queued cheats)
    if cheat:
        return {"status": f"error: reward-hack blocked -- {cheat}"[:300], "log": ""}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            from neuro_eval.datasets import get_dataset
            if metric == "rsa":
                from neuro_eval.rsa import evaluate_subjects
            else:
                from neuro_eval.encoding import evaluate_subjects
            from neuro_eval.model import IsolatedModel
            model = IsolatedModel(model_file)
            row = evaluate_subjects(model, get_dataset(dataset, region=region, metric=metric), stage="search")
            row["repair_log"] = model.repair_log
            row["candidate_error"] = model.repair_log[-1500:]
        res = dict(row)
    except Exception as e:  # noqa: BLE001  -- any eval fault is a recorded error, not a crash
        traceback.print_exc(file=buf)
        res = {"status": f"error: {type(e).__name__}: {e}"[:300]}
    res.setdefault("log", buf.getvalue()[-4000:])
    return res



def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Score a model on the complete search cohort.')
    parser.add_argument('--dataset', default='laion', choices=('laion', 'nsd', 'triplen'))
    parser.add_argument('--region', default=None, choices=('visual', 'early', 'ventral'))
    parser.add_argument('--metric', default=None, choices=('rsa', 'encoding'))
    parser.add_argument('--model', default='model.py')
    args = parser.parse_args(argv)
    result = score(args.model, args.dataset, region=args.region, metric=args.metric)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if str(result.get('status', '')).startswith('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
