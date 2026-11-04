"""Score each generated-task run's best program (first attempt at its best accuracy, among saved programs) on 2,000 new examples from an unseen seed."""
import contextlib, csv, glob, importlib.util, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, N = 987654321, 2000
DATASET = ("sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse")


def score(rd):
    os.environ["OMP_NUM_THREADS"] = "4"
    sys.path.insert(0, rd)
    os.chdir(rd)
    import src.task as T
    from src.eval import evaluate
    task_name = os.path.basename(rd).rsplit("-opus55-s", 1)[0]
    rows = list(csv.DictReader(open(os.path.join(rd, "results", "overall_results.csv"))))
    top = max(float(r["accuracy"] or 0) for r in rows)
    for r in rows:
        path = os.path.join(rd, "interpretable_transformers_lib", r["model_shorthand_name"] + ".py")
        if float(r["accuracy"] or 0) < top or not os.path.exists(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("snap", path)
            mod = importlib.util.module_from_spec(spec)
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(mod)
                task = T.get_task(task_name)
                model = mod.build_model(task)
                acc, _ = evaluate(model, task, n_samples=N, seed=SEED, device="cpu")
            return os.path.basename(rd), dict(program=r["model_shorthand_name"], eval_200=top, fresh_seed=acc)
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:100]}"
    return os.path.basename(rd), dict(error=locals().get("last", "no snapshot"), eval_200=top)


if __name__ == "__main__":
    runs = [d.rstrip("/") for d in sorted(glob.glob(os.path.join(ROOT, "runs-opus55", "*-opus55-s*/"))) if not os.path.basename(d.rstrip("/")).startswith(DATASET)]
    out = {}
    with ProcessPoolExecutor(13) as ex:
        for k, v in ex.map(score, runs):
            out[k] = v
            print(k, v, flush=True)
    json.dump(out, open(os.path.join(ROOT, "runs-opus55", "heldout_fresh_seed.json"), "w"), indent=1)
