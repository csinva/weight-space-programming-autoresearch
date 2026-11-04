"""Score each trec-coarse run's best program on data outside its 2,000-example pool: the official test split and the rest of the training split."""
import contextlib, csv, glob, importlib.util, io, json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    out = {}
    for rd in sorted(glob.glob(os.path.join(ROOT, "runs-opus55", "trec-coarse-opus55-s*"))):
        rows = list(csv.DictReader(open(os.path.join(rd, "results", "overall_results.csv"))))
        best = max(rows, key=lambda r: float(r["accuracy"] or 0))
        sys.path.insert(0, rd)
        os.chdir(rd)
        import src.task as T
        from src.eval import autoregressive_generate
        from datasets import load_dataset
        path = os.path.join(rd, "interpretable_transformers_lib", best["model_shorthand_name"] + ".py")
        spec = importlib.util.spec_from_file_location("snap_" + os.path.basename(rd).replace("-", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
            task = T.get_task("trec-coarse")
            model = mod.build_model(task)
        allowed = set("abcdefghijklmnopqrstuvwxyz '")
        ds = load_dataset("CogComp/trec", revision="refs/convert/parquet")
        def acc(rows_):
            ex = [(T._clean_text(r["text"], allowed, 80) + "=", str(r["coarse_label"])) for r in rows_]
            n = 0
            for i in range(0, len(ex), 64):
                ids = torch.tensor([task.encode(p) for p, _ in ex[i:i + 64]])
                with torch.no_grad():
                    o = autoregressive_generate(model, ids, 1)
                n += sum(task.decode(a) == b for a, (_, b) in zip(o.tolist(), ex[i:i + 64]))
            return n / len(ex), len(ex)
        res = dict(run=os.path.basename(rd), program=best["model_shorthand_name"], eval_200=float(best["accuracy"]),
                   pool_2000=acc(list(ds["train"])[:2000])[0], rest_of_train=acc(list(ds["train"])[2000:]), official_test=acc(list(ds["test"])))
        out[res["run"]] = res
        print(res)
        sys.path.remove(rd)
    json.dump(out, open(os.path.join(ROOT, "runs-opus55", "heldout_trec.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
