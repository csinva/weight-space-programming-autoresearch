"""Score one weight-space program (a snapshot .py with build_model(task)) on three example sets and save per-example predictions.

  visible: the 200 seed-0 examples every attempt in the original runs was scored on
  hidden : generated tasks: 2,000 fresh examples (seed 987654321, as in evolve/heldout_fresh_seed.py)
           dataset tasks (sst2, mrpc, snli, trec): up to 2,000 examples sampled (seed 1) from training rows outside the agent's 2,000-example pool
  extra  : generated tasks only: 20,000 more fresh examples (seed 555), to expose rare errors a 2,000 sample would miss

  python eval_program.py PROGRAM.py TASK OUT.json
"""
import contextlib, importlib.util, io, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evolve"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
import src.task as T
from src.eval import autoregressive_generate

DATASET = ("sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse")
ALLOWED = set("abcdefghijklmnopqrstuvwxyz '")


def outside_pool(name):
    """All (prompt, answer) rows of the task's source split that are not in its 2,000-example pool, built exactly as the task builds its pool."""
    from datasets import load_dataset
    c = T._clean_text
    if name == "sentiment-sst2":
        rows = []
        for ex in load_dataset("stanfordnlp/sst2", split="train"):
            t = " ".join("".join(ch if ch in ALLOWED else " " for ch in ex["sentence"].lower()).split())
            if t:
                rows.append((t[:80].ljust(80, "_") + "=", str(ex["label"])))
    elif name == "paraphrase-mrpc":
        rows = [(f"{c(e['sentence1'], ALLOWED, 60)}|{c(e['sentence2'], ALLOWED, 60)}=", str(e["label"])) for e in load_dataset("nyu-mll/glue", "mrpc", split="train")]
    elif name == "nli-snli":
        rows = [(f"{c(e['premise'], ALLOWED, 60)}|{c(e['hypothesis'], ALLOWED, 60)}=", str(e["label"])) for e in load_dataset("stanfordnlp/snli", split="train") if e["label"] in (0, 1, 2)]
    else:
        rows = [(c(e["text"], ALLOWED, 80) + "=", str(e["coarse_label"])) for e in load_dataset("CogComp/trec", split="train", revision="refs/convert/parquet")]
    return rows[2000:]


def sets(name, task):
    vis = [(e.prompt, e.answer) for e in task.generate_examples(200, seed=0)]
    if name in DATASET:
        rest = outside_pool(name)
        hid = random.Random(1).sample(rest, min(2000, len(rest)))  # MRPC has only 1,668 rows outside its pool
        return dict(visible=vis, hidden=hid)
    return dict(visible=vis, hidden=[(e.prompt, e.answer) for e in task.generate_examples(2000, seed=987654321)],
                extra=[(e.prompt, e.answer) for e in task.generate_examples(20000, seed=555)])


@torch.no_grad()
def predict(model, task, ex, bs=250):
    out = []
    for i in range(0, len(ex), bs):
        ids = torch.tensor([task.encode(p) for p, _ in ex[i:i + bs]])
        out += [task.decode(r) for r in autoregressive_generate(model, ids, task.answer_len).tolist()]
    return out


if __name__ == "__main__":
    path, name, out = sys.argv[1:4]
    t0 = time.time()
    task = T.get_task(name)
    spec = importlib.util.spec_from_file_location("snap", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
        model = mod.build_model(task)
    model.eval()
    res = dict(task=name, program=path, n_params=sum(p.numel() for p in model.parameters()))
    for k, ex in sets(name, task).items():
        preds = predict(model, task, ex)
        ok = [task.is_correct(p, a) for p, (_, a) in zip(preds, ex)]
        res[k] = dict(n=len(ex), acc=sum(ok) / len(ok), prompts=[p for p, _ in ex], targets=[a for _, a in ex], preds=preds, correct=ok)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"))
    print(name, os.path.basename(path), {k: res[k]["acc"] for k in ("visible", "hidden", "extra") if k in res}, res["seconds"], "s", flush=True)
