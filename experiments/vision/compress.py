"""Natural-language compression: describe a model in <= N words, rebuild it from the text alone, compare.

    python compress.py --model path/to/model.py --words 200 --builders 4
    python compress.py --model path/to/model.py --words 50,100,200,400
    python compress.py --model path/to/model.py --description "A V1-like Gabor energy model."

1. Describe: Claude reads only the code and writes <= N words of plain prose (no code or identifiers).
2. Rebuild: K fresh Claude calls each see only that text plus the model contract and write a model.py;
   the only feedback is a contract check on random images (up to 3 attempts).
3. Compare: score the original, every rebuild and the gabor_norm floor with the search evaluator
   (LAION by default; --final adds held-out LAION subjects and images), report score retention
   (rebuild - floor) / (original - floor) and linear CKA between rebuild and original features.
Both Claude steps run without tools, settings or memory, so the builder cannot see the original.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
CLAUDE = ["claude", "-p", "--model", "claude-opus-5-5", "--effort", "medium", "--tools", "",
          "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json"]
FLOOR = "gabor_norm"

DESCRIBE = """Below is the Python source of an image-feature model of human visual cortex. Its features are fed to a
ridge-regression encoding model that predicts fMRI responses in early (V1-V3) and ventral visual cortex.

Describe the computation in at most {words} words of plain prose, so that a competent engineer who never sees
this code could implement a model that behaves the same. Prioritize what matters most for the features it
produces. No code, no formulas in code syntax, no variable or function names. Numbers are allowed and count
as words. Reply with the description only.

```python
{code}
```"""

CONTRACT = """Write one self-contained Python file, model.py, that exports MODEL: a callable taking a torch float tensor X
of shape (N, 128*128*3) (flattened HWC RGB images in [0, 1], possibly on GPU, N up to 1024) and returning a
finite torch tensor of shape (N, D) with D < 10,000. Rules: no training and no pretrained weights; every
parameter is fixed, analytic or seeded random, at most 100,000 materialized parameters in total; each image's
features must not depend on the other images in the batch; no file or network access; imports limited to
torch, numpy, scipy and math; bound GPU memory (process images in chunks). The features will be fed to a
ridge-regression encoding model of fMRI responses in human early and ventral visual cortex.

Implement the model described here:

{description}

Reply with the complete model.py in a single ```python code block."""

RETRY = """Your previous model.py failed the contract check:

{error}

Previous code:

```python
{code}
```

Fix it and reply with the complete model.py in a single ```python code block."""


def claude(prompt):
    with tempfile.TemporaryDirectory() as empty:          # no project files, CLAUDE.md or memory
        out = subprocess.run(CLAUDE, input=prompt, capture_output=True, text=True, cwd=empty, timeout=1800)
    if out.returncode:
        raise RuntimeError(f"claude failed: {out.stderr[-500:] or out.stdout[-500:]}")
    reply = json.loads(out.stdout)
    return reply["result"], reply.get("total_cost_usd", 0.0)


def describe(code, words):
    """<= words of prose with no code identifiers; one corrective retry."""
    identifiers = set(re.findall(r"\b[A-Za-z]\w*_\w*\b", code))
    prompt, cost = DESCRIBE.format(words=words, code=code), 0.0
    for _ in range(2):
        text, c = claude(prompt)
        cost += c
        text = text.strip()
        leaked = sorted(identifiers & set(re.findall(r"\b\w+\b", text))) + (["a code block"] if "```" in text else [])
        problems = ([f"it has {len(text.split())} words (limit {words})"] if len(text.split()) > words else []) + (
            [f"it contains code: {', '.join(leaked)}"] if leaked else [])
        if not problems:
            return text, cost
        prompt = DESCRIBE.format(words=words, code=code) + f"\n\nYour previous reply was rejected because {' and '.join(problems)}:\n\n{text}"
    raise ValueError(f"no valid description within {words} words")


def check(path):
    """Contract check on random images: shape, finiteness, batch independence, feature and parameter caps."""
    from neuro_eval.model import load_model
    from neuro_eval.model import _validate_model_size
    model = load_model(path)
    X = torch.rand(8, 128 * 128 * 3, generator=torch.Generator().manual_seed(0))
    Y = torch.as_tensor(model(X))
    if Y.shape[0] != 8 or not torch.isfinite(Y).all():
        raise ValueError(f"expected finite (8, D) output, got {tuple(Y.shape)}")
    _validate_model_size(model, Y.shape[1])
    if not torch.allclose(torch.as_tensor(model(X[:1])), Y[:1], atol=1e-4, rtol=1e-4):
        raise ValueError("an image's features change with the rest of the batch")


def rebuild(description, path, attempts=3):
    """Write path/model.py from the description alone; return (ok, cost, log)."""
    prompt, cost, log = CONTRACT.format(description=description), 0.0, []
    for attempt in range(attempts):
        reply, c = claude(prompt)
        cost += c
        blocks = re.findall(r"```python\n(.*?)```", reply, re.S)
        code = blocks[-1] if blocks else reply
        path.write_text(code)
        try:
            check(path)
            return True, cost, log
        except Exception as exc:  # noqa: BLE001 -- the error is the builder's feedback
            log.append(f"attempt {attempt + 1}: {type(exc).__name__}: {str(exc)[-1500:]}")
            prompt = RETRY.format(error=log[-1], code=code)
    return False, cost, log


def cka(paths, n=1000):
    """Linear CKA between each model's features and the first model's, on LAION shared test images."""
    from neuro_eval.datasets import get_dataset, with_subject, load_split
    from neuro_eval.model import load_model
    from neuro_eval.environment import preserved_environment

    with preserved_environment():
        os.environ.update(NEURO_DATASET="laion", NEURO_REGION="visual", NEURO_METRIC="encoding", NEURO_STAGE="search")
        X = load_split(with_subject(get_dataset("laion", region="visual", metric="encoding"), "sub-01"))[1][:n]
        feats = {}
        for name, path in paths.items():
            Y = torch.as_tensor(load_model(path)(X)).double()
            Y = Y - Y.mean(0)
            feats[name] = Y @ Y.T                              # image x image Gram matrix
        ref = feats[next(iter(feats))]
        return {name: float((ref * K).sum() / (ref.norm() * K.norm())) for name, K in feats.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="self-contained model.py to compress")
    parser.add_argument("--words", default="200", help="word budget(s), comma-separated")
    parser.add_argument("--builders", type=int, default=4, help="independent rebuilds per description")
    parser.add_argument("--description", help="use this text instead of describing the model (control)")
    parser.add_argument("--final", action="store_true", help="also score held-out LAION subjects and images")
    parser.add_argument("--name", help="output name (default: the model's parent directory)")
    args = parser.parse_args(argv)

    code = args.model.read_text()
    name = args.name or args.model.resolve().parent.name[:12]
    out = REPO / "final_results" / "compression" / f"{name}__{date.today()}"
    out.mkdir(parents=True, exist_ok=True)
    budgets = ["given"] if args.description else args.words.split(",")
    paths, runs, cost = {"original": args.model.resolve()}, {}, 0.0

    for budget in budgets:
        text, c = (args.description, 0.0) if args.description else describe(code, int(budget))
        cost += c
        tag = "given" if args.description else f"{budget}w"
        (out / f"description_{tag}.txt").write_text(text + "\n")
        print(f"[{tag}] {len(text.split())} words:\n{text}\n", flush=True)
        dirs = [out / f"rebuild_{tag}_{k}" for k in range(args.builders)]
        for d in dirs:
            d.mkdir(exist_ok=True)
        with ThreadPoolExecutor(args.builders) as pool:
            results = list(pool.map(lambda d: rebuild(text, d / "model.py"), dirs))
        for d, (ok, c, log) in zip(dirs, results):
            cost += c
            (d / "attempts.log").write_text("\n\n".join(log) + "\n")
            runs[d.name] = {"description": tag, "contract_ok": ok, "attempts": len(log) + ok}
            print(f"{d.name}: {'ok' if ok else 'FAILED'} after {len(log) + ok} attempt(s)", flush=True)
            if ok:
                paths[d.name] = d / "model.py"

    from baselines import load
    from neuro_eval.comparison import compare_models
    from neuro_eval.model import load_model

    models = {name: load_model(path) for name, path in paths.items()}
    models.update(load([FLOOR]))
    scores = {}
    for final in ([False, True] if args.final else [False]):
        stem = "laion__visual__encoding" + ("__final" if final else "")
        rows = compare_models(models, dataset="laion", region="visual", final=final, trusted=(FLOOR,))
        scores[stem] = rows
        (out / f"{stem}.json").write_text(json.dumps(rows, indent=2, default=str))
    similarity = cka(paths)
    report(out, runs, scores, similarity, cost)


def report(out, runs, scores, similarity, cost):
    lines = [f"# Natural-language compression ({date.today()})", "",
             f"Claude cost: ${cost:.2f}. Retention = (rebuild - {FLOOR}) / (original - {FLOOR}); "
             "CKA = linear CKA with the original's features on 1,000 LAION test images.", ""]
    for stem, rows in scores.items():
        floor, orig = rows[FLOOR]["test_corr"], rows["original"]["test_corr"]
        lines += [f"## {stem}", "", "| model | pooled | early | ventral | retention | CKA |", "|---|---|---|---|---|---|"]
        for model, r in rows.items():
            a = r.get("areas", {})
            ret = (r["test_corr"] - floor) / (orig - floor) if r.get("test_corr") is not None else float("nan")
            lines.append(f"| {model} | {r.get('test_corr') or float('nan'):.3f} | {a.get('early visual stream', float('nan')):.3f} | "
                         f"{a.get('ventral visual stream', float('nan')):.3f} | {ret:.2f} | {similarity.get(model, float('nan')):.2f} |")
        lines.append("")
    failed = [k for k, r in runs.items() if not r["contract_ok"]]
    lines += [f"Rebuilds failing the contract check: {', '.join(failed) or 'none'}.", ""]
    for f in sorted(out.glob("description_*.txt")):
        lines += [f"## {f.stem}", "", f.read_text()]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "runs.json").write_text(json.dumps({"runs": runs, "cka": similarity, "cost_usd": cost}, indent=2))
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
