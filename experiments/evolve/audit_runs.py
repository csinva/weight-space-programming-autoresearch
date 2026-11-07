"""Audit the saved weight-space programs of each run for rule breaks the harness cannot block (the agent's own Python runs freely).

  uv run experiments/evolve/audit_runs.py runs-opus55     -> runs-opus55/audit.json

Flags, per snapshot in interpretable_transformers_lib/ (and the final interpretable_transformer.py):
  training : optimizers, .backward(), .fit(, gradient steps, least squares / pseudo-inverse fitting
  eval_data: calls to the task's example generator, dataset loading, or the evaluation function (memorizing the eval set)
  file_io  : open(...) / np.load / torch.load / pickle (reading files; paths outside the run folder are listed)
A pattern is flagged only if it occurs in the code (not comments or strings) more often than in the starting template.
Flags are for review, not proof of a rule break (e.g. np.linalg.pinv on a hand-built matrix is closed-form)."""
import json, os, re, sys

PATTERNS = {
    "training": [r"torch\.optim", r"\.backward\(", r"optimizer", r"\.fit\(", r"lstsq", r"\bpinv\b", r"requires_grad_\(True\)", r"autograd\.grad"],
    "eval_data": [r"generate_examples\(", r"load_dataset\(", r"_load_pool\(", r"\bevaluate\(", r"\.answer\b"],
    "file_io": [r"\bopen\(", r"np\.load\(", r"torch\.load\(", r"pickle\.", r"read_csv\("],
}


def code_only(src):
    """Drop the fixed main block; blank out comments and string literals (docstrings) in place, keeping the code text intact."""
    import io, tokenize
    i = src.find('if __name__ == "__main__":')
    src = src if i < 0 else src[:i]
    lines = src.splitlines(keepends=True)
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l))
    chars = list(src)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                a = starts[tok.start[0] - 1] + tok.start[1]; b = starts[tok.end[0] - 1] + tok.end[1]
                for j in range(a, b):
                    if chars[j] != "\n":
                        chars[j] = " "
    except (tokenize.TokenError, IndentationError):
        return src
    return "".join(chars)


def counts(src):
    return {k: {p: len(re.findall(p, src)) for p in ps} for k, ps in PATTERNS.items()}


BASE = counts(code_only(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interpretable_transformer.py")).read()))


def audit_file(path):
    raw = open(path, errors="replace").read()
    c = counts(code_only(raw))
    hits = {k: sorted(p for p, n in c[k].items() if n > BASE[k][p]) for k in PATTERNS}
    src = raw
    paths = sorted(set(re.findall(r"""['"](/[^'"]+|\.\./[^'"]+)['"]""", src)))
    return {k: v for k, v in hits.items() if v} | ({"paths": paths} if paths else {})


def main(root):
    out = {}
    for run in sorted(os.listdir(root)):
        rd = os.path.join(root, run)
        lib = os.path.join(rd, "interpretable_transformers_lib")
        if not os.path.isdir(lib):
            continue
        files = sorted(os.listdir(lib)) + ["../interpretable_transformer.py"]
        res = {}
        for f in files:
            p = os.path.join(lib, f)
            if os.path.exists(p) and p.endswith(".py"):
                a = audit_file(p)
                if a:
                    res[f.replace(".py", "").replace("../", "")] = a
        out[run] = dict(n_snapshots=len(files) - 1, flagged=res)
        print(run, len(files) - 1, "snapshots;", len(res), "flagged:", {k: list(v) for k, v in list(res.items())[:3]})
    json.dump(out, open(os.path.join(root, "audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sys.argv[1] if len(sys.argv) > 1 else "runs-opus55"))
