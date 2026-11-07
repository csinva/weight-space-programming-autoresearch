# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the Weight-Space Programming toy-task blog post.

Reads experiments/runs/*/results/overall_results.csv, computes per-task summaries, and
fills blog_template.html. Writes a standalone page to
analyze/toy_examples/report.html.

    uv run analyze/toy_examples/build_blog.py            # standalone page
    uv run analyze/toy_examples/build_blog.py --fragment OUT.html
        # body-only fragment (no <html>/<head> wrapper), for hosts that add one
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "experiments" / "runs"
TEMPLATE = HERE / "blog_template.html"
OUT = HERE / "report.html"

ALGORITHMIC = "Algorithmic"
LANGUAGE = "Language"
CATEGORY = {
    "addition-five-digits": ALGORITHMIC,
    "multiplication-five-digits": ALGORITHMIC,
    "sort-five-digits": ALGORITHMIC,
    "digit-counting-10": ALGORITHMIC,
    "parity-upto10-bits": ALGORITHMIC,
    "boolean-circuit-5-bits": ALGORITHMIC,
    "linear-interpolation-two-points": ALGORITHMIC,
    "word-reversal-3x3": ALGORITHMIC,
    "gcd-three-digits": ALGORITHMIC,
    "decimal-to-binary-8bit": ALGORITHMIC,
    "sentiment-sst2": LANGUAGE,
    "paraphrase-mrpc": LANGUAGE,
    "nli-snli": LANGUAGE,
}

# One-line mechanism summaries, condensed by hand from the agent's own
# `description` strings for the smallest perfect program in each run.
MECHANISM = {
    "addition-five-digits": "Carry-lookahead adder. Heads gather digit pairs, an MLP computes generate/propagate bits, the next block ORs carry paths, the last emits (a+b+c) mod 10.",
    "multiplication-five-digits": "Long multiplication. Block 0 forms column sums of a_i·b_j; nine blocks ripple decimal carries; the last block selects the digit for the current position.",
    "sort-five-digits": "Counting sort. One attention head builds a digit histogram; two ReLUs per digit turn (histogram, k) into the k-th smallest digit. Keeps LayerNorm, with constants derived around it.",
    "digit-counting-10": "One head broadcasts the query digit, a uniform head builds a histogram, the MLP selects the count and writes its tens and ones digits.",
    "parity-upto10-bits": "Uniform attention with value gain 11 counts the 1-bits exactly; nine ReLUs compute N − 2·⌊N/2⌋ in place.",
    "boolean-circuit-5-bits": "One head per prompt position gathers the bits and operators; each of four MLP layers applies one left-to-right reduction step from the truth table.",
    "linear-interpolation-two-points": "Hypothesis elimination. The MLP scores all 50 candidate lines (m, b) against both points; only the true line scores 1.0 and gates the answer lookup.",
    "word-reversal-3x3": "Pure routing. Position-only query/key weights point each output step at a fixed source position; identity value/output weights copy the token. The MLP is zeroed.",
    "gcd-three-digits": "Lookup table. Attention packs both inputs into one scalar s = 1000a + b; a ~10⁶-wide MLP stores gcd(a, b) for every valid pair.",
    "decimal-to-binary-8bit": "Attention whose keys are log place values makes the softmax sum the digits into N; eight stacked blocks each peel off one bit.",
}

# Language tasks: what the agent's best circuits actually were, and the
# majority-class baseline (attempt #1 in every language run was a constant predictor).
LANGUAGE_NOTE = {
    "sentiment-sst2": "Character bigram and trigram log-odds, counted on the 2,000-sentence pool, summed by attention and thresholded.",
    "paraphrase-mrpc": "Two hand-set features over character bags: weighted L1 overlap between the sentences, and a length-difference penalty.",
    "nli-snli": "Trigram detectors read by a linear head whose coefficients were fit with logistic regression on the same 200 evaluation pairs.",
}


def fnum(row, key, default):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def collect():
    tasks = []
    for run_dir in sorted(RUNS.iterdir()):
        csv_path = run_dir / "results" / "overall_results.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open()))
        if not rows:
            continue
        task = rows[0].get("task") or run_dir.name
        methods = [
            {
                "i": i,
                "name": r.get("model_shorthand_name") or "?",
                "acc": fnum(r, "accuracy", 0.0),
                "params": fnum(r, "n_params", 0.0),
                "desc": r.get("description") or "",
            }
            for i, r in enumerate(rows, start=1)
        ]
        best = max(m["acc"] for m in methods)
        near = [m for m in methods if m["acc"] >= best - 0.01]
        first = near[0]
        nonzero = [m for m in near if m["params"] > 0]
        smallest = min(nonzero, key=lambda m: (m["params"], m["i"]))
        tasks.append({
            "task": task,
            "run": run_dir.name,
            "category": CATEGORY.get(task, "Other"),
            "attempts": len(methods),
            "baseline": methods[0]["acc"],
            "best": best,
            "first_best_i": first["i"],
            "first_best_params": first["params"],
            "first_best_name": first["name"],
            "smallest_i": smallest["i"],
            "smallest_params": smallest["params"],
            "smallest_name": smallest["name"],
            "zero_param_attempts": sum(1 for m in near if m["params"] == 0),
            "mechanism": MECHANISM.get(task) or LANGUAGE_NOTE.get(task, ""),
            "methods": methods,
        })
    order = [ALGORITHMIC, LANGUAGE]
    tasks.sort(key=lambda t: (order.index(t["category"]) if t["category"] in order else 9, t["task"]))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=Path, help="also write a body-only fragment here")
    args = ap.parse_args()

    tasks = collect()
    body = TEMPLATE.read_text().replace("__DATA__", json.dumps(tasks, separators=(",", ":")))
    OUT.write_text("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                   "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                   "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    if args.fragment:
        args.fragment.write_text(body)
    n = sum(t["attempts"] for t in tasks)
    print(f"Wrote {OUT} ({len(tasks)} tasks, {n} attempts)")


if __name__ == "__main__":
    main()
