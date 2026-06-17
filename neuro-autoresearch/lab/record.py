#!/usr/bin/env python3
"""Append one evaluated model to lab/results.csv (CSV-safe). Called by the evaluator.

    python lab/record.py lab/queue/<id>.py

Reads that model's harness result -- runs/lab-eval/results/NsdSubj01__ventral__<id>.csv,
written by the eval run with NEURO_RUN=<id> -- plus the model source, and appends one row
{id, iter, idea, scores, status, code} to lab/results.csv. If no result file exists (the
model failed to import), records a status=error row so nothing is silently dropped.
"""
from __future__ import annotations
import csv, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(REPO, "lab")
RESULTS = os.path.join(LAB, "results.csv")
COLS = ["id", "iter", "idea", "test_corr", "corr_V1", "corr_hV4", "corr_early",
        "corr_ventral", "corr_FFA", "corr_PPA", "corr_EBA", "corr_VWFA",
        "train_corr", "gap", "status", "code"]


def main() -> None:
    model_path = sys.argv[1]
    mid = os.path.splitext(os.path.basename(model_path))[0]
    code = open(model_path).read()
    m = re.search(r"#\s*idea:\s*(.+)", code)
    idea = (m.group(1).strip() if m else "")[:300]
    iter_file = os.path.join(LAB, "ITER")
    it = open(iter_file).read().strip() if os.path.exists(iter_file) else ""

    csvp = os.path.join(REPO, "runs", "lab-eval", "results", f"NsdSubj01__ventral__{mid}.csv")
    rows = list(csv.DictReader(open(csvp))) if os.path.exists(csvp) else []
    r = rows[-1] if rows else {"status": "error: no result produced (import/syntax failure)"}

    g = lambda k: r.get(k, "")
    out = {c: "" for c in COLS}
    out.update({"id": mid, "iter": it, "idea": idea, "code": code, "status": g("status"),
                "test_corr": g("test_corr"), "train_corr": g("train_corr"), "gap": g("gap"),
                "corr_V1": g("corr_V1"), "corr_hV4": g("corr_hV4"), "corr_early": g("corr_early"),
                "corr_ventral": g("corr_ventral"), "corr_FFA": g("corr_FFA"), "corr_PPA": g("corr_PPA"),
                "corr_EBA": g("corr_EBA"), "corr_VWFA": g("corr_VWFA")})
    with open(RESULTS, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLS).writerow(out)
    print(f"[record] {mid}: test_corr={g('test_corr') or '-'} status={g('status') or '-'}")


if __name__ == "__main__":
    main()
