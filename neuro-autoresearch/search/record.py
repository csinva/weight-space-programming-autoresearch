#!/usr/bin/env python3
"""Insert one evaluated model into the SQLite db (``search/state/results.db``). Called by the evaluator.

    python search/record.py search/state/queue/<id>.py

Reads that model's harness result CSV (``runs/lab-eval/results/<STUDY>__<ROI>__<id>.csv``,
written by the eval run with ``NEURO_RUN=<id>``) plus the model source, and inserts one row
via ``search/db.py``. If no result file exists (the model failed to import), records a
``status=error`` row so nothing is silently dropped. ``NEURO_STUDY``/``NEURO_ROI``/
``NEURO_DATASET`` (default ``NsdSubj01``/``ventral``/``nsd``) locate the file and tag the row.
"""
from __future__ import annotations
import csv, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db as _db

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.environ.get("NEURO_STUDY", "NsdSubj01")
ROI = os.environ.get("NEURO_ROI", "ventral")
DATASET = os.environ.get("NEURO_DATASET", "nsd")
STATE = os.path.join(REPO, "search", "state", f"{STUDY}__{ROI}")   # per-dataset namespace


def main() -> None:
    model_path = sys.argv[1]
    mid = os.path.splitext(os.path.basename(model_path))[0]
    code = open(model_path).read()
    m = re.search(r"#\s*idea:\s*(.+)", code)
    idea = (m.group(1).strip() if m else "")[:300]
    itf = os.path.join(STATE, "ITER")
    it = open(itf).read().strip() if os.path.exists(itf) else ""

    csvp = os.path.join(REPO, "runs", "lab-eval", "results", f"{STUDY}__{ROI}__{mid}.csv")
    rows = list(csv.DictReader(open(csvp))) if os.path.exists(csvp) else []
    r = rows[-1] if rows else {"status": "error: no result produced (import/syntax failure)"}
    g = lambda k: r.get(k, "")

    _db.insert({
        "id": mid, "iter": it, "idea": idea, "dataset": DATASET, "roi": ROI,
        "test_corr": g("test_corr"), "train_corr": g("train_corr"), "gap": g("gap"),
        "corr_V1": g("corr_V1"), "corr_hV4": g("corr_hV4"), "corr_early": g("corr_early"),
        "corr_ventral": g("corr_ventral"), "corr_FFA": g("corr_FFA"), "corr_PPA": g("corr_PPA"),
        "corr_EBA": g("corr_EBA"), "corr_VWFA": g("corr_VWFA"),
        "status": g("status") or "error", "code": code,
    }, path=os.path.join(STATE, "results.db"))
    print(f"[record] {mid}: test_corr={g('test_corr') or '-'} status={g('status') or '-'}")


if __name__ == "__main__":
    main()
