"""Search trajectory of the LAION visual encoding run (panel B of the fMRI figure).

Reads the run database read-only. Scored candidates are rows with status 'ok' and a test score; they are
ordered by evaluation order (created_at, ties by id) and numbered from 1. For each candidate the running
best is the highest pooled test score so far, reported with that model's early- and ventral-stream scores.
All scores are encoding r on the LAION search test images (format in figure/README.md).

Usage: python figure/b_trajectory.py --run RUN_DIR [--out OUT_JSON]
"""
import argparse
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGURE = Path(__file__).resolve().parent
OUT = FIGURE / "output/data/b.json"
CHAMPION = "erjpu"
POOLED, EARLY, VENTRAL = "early and ventral visual streams", "early visual stream", "ventral visual stream"


def load(run):
    db = sqlite3.connect(f"file:{run / 'search.db'}?mode=ro", uri=True)
    rows = db.execute("SELECT id, status, test_corr, roi_corr, created_at FROM models").fetchall()
    db.close()
    return rows


def trajectory(rows):
    scored = sorted((r for r in rows if r[1] == "ok" and r[2] is not None), key=lambda r: (r[4], r[0]))
    candidates = []
    for i, (mid, _, test, roi, created) in enumerate(scored, 1):
        roi = json.loads(roi)
        pooled = roi[POOLED]["test"]
        assert abs(pooled - test) < 1e-6, (mid, pooled, test)
        candidates.append({"index": i, "id": mid, "created_at": created,
                           "pooled": pooled, "early": roi[EARLY]["test"], "ventral": roi[VENTRAL]["test"]})
    running, best = [], None
    for c in candidates:
        if best is None or c["pooled"] > best["pooled"]:
            best = c
        running.append({"index": c["index"], **{k: best[k] for k in ("id", "pooled", "early", "ventral")}})
    return candidates, running


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="Historical run directory containing search.db and env.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = load(args.run)
    candidates, running = trajectory(rows)
    assert running[-1]["id"] == CHAMPION, f"overall pooled best is {running[-1]['id']}, not {CHAMPION}"
    env = json.loads((args.run / "env.json").read_text())
    status_counts = {}
    for r in rows:
        status_counts[r[1]] = status_counts.get(r[1], 0) + 1
    data = {
        "run": str(args.run.relative_to(REPO) if args.run.is_relative_to(REPO) else args.run),
        "champion": CHAMPION,
        "n_excluded": len(rows) - len(candidates),
        "candidates": candidates,
        "running_best": running,
        "provenance": {
            "run": str(args.run.resolve()),
            "database": "search.db (table models, read-only)",
            "protocol": env["protocol"]["name"],
            "protocol_version": env["protocol"]["version"],
            "dataset": env["protocol"]["search_dataset"],
            "subjects": env["protocol"]["search_subjects"],
            "region": env["protocol"]["search_region"],
            "metric": env["protocol"]["search_metric"],
            "score": "test r on LAION search images, from roi_corr[region]['test']",
            "order": "created_at, ties by id",
            "db_rows": len(rows), "db_status_counts": status_counts, "n_scored": len(candidates),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    print(f"wrote {args.out}: {len(candidates)} scored, {data['n_excluded']} excluded, "
          f"best {running[-1]['id']} r = {running[-1]['pooled']:.4f}")


if __name__ == "__main__":
    main()
