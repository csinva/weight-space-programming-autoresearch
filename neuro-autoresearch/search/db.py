#!/usr/bin/env python3
"""SQLite gateway for the open-ended search database (``search/state/results.db``).

One table ``models`` -- one row per evaluated model. Python owns all writes (WAL mode +
busy_timeout, so the sequential evaluator, and any future concurrent writer, is safe);
``search/lib/db.mjs`` reads the same file via ``node:sqlite``.

    python search/db.py init                            # create the db
    python search/db.py migrate search/state/results.csv   # import existing CSV history
"""
from __future__ import annotations
import csv, os, sqlite3, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "search", "state", "results.db")
# NSD region panel (REAL); dataset/roi tag the run so the schema is multi-dataset-ready.
NUM = ["test_corr", "corr_V1", "corr_hV4", "corr_early", "corr_ventral", "corr_FFA",
       "corr_PPA", "corr_EBA", "corr_VWFA", "train_corr", "gap"]
COLS = ["id", "iter", "idea", "dataset", "roi"] + NUM + ["status", "code"]


def connect(path=DB):
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.row_factory = sqlite3.Row
    return con


def init(path=DB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    defs = []
    for c in COLS:
        if c == "id":     defs.append("id TEXT PRIMARY KEY")
        elif c == "iter": defs.append("iter INTEGER")
        elif c in NUM:    defs.append(f"{c} REAL")
        else:             defs.append(f"{c} TEXT")
    with connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS models (%s)" % ", ".join(defs))
    return path


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def insert(row: dict, path=DB):
    init(path)   # ensure the namespace db + table exist (idempotent; first insert of a new dataset)
    r = {c: row.get(c) for c in COLS}
    for c in NUM:
        r[c] = _num(r[c])
    it = str(r.get("iter") or "").strip()
    r["iter"] = int(it) if it.isdigit() else None
    r["dataset"] = r.get("dataset") or "nsd"
    r["roi"] = r.get("roi") or "ventral"
    ph = ",".join("?" * len(COLS))
    with connect(path) as con:
        con.execute(f"INSERT OR REPLACE INTO models ({','.join(COLS)}) VALUES ({ph})",
                    [r[c] for c in COLS])


def migrate(csv_path, path=DB):
    init(path)
    n = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            insert(row, path)
            n += 1
    print(f"migrated {n} rows -> {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    if cmd == "init":
        print("init", init())
    elif cmd == "migrate":
        csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "search", "state", "results.csv")
        db_path = sys.argv[3] if len(sys.argv) > 3 else DB
        migrate(csv_path, db_path)
    else:
        sys.exit(f"unknown cmd {cmd!r} (use: init | migrate <csv>)")
