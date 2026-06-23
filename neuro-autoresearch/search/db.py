#!/usr/bin/env python3
"""SQLite gateway for the open-ended search database (``search/state/<study>__<roi>/results.db``).

One table ``models`` -- one row per model -- is the **single home for both history and the
queue**. There is no ``queue/`` + ``done/`` directory handoff and nothing is ever moved
between dirs: a row carries a ``path`` pointer to where its ``model.py`` lives on disk plus a
``status`` flag, and the queue advances by updating that flag in place.

Status is an explicit state machine (see ``STATUSES``)::

    queued   proposer wrote the row (its model file is on disk; awaiting evaluation)
      |
      v
    running  evaluator has claimed the row (one model on the GPU at a time)
      |
      +--> ok      scored cleanly; has a ``test_corr``
      +--> error   raised / bad shape / timeout

The model source stays on the filesystem -- the row keeps only a ``path`` pointer to its
``model.py``, never an inlined code column, so the DB stays small and diff-friendly and the
harness loads the file as it already does. Python owns all writes (WAL mode + busy_timeout, so
concurrent proposer inserts and evaluator updates coexist safely); ``search/lib/db.mjs`` reads
the same file via ``node:sqlite``. Each ``search/state/<study>__<roi>/results.db`` is the unit
of isolation -- both loops target one namespace and datasets never mix.

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
# A row is its own queue entry: a ``path`` pointer to the on-disk model file (NOT an inlined
# code column) plus the ``status`` state flag.
COLS = ["id", "iter", "idea", "dataset", "roi"] + NUM + ["status", "path"]

# The status lifecycle. The queue is just these flags advancing in place:
#   queued -> running -> ok | error
STATUSES = ("queued", "running", "ok", "error")


def _coltype(c):
    if c == "iter":  return "INTEGER"
    if c in NUM:     return "REAL"
    return "TEXT"


def connect(path=DB):
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.row_factory = sqlite3.Row
    return con


def _ensure_columns(con):
    """Forward-migrate an existing table: add any column missing from ``COLS`` in place.

    Lets a db created by an older schema (e.g. one with a ``code`` column and no ``path``)
    gain the ``path`` pointer without a rebuild. Old columns are left untouched (harmless).
    """
    have = {r["name"] for r in con.execute("PRAGMA table_info(models)")}
    if not have:
        return
    for c in COLS:
        if c not in have:
            con.execute(f"ALTER TABLE models ADD COLUMN {c} {_coltype(c)}")


def init(path=DB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    defs = []
    for c in COLS:
        if c == "id":  defs.append("id TEXT PRIMARY KEY")
        else:          defs.append(f"{c} {_coltype(c)}")
    with connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS models (%s)" % ", ".join(defs))
        _ensure_columns(con)
    return path


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _status(v):
    """Canonicalise any status to the four-state machine (``STATUSES``). Empty -> ``queued`` (a
    freshly written row starts queued); a verbose ``error: ...`` collapses to ``error``."""
    s = (v or "").strip().lower()
    if not s:
        return "queued"
    return next((st for st in STATUSES if s.startswith(st)), "error")


def insert(row: dict, path=DB):
    init(path)   # ensure the namespace db + table exist (idempotent; first insert of a new dataset)
    r = {c: row.get(c) for c in COLS}
    for c in NUM:
        r[c] = _num(r[c])
    it = str(r.get("iter") or "").strip()
    r["iter"] = int(it) if it.isdigit() else None
    r["dataset"] = r.get("dataset") or "nsd"
    r["roi"] = r.get("roi") or "ventral"
    r["status"] = _status(r.get("status"))
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
        print("init", init(sys.argv[2] if len(sys.argv) > 2 else DB))
    elif cmd == "migrate":
        csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "search", "state", "results.csv")
        db_path = sys.argv[3] if len(sys.argv) > 3 else DB
        migrate(csv_path, db_path)
    else:
        sys.exit(f"unknown cmd {cmd!r} (use: init | migrate <csv>)")
