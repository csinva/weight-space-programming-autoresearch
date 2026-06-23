#!/usr/bin/env python3
"""Iteration barrier for the proposer pool -- the deterministic per-wave re-sync.

A *wave* is the set of rows sharing an ``iter``. Before a proposer emits its next model it
waits until the latest wave is fully scored, so every proposal is made against a fully
evaluated DB (this replaces a queue-depth backpressure cap: there is no depth cap at all).
The barrier is pure DB state -- there is no external coordinator and no clock:

    cur = MAX(iter)                                                   # latest wave (0 if empty)
    spin while COUNT(iter = cur AND status IN ('queued','running')) > 0   # wait it out
    -> the proposer then stamps its next model with iter = cur + 1

With the whole pool blocked on the same wave clearing, the workers wake together and -- since
no ``iter`` advances until someone enqueues, and a worker spends real time writing its model
before enqueuing -- they all read the same ``cur`` and land in the same next wave. So waves
self-organise to ~N through DB inserts alone; a worker that falls behind simply joins a later
wave (the barrier "lightly re-syncs" the pool rather than locking it in perfect lockstep).

The wait BLOCKS inside Python (a fixed sleep between retries), so a headless Copilot proposer
spends no tokens spinning -- one tool call parks until the wave clears and prints the iter to
stamp.

CLI::

    python search/barrier.py wait   <db> [--sleep 5]   # BLOCKS until clear; prints ITER=<n>
    python search/barrier.py status <db>               # cur + pending count (no block)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db as _db  # noqa: E402  (db.connect = WAL + busy_timeout, the shared connection settings)

PENDING = ("queued", "running")  # a wave is "open" while any of its rows is unscored
POLL = float(os.environ.get("NEURO_BARRIER_POLL", "5"))  # fixed sleep between retries (seconds)


def wave_state(db_path):
    """One snapshot of the latest wave: ``(cur, pending)`` -- ``cur`` is ``MAX(iter)`` (0 on a
    fresh/empty DB) and ``pending`` is how many of that wave's rows are still unscored
    (``queued``/``running``). A single connection + query so the poll loop costs one round-trip.
    """
    if not os.path.exists(db_path):
        return 0, 0
    ph = ",".join("?" * len(PENDING))
    with _db.connect(db_path) as con:
        try:
            r = con.execute(
                "SELECT COALESCE(MAX(iter), 0) AS cur, "
                f"COALESCE(SUM(CASE WHEN status IN ({ph}) THEN 1 ELSE 0 END), 0) AS pending "
                "FROM models WHERE iter = (SELECT MAX(iter) FROM models)",
                PENDING,
            ).fetchone()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():  # cold start: db exists but models table does not
                return 0, 0
            raise
    return int(r["cur"]), int(r["pending"])


def wait(db_path, sleep=POLL):
    """Block until the latest wave is fully scored, then return the iter to stamp (``cur+1``).

    ``cur`` is re-read each retry: if the pool advances while we wait we simply track the new
    latest wave, so we always unblock against a fully-scored DB and never stamp a stale iter.
    """
    cur, pending = wave_state(db_path)
    while pending > 0:
        time.sleep(sleep)
        cur, pending = wave_state(db_path)
    return cur + 1


def main(argv=None):
    p = argparse.ArgumentParser(description="Proposer iteration barrier.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("wait", help="block until the latest wave clears; print ITER=<n>")
    sp.add_argument("db"); sp.add_argument("--sleep", type=float, default=POLL)
    sp = sub.add_parser("status", help="print latest wave + pending count, no block")
    sp.add_argument("db")
    a = p.parse_args(argv)

    if a.cmd == "wait":
        print(f"ITER={wait(a.db, a.sleep)}")
    elif a.cmd == "status":
        cur, pending = wave_state(a.db)
        print(f"cur={cur} pending={pending}")


if __name__ == "__main__":
    main()
