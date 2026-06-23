#!/usr/bin/env python3
"""Evaluator-side DB ops: startup crash recovery, the atomic claim, and result write-back.

The queue is just the ``status='queued'`` rows; the evaluator drains them oldest-wave-first.
Claiming uses a single atomic statement (``UPDATE ... WHERE id=(SELECT ... LIMIT 1)
RETURNING``), so one active evaluator can drain the queue without racing the proposers'
concurrent inserts under WAL + busy_timeout (``db.connect``). Multiple active evaluators for
one namespace would need a lease/owner column before startup recovery could be made safe.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db as _db  # noqa: E402  (db.connect = WAL + busy_timeout; db.NUM = the numeric score cols)


def requeue_orphans(db_path):
    """Startup crash recovery: send any ``running`` row back to ``queued``.

    A ``running`` row at startup was claimed but never finished -- the sole evaluator died
    mid-eval. Requeuing it makes the loop restart-safe (the model is simply re-claimed and
    re-scored). Returns the number of rows recovered.
    """
    if not os.path.exists(db_path):
        return 0
    with _db.connect(db_path) as con:
        return con.execute("UPDATE models SET status='queued' WHERE status='running'").rowcount


def claim_one(db_path):
    """Atomically claim the oldest queued row: ``queued -> running`` in ONE statement.

    Ordering by ``(iter, rowid)`` drains each wave in the order it was proposed -- ``rowid``
    is SQLite's monotonic insertion counter, the schema's creation-order proxy (there is no
    separate ``created_at`` column). Returns ``{id, path, dataset, roi, iter}`` for the
    claimed row, or ``None`` when the queue is empty.
    """
    if not os.path.exists(db_path):
        return None
    with _db.connect(db_path) as con:
        r = con.execute(
            "UPDATE models SET status='running' "
            "WHERE id = (SELECT id FROM models WHERE status='queued' "
            "            ORDER BY iter, rowid LIMIT 1) "
            "RETURNING id, path, dataset, roi, iter"
        ).fetchone()
    return dict(r) if r else None


def _to_float(v):
    f = _db._num(v)  # reuse the gateway's coercion; add the finite guard for score columns
    return f if f is not None and math.isfinite(f) else None  # NaN/inf -> store NULL, not NaN


def update_result(db_path, model_id, metrics, status):
    """Write-back: ``running -> ok|error`` and fill the numeric score columns the harness
    produced (``test_corr`` / region ``corr_*`` / ``train_corr`` / ``gap``).

    Only the score columns actually present in ``metrics`` are set -- so dataset-specific
    region panels need no special-casing -- and ``id``/``iter``/``idea``/``path``/``dataset``/
    ``roi`` are left untouched. ``status`` is canonicalised to ``ok`` or ``error``.
    """
    st = "ok" if str(status).strip().lower().startswith("ok") else "error"
    cols = [c for c in _db.NUM if metrics.get(c) not in (None, "")]
    assignments = [f"{c}=?" for c in cols] + ["status=?"]
    vals = [_to_float(metrics[c]) for c in cols] + [st, model_id]
    with _db.connect(db_path) as con:
        con.execute(f"UPDATE models SET {', '.join(assignments)} WHERE id=?", vals)
    return st


if __name__ == "__main__":  # tiny CLI, handy for ops / smoke tests
    import argparse

    p = argparse.ArgumentParser(description="Evaluator DB ops (claim / recovery).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("recover").add_argument("db")
    sub.add_parser("claim").add_argument("db")
    a = p.parse_args()
    if a.cmd == "recover":
        print(f"requeued {requeue_orphans(a.db)} orphan(s)")
    elif a.cmd == "claim":
        print(claim_one(a.db))
