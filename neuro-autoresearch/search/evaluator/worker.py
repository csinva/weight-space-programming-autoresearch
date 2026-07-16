#!/usr/bin/env python3
"""The single GPU evaluator loop -- one model on the GPU at a time, drawn from the run's DB.

The queue is just the ``status='queued'`` rows in ``runs/<dataset>__<date>/search.db``; this loop
claims them oldest-first via the ``hub`` API and scores each with ``hub.score`` (which runs the
model in an isolated subprocess against the FIXED harness, so a crashing model can't take the
loop down). There is no queue directory and no wave barrier.

    requeue orphaned 'running' rows                  # startup crash recovery
    loop forever:
        row = hub.get_next_model()                   # atomic queued -> running
        if none: sleep and poll again                # empty queue
        res = hub.score(row.path, ...)               # subprocess-isolated, time-capped, whole-dataset reward
        on error/timeout: low-effort repair Copilot edits the model in place, rerun (<= 2x)
        hub.record(id, res, status)                  # running -> ok | error

    LAB_RUN=runs/nsd__2026-06-24 python search/evaluator/worker.py --dataset nsd
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import hub  # noqa: E402  (the run API: get_next_model / score / record / requeue_orphans)


def _is_ok(res):
    return str(res.get("status", "")).strip().lower().startswith("ok")


def repair(model_file, log_file, rmodel, reffort, timeout=300):
    """Low-effort repair Copilot edits ``model_file`` in place to fix a small fault. Best-effort:
    a stuck/failed repair just leaves the model as-is for the final error."""
    try:
        subprocess.run(["bash", os.path.join(REPO, "search", "evaluator", "repair.sh"),
                        model_file, log_file, rmodel, reffort, REPO], cwd=REPO, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[evaluator]   repair timed out after {timeout}s; recording as-is")


def score_guarded(mid, src, dataset, timeout):
    """``hub.score`` under a hard wall-clock watchdog. In-process scoring can't interrupt a model hung
    in native CUDA/C, so a watchdog thread records the model as errored and hard-exits on timeout; the
    supervisor restarts this evaluator and ``requeue_orphans`` leaves the now-errored row alone, so the
    queue moves past the poison model instead of wedging forever."""
    done = threading.Event()
    def watchdog():
        if not done.wait(timeout):
            os.write(2, f"[evaluator] {mid}: exceeded {timeout}s -> error + restart\n".encode())  # raw fd: hub.score redirects sys.stderr
            hub.record(mid, {}, "error")
            os._exit(70)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        return hub.score(src, dataset)
    finally:
        done.set()


def process_model(row, log_dir, dataset, timeout, max_repairs, rmodel, reffort):
    mid, src = row["id"], row.get("path")
    if not src or not os.path.exists(src):
        print(f"[evaluator] {mid}: model file missing ({src!r}) -> error")
        hub.record(mid, {}, "error")
        return
    log_file = os.path.join(log_dir, f"{mid}.log")            # per-model; fed to repair, kept only if it ends failed
    repairs = 0
    while True:
        res = score_guarded(mid, src, dataset, timeout)       # watchdog-guarded; never raises
        with open(log_file, "w") as f:
            f.write(res.get("log", "") or "")
        ok = _is_ok(res)
        if ok or repairs >= max_repairs:
            break
        repairs += 1
        print(f"[evaluator] {mid}: failed -> repair {repairs}/{max_repairs}")
        repair(src, log_file, rmodel, reffort)                # edits the on-disk source in place
    if ok:
        os.remove(log_file)                                   # success leaves no log; keep only failures
    status = hub.record(mid, res, "ok" if ok else "error")
    print(f"[evaluator] {mid}: status={status} test_corr={res.get('test_corr', '-')} "
          f"(proposer={row.get('proposer')} repairs={repairs})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Single GPU evaluator (hub-backed).")
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "nsd"))
    p.add_argument("--poll", type=float, default=5.0, help="empty-queue sleep (seconds)")
    p.add_argument("--timeout", type=int, default=int(os.environ.get("NEURO_EVAL_TIMEOUT", "420")))
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--repair-model", default=os.environ.get("NEURO_REPAIR_MODEL", "claude-opus-4.8"))
    p.add_argument("--repair-effort", default=os.environ.get("NEURO_REPAIR_EFFORT", "low"))
    a = p.parse_args(argv)

    run = hub.find_run()
    log_dir = os.path.join(run, "eval")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["NEURO_DATASET"] = a.dataset
    print(f"[evaluator] run={run} dataset={a.dataset}")
    recovered = hub.requeue_orphans(run)
    if recovered:
        print(f"[evaluator] crash recovery: requeued {recovered} orphaned 'running' row(s)")

    while True:
        row = None
        try:
            row = hub.get_next_model(run)
            if row is None:
                time.sleep(a.poll)
                continue
            process_model(row, log_dir, a.dataset, a.timeout, a.max_repairs,
                          a.repair_model, a.repair_effort)
        except KeyboardInterrupt:
            print("\n[evaluator] stopped (any in-flight 'running' row requeues on restart)")
            return
        except Exception as e:  # noqa: BLE001  -- one bad model must not kill the loop
            print(f"[evaluator] unexpected error: {e}")
            if row is not None:
                try:
                    hub.record(row["id"], {}, "error")
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
