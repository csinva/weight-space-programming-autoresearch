#!/usr/bin/env python3
"""The single GPU evaluator loop -- one model on the GPU at a time, drawn from the run's DB.

The queue is just the ``status='queued'`` rows in ``runs/<dataset>__<date>/search.db``; this loop
claims them oldest-first via the ``hub`` API and scores each with ``hub.score`` (which runs the
model in an isolated subprocess against the FIXED harness, so a crashing model can't take the
loop down). There is no queue directory and no wave barrier.

    # Pool supervisor requeues orphaned rows once before starting any worker.
    loop forever:
        row = hub.get_next_model()                   # atomic queued -> running
        if none: sleep and poll again                # empty queue
        res = hub.score(row.path, ...)               # subprocess-isolated, time-capped, whole-dataset reward
        on error: low-effort repair edits a scratch copy and queues a new child (<= 2x)
        hub.record(id, res, status)                  # running -> ok | error

    LAB_RUN=runs/laion__2026-10-21 python search/evaluator/worker.py --dataset laion
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import tempfile
import shutil
import hashlib
import signal
import stat
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import hub  # noqa: E402  (the run API: get_next_model / score / record / requeue_orphans)


def _is_ok(res):
    return str(res.get("status", "")).strip().lower().startswith("ok")


def repair(model_file, log_file, rmodel, reffort, timeout=300):
    """Low-effort repair agent edits ``model_file`` in place to fix a small fault. Best-effort:
    a stuck/failed repair just leaves the model as-is for the final error."""
    with subprocess.Popen(["bash", os.path.join(REPO, "search", "evaluator", "repair.sh"),
                           model_file, log_file, rmodel, reffort, REPO],
                          cwd=REPO, start_new_session=True) as child:
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            print(f"[evaluator]   repair timed out after {timeout}s")


def _read_repair(path):
    """An agent may replace model.py with a symlink; never follow it on the host."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            raise ValueError("Repair must be a regular source file under 1 MB")
        return source.read(1_000_001)


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
    if hashlib.sha256(open(src, "rb").read()).hexdigest() != row.get("model_sha256"):
        hub.record(mid, {"status": "error: archived source identity changed"}, "error")
        return
    log_file = os.path.join(log_dir, f"{mid}.log")            # per-model; fed to repair, kept only if it ends failed
    res = score_guarded(mid, src, dataset, timeout)
    with open(log_file, "w") as f:
        f.write(res.get("log", "") or "")
    ok = _is_ok(res)
    status = hub.record(mid, res, "ok" if ok else "error")
    # Repairs are new submissions. The failed source and its hash never change.
    if not ok and max_repairs > 0 and res.get("repair_log"):
        depth = int(row.get("repair_depth") or 0)
        if depth < max_repairs:
            with tempfile.TemporaryDirectory(prefix="neuro-repair-") as directory:
                candidate = os.path.join(directory, "model.py")
                shutil.copyfile(src, candidate)
                safe_log = os.path.join(directory, "candidate-error.txt")
                with open(safe_log, "w") as file:
                    file.write(str(res["repair_log"])[-6000:])
                repair(candidate, safe_log, rmodel, reffort)
                try:
                    repaired = _read_repair(candidate)
                except (OSError, ValueError):
                    repaired = None
                if repaired is not None and repaired != open(src, "rb").read():
                    # Freeze bytes in a fresh directory the repair session never mounted.
                    with tempfile.TemporaryDirectory(prefix="neuro-repaired-submit-") as submit_dir:
                        submission = os.path.join(submit_dir, "model.py")
                        with open(submission, "wb") as file:
                            file.write(repaired)
                        child = hub.propose(submission, row.get("idea", ""), row.get("proposer"), parent_id=mid, repair_depth=depth + 1)
                    print(f"[evaluator] {mid}: repair queued as {child}")
    if ok:
        os.remove(log_file)
    print(f"[evaluator] {mid}: status={status} test_corr={res.get('test_corr', '-')} "
          f"(proposer={row.get('proposer')})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Single GPU evaluator (hub-backed).")
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "laion"))
    p.add_argument("--poll", type=float, default=5.0, help="empty-queue sleep (seconds)")
    p.add_argument("--timeout", type=int, default=int(os.environ.get("NEURO_EVAL_TIMEOUT", "420")))
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--repair-model", default=os.environ.get("NEURO_REPAIR_MODEL", "codex:astra"))
    p.add_argument("--repair-effort", default=os.environ.get("NEURO_REPAIR_EFFORT", "low"))
    a = p.parse_args(argv)

    from harness.protocols import validate_search_dataset
    a.dataset = validate_search_dataset(a.dataset)
    if os.environ.get("NEURO_METRIC", "rsa") != "rsa" or os.environ.get("NEURO_STAGE", "search") != "search":
        p.error("Search evaluators require NEURO_METRIC=rsa and NEURO_STAGE=search")
    os.environ["NEURO_METRIC"] = "rsa"
    os.environ["NEURO_STAGE"] = "search"
    run = hub.find_run()
    log_dir = os.path.join(run, "eval")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["NEURO_DATASET"] = a.dataset
    print(f"[evaluator] run={run} dataset={a.dataset}")
    # Recovery belongs to pool.sh, before peers start; resetting here races live claims.

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
