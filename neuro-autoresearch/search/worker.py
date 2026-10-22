#!/usr/bin/env python3
"""Consume queue submissions, score RSA, and archive bounded repair children."""
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
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from search import queue
from search.score import score
# noqa: E402  (the run API: get_next_model / score / record / requeue_orphans)


def _is_ok(res):
    return str(res.get("status", "")).strip().lower().startswith("ok")


def repair(model_file, log_file, rmodel, reffort, timeout=300):
    """Low-effort repair agent edits ``model_file`` in place to fix a small fault. Best-effort:
    a stuck/failed repair just leaves the model as-is for the final error."""
    prompt = ("Fix only small mechanical faults in model.py. Keep the fixed, untrained, "
              "per-image representation and core idea unchanged. Edit only model.py, then stop. "
              "Do not run it. Model must define MODEL = Model() and return finite (N, features) "
              "activations for flattened RGB images. Error:\n" + open(log_file).read()[-6000:])
    with subprocess.Popen([sys.executable, "-m", "search.providers", "run", "--agent", rmodel,
                           "--effort", reffort, "--role", "repair", "--cwd", os.path.dirname(model_file)],
                          stdin=subprocess.PIPE, text=True, cwd=REPO, start_new_session=True) as child:
        try:
            child.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def _read_repair(path):
    """An agent may replace model.py with a symlink; never follow it on the host."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            raise ValueError("Repair must be a regular source file under 1 MB")
        return source.read(1_000_001)


def score_guarded(mid, src, dataset, timeout):
    """``score`` under a hard wall-clock watchdog. In-process scoring can't interrupt a model hung
    in native CUDA/C, so a watchdog thread records the model as errored and hard-exits on timeout; the
    supervisor restarts this evaluator and ``requeue_orphans`` leaves the now-errored row alone, so the
    queue moves past the poison model instead of wedging forever."""
    done = threading.Event()
    def watchdog():
        if not done.wait(timeout):
            os.write(2, f"[evaluator] {mid}: exceeded {timeout}s -> error + restart\n".encode())  # raw fd: score redirects sys.stderr
            queue.record(mid, {}, "error")
            os._exit(70)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        return score(src, dataset)
    finally:
        done.set()


def process_model(row, log_dir, dataset, timeout, max_repairs, rmodel, reffort):
    mid, src = row["id"], row.get("path")
    started = time.monotonic()
    print(f"[evaluator] {datetime.now(timezone.utc).isoformat()} {mid}: started "
          f"proposer={row.get('proposer')}", flush=True)
    if not src or not os.path.exists(src):
        print(f"[evaluator] {mid}: model file missing ({src!r}) -> error")
        queue.record(mid, {}, "error")
        return
    if hashlib.sha256(open(src, "rb").read()).hexdigest() != row.get("model_sha256"):
        queue.record(mid, {"status": "error: archived source identity changed"}, "error")
        return
    log_file = os.path.join(log_dir, f"{mid}.log")            # per-model; fed to repair, kept only if it ends failed
    res = score_guarded(mid, src, dataset, timeout)
    with open(log_file, "w") as f:
        f.write(res.get("log", "") or "")
    ok = _is_ok(res)
    status = queue.record(mid, res, "ok" if ok else "error")
    print(f"[evaluator] {datetime.now(timezone.utc).isoformat()} {mid}: completed "
          f"status={status} test_corr={res.get('test_corr', '-')} "
          f"wall_seconds={time.monotonic() - started:.2f}", flush=True)
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
                        child = queue.propose(submission, row.get("idea", ""), row.get("proposer"), parent_id=mid, repair_depth=depth + 1)
                    print(f"[evaluator] {mid}: repair queued as {child}")
    if ok:
        os.remove(log_file)


def main(argv=None):
    p = argparse.ArgumentParser(description="Single GPU evaluator (queue-backed).")
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "laion"))
    p.add_argument("--poll", type=float, default=5.0, help="empty-queue sleep (seconds)")
    p.add_argument("--timeout", type=int, default=int(os.environ.get("NEURO_EVAL_TIMEOUT", "420")))
    p.add_argument("--max-repairs", type=int, default=0,
                   help="Optional automatic repair depth; default 0 leaves repairs to proposers")
    p.add_argument("--repair-model", default=os.environ.get("NEURO_REPAIR_MODEL", "codex:astra"))
    p.add_argument("--repair-effort", default=os.environ.get("NEURO_REPAIR_EFFORT", "low"))
    a = p.parse_args(argv)

    from neuro_eval.benchmarks import validate_search_dataset
    a.dataset = validate_search_dataset(a.dataset)
    if os.environ.get("NEURO_METRIC", "rsa") != "rsa" or os.environ.get("NEURO_STAGE", "search") != "search":
        p.error("Search evaluators require NEURO_METRIC=rsa and NEURO_STAGE=search")
    os.environ["NEURO_METRIC"] = "rsa"
    os.environ["NEURO_STAGE"] = "search"
    run = queue.find_run()
    log_dir = os.path.join(run, "eval")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["NEURO_DATASET"] = a.dataset
    print(f"[evaluator] run={run} dataset={a.dataset}")
    # Recovery belongs to the supervisor, before peers start; resetting here races live claims.

    while True:
        row = None
        try:
            row = queue.get_next_model(run)
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
                    queue.record(row["id"], {}, "error")
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
