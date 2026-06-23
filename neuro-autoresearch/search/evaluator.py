#!/usr/bin/env python3
"""The long-lived GPU evaluator loop -- one model on the GPU at a time, pulled from the DB.

Mirrors the old ``evaluate.sh`` behaviour but draws work from the ``models`` table instead of
a queue directory: the queue is just the ``status='queued'`` rows, and the instant any turns
up the evaluator claims and runs it (it does not wait for a whole proposer wave to land).

    requeue orphaned 'running' rows                      # startup crash recovery
    loop forever:
        row = claim_one()                                # atomic queued -> running (oldest wave first)
        if none: sleep and poll again                    # empty queue
        copy row.path -> runs/lab-eval-<study>__<roi>/model.py
        run `python -m harness.run` (NEURO_DATASET/ROI from the row, 420s cap)
        on error/timeout: low-effort repair Copilot edits the model in place, rerun (<= 2x)
        write back test_corr / region corr_* / status ('ok' | 'error')

Launched once per namespace (one search, one GPU). Its lifetime and the search stop-condition
are the launcher's job (chunk 03); this module is just the worker. Knobs: ``--imsize`` (the
input resolution, default 128) and the repair Copilot's ``--repair-model`` / ``--repair-effort``.

    python search/evaluator.py --dataset nsd --roi ventral [--imsize 128]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "search"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import claim as _claim  # noqa: E402

# dataset key -> NeuralSet study name (matches each preset's DATASET.study in harness/datasets.py).
STUDY = {"nsd": "NsdSubj01", "tvsd": "TvsdMonkeyF", "nnn": "NnnV1"}


def ns_db(dataset, roi):
    study = STUDY.get(dataset, dataset)
    return os.path.join(REPO, "search", "state", f"{study}__{roi}", "results.db")


def eval_tag(study, roi):
    raw = study if roi in (None, "nsdgeneral") else f"{study}__{roi}"
    return "lab-eval-" + re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def ensure_eval_dir(tag, eval_dir):
    """Ensure the namespace's lab-eval run folder is a runnable workspace (symlinked harness/prompts + local
    model.py/experiment.py), built the same way scripts/setup_run.py does so
    ``python -m harness.run`` resolves. Rebuilds a missing or partial workspace; the dir is
    disposable scratch (gitignored), so a stale one is just removed and recreated."""
    if os.path.exists(os.path.join(eval_dir, "harness", "run.py")):
        return  # the harness entrypoint resolves -> the workspace is good
    import setup_run
    if os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)
    setup_run.setup_run(tag, REPO)


def _result_csv(eval_dir, study, roi, model_id):
    # Mirror harness.results._csv_path: drop the roi only for the all-voxel "nsdgeneral" reward.
    stem = study if roi in (None, "nsdgeneral") else f"{study}__{roi}"
    return os.path.join(eval_dir, "results", f"{stem}__{model_id}.csv")


def read_result(eval_dir, study, roi, model_id):
    """The last row of this model's harness result CSV as a dict, or ``None`` if it produced
    no file (failed to import/run before recording)."""
    csvp = _result_csv(eval_dir, study, roi, model_id)
    if not os.path.exists(csvp):
        return None
    with open(csvp, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def run_once(eval_dir, model_id, dataset, roi, study, imsize, timeout):
    """One harness eval pass under the model's dataset/roi (cap ``timeout`` s). Returns
    ``(metrics_or_None, ok, log_text)``; ``ok`` iff it exited cleanly AND recorded status=ok."""
    env = {
        **os.environ,
        "NEURO_DATASET": dataset,
        "NEURO_ROI": roi,
        "NEURO_IMSIZE": str(imsize),
        "NEURO_RUN": model_id,
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "harness.run"],
            cwd=eval_dir, env=env, capture_output=True, text=True, timeout=timeout,
        )
        rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = 124  # subprocess ran with text=True, so stdout/stderr are str|None
        out = (e.stdout or "") + (e.stderr or "")
        out += f"\nTIMEOUT after {timeout}s (model too slow/heavy)\n"
        out += f"[evaluator] killed: exceeded the {timeout}s cap\n"
    metrics = read_result(eval_dir, study, roi, model_id)
    ok = rc == 0 and metrics is not None and \
        str(metrics.get("status", "")).strip().lower().startswith("ok")
    return metrics, ok, out


def repair(model_file, done_log, repair_model, repair_effort, timeout=300):
    """Run the low-effort repair Copilot on ``model_file`` (edited in place). Best-effort: a
    failed/stuck repair just leaves the model as-is for the next rerun (or the final error)."""
    try:
        subprocess.run(
            ["bash", os.path.join(REPO, "search", "repair.sh"),
             model_file, done_log, repair_model, repair_effort, REPO],
            cwd=REPO, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[evaluator]   repair timed out after {timeout}s; rerunning as-is")


def process_model(db, row, eval_dir, imsize, timeout, max_repairs, repair_model, repair_effort):
    model_id = row["id"]
    dataset = row.get("dataset") or "nsd"
    roi = row.get("roi") or "ventral"
    study = STUDY.get(dataset, dataset)
    src = row.get("path")

    if not src or not os.path.exists(src):
        print(f"[evaluator] {model_id}: model file missing ({src!r}) -> error")
        _claim.update_result(db, model_id, {}, "error")
        return

    eval_model = os.path.join(eval_dir, "model.py")
    done_log = os.path.join(eval_dir, "last_eval.log")
    shutil.copyfile(src, eval_model)
    # Start this id's result CSV fresh so we read THIS evaluation, not a stale prior one.
    stale = _result_csv(eval_dir, study, roi, model_id)
    if os.path.exists(stale):
        os.remove(stale)

    repairs = 0
    while True:
        metrics, ok, log = run_once(eval_dir, model_id, dataset, roi, study, imsize, timeout)
        _write_log(eval_dir, log)
        if ok or repairs >= max_repairs:
            break
        repairs += 1
        print(f"[evaluator] {model_id}: failed -> repair {repairs}/{max_repairs}")
        repair(src, done_log, repair_model, repair_effort)    # edits the on-disk source in place
        shutil.copyfile(src, eval_model)                      # re-sync the eval copy

    status = _claim.update_result(db, model_id, metrics or {}, "ok" if ok else "error")
    tc = (metrics or {}).get("test_corr", "-")
    print(f"[evaluator] {model_id}: status={status} test_corr={tc} "
          f"(iter={row.get('iter')} repairs={repairs})")


def _write_log(eval_dir, text):
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "last_eval.log"), "w") as f:
        f.write(text or "")


def main(argv=None):
    p = argparse.ArgumentParser(description="Long-lived GPU evaluator (DB-backed).")
    p.add_argument("--dataset", default="nsd")
    p.add_argument("--roi", default="ventral")
    p.add_argument("--imsize", type=int, default=128)
    p.add_argument("--poll", type=float, default=5.0, help="empty-queue sleep (seconds)")
    p.add_argument("--timeout", type=int, default=420, help="per-model eval cap (seconds)")
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--repair-model", default="claude-opus-4.8")
    p.add_argument("--repair-effort", default="low")
    a = p.parse_args(argv)

    db = ns_db(a.dataset, a.roi)
    study = STUDY.get(a.dataset, a.dataset)
    tag = eval_tag(study, a.roi)
    eval_dir = os.path.join(REPO, "runs", tag)
    ensure_eval_dir(tag, eval_dir)
    print(f"[evaluator] dataset={a.dataset} roi={a.roi} imsize={a.imsize} db={db} eval_dir={eval_dir}")
    recovered = _claim.requeue_orphans(db)
    if recovered:
        print(f"[evaluator] crash recovery: requeued {recovered} orphaned 'running' row(s)")

    while True:
        row = None
        try:
            row = _claim.claim_one(db)
            if row is None:
                time.sleep(a.poll)
                continue
            process_model(db, row, eval_dir, a.imsize, a.timeout, a.max_repairs,
                          a.repair_model, a.repair_effort)
        except KeyboardInterrupt:
            print("\n[evaluator] stopped (any in-flight 'running' row is requeued on restart)")
            return
        except Exception as e:  # noqa: BLE001  -- one bad model must not kill the loop or
            print(f"[evaluator] unexpected error: {e}")          # deadlock the wave/barrier
            if row is not None:
                try:
                    _claim.update_result(db, row["id"], {}, "error")  # clear it -> wave can advance
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
