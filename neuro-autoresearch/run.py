#!/usr/bin/env python3
"""Boot the open-ended ventral-model search.

Two engines, one shared memory:
  * Claude Code (this conductor) drives the loop and does the PARALLEL search via its
    built-in dynamic Workflow -- each iteration, many agents invent new untrained
    `model.py` files and drop them in `lab/queue/`.
  * The Copilot CLI is the SEQUENTIAL evaluator -- it runs every queued model one at a
    time (fixing small bugs), so the GPU is never shared, and appends scores to
    `lab/results.csv` (the database + the only memory that needs to persist).

`run.py` sets up the `lab/` workspace + an isolated eval run-folder, then launches a
persistent Claude Code conductor session and keeps it alive (relaunching with
`--continue`, which preserves its context) until the time budget is spent.

    python run.py                 # default 5h on NSD ventral @ 128px
    python run.py --hours 8
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys, time

REPO = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(REPO, "lab")
COLUMNS = ["id", "iter", "idea", "test_corr", "corr_V1", "corr_hV4", "corr_early",
           "corr_ventral", "corr_FFA", "corr_PPA", "corr_EBA", "corr_VWFA",
           "train_corr", "gap", "status", "code"]

CONDUCTOR = """You are the CONDUCTOR of an OPEN-ENDED search for an UNTRAINED model that \
maximises held-out NSD ventral test_corr. Trained AlexNet = 0.2838 (the bar to beat); \
target = 0.4257; noise ceiling ~0.50. The database is lab/results.csv -- every model ever \
tried, with its idea, per-region scores (corr_V1..corr_VWFA) and FULL source code. \
Substrate: NSD ventral at NEURO_IMSIZE=128.

Run this loop and KEEP GOING -- never stop on your own, never ask questions:
1. PROPOSE (parallel): increment the integer in lab/ITER (read it, add 1, write it back). \
Then call the Workflow tool with scriptPath "lab/propose.js" -- it fans out many agents \
that read lab/results.csv and write newly-invented untrained model.py files into lab/queue/.
2. EVALUATE (sequential): run `bash lab/evaluate.sh` and WAIT for it to finish. It spins up \
the Copilot CLI, which runs every queued model one-by-one (repairing small bugs) and appends \
results to lab/results.csv.
3. REVIEW: read lab/results.csv, note the new best ventral test_corr, then go back to step 1.

Do NOT evaluate models yourself; the Copilot CLI does that. Just run the loop forever."""

CONTINUE = "Continue the search loop (propose -> evaluate -> review -> repeat). Keep going; never stop."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=5.0, help="wall-clock budget")
    args = ap.parse_args()

    os.makedirs(os.path.join(LAB, "queue"), exist_ok=True)
    os.makedirs(os.path.join(LAB, "done"), exist_ok=True)
    results = os.path.join(LAB, "results.csv")
    if not os.path.exists(results):
        with open(results, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)
    iter_file = os.path.join(LAB, "ITER")
    if not os.path.exists(iter_file):
        open(iter_file, "w").write("0")
    # isolated eval run-folder (symlinks the fixed harness; per-model results land here)
    if not os.path.isdir(os.path.join(REPO, "runs", "lab-eval")):
        subprocess.run([sys.executable, "scripts/setup_run.py", "lab-eval"], cwd=REPO, check=True)

    deadline = time.time() + args.hours * 3600
    first = True
    while time.time() < deadline:
        remaining = int(deadline - time.time())
        if remaining < 60:
            break
        cmd = ["claude", "--dangerously-skip-permissions", "-p", CONDUCTOR if first else CONTINUE]
        if not first:
            cmd.insert(1, "--continue")
        print(f"[run.py] launching conductor ({remaining // 60} min left)", flush=True)
        try:
            subprocess.run(cmd, cwd=REPO, timeout=remaining)
        except subprocess.TimeoutExpired:
            print("[run.py] budget reached; conductor stopped.")
            break
        first = False  # conductor exited early -> relaunch with --continue (keeps its context)
    print("[run.py] done.")


if __name__ == "__main__":
    main()
