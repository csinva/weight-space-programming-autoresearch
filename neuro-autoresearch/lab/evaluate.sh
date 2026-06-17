#!/usr/bin/env bash
# Sequentially evaluate every model in lab/queue/ via the Copilot CLI, repairing small
# errors, and append the results to lab/results.csv. The GPU is used one model at a time
# (no throttling). Called by the conductor each iteration; exits when the queue is drained.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

shopt -s nullglob
queued=(lab/queue/*.py)
if [ ${#queued[@]} -eq 0 ]; then echo "[evaluate] queue empty"; exit 0; fi
echo "[evaluate] ${#queued[@]} models queued"

TASK="Evaluate every model in lab/queue/*.py SEQUENTIALLY, ONE AT A TIME (the GPU must never be shared). cwd is the repo root ${REPO}. For each file lab/queue/<id>.py, in filename order:
1. id=\$(basename <file> .py); cp the file to runs/lab-eval/model.py
2. run it WITH A TIME CAP (some models are slow/heavy): (cd runs/lab-eval && timeout 420 env NEURO_DATASET=nsd NEURO_ROI=ventral NEURO_IMSIZE=128 NEURO_RUN=\$id python -m harness.run)  -- this writes runs/lab-eval/results/NsdSubj01__ventral__\$id.csv
3. if it raises (import/shape/CUDA bug) or records status!=ok, READ the traceback and fix the small bug in BOTH runs/lab-eval/model.py and lab/queue/<id>.py, then rerun -- AT MOST TWICE. If it times out (exit 124, no result row) or is still broken after two fixes, leave it as status=error and MOVE ON -- never get stuck on one model.
4. record it: python lab/record.py lab/queue/<id>.py
5. mv lab/queue/<id>.py lab/done/
Do them strictly one after another -- never run two models at once. When lab/queue/ is empty you are finished; stop. Do not propose new models; only evaluate the queue."

exec timeout 7200 copilot -p "$TASK" --allow-all --model claude-opus-4.8 --add-dir "$REPO"
