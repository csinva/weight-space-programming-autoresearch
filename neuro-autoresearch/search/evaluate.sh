#!/usr/bin/env bash
# Sequentially evaluate every model in the per-dataset queue via the Copilot CLI, repairing
# small errors, and insert the results into that namespace's SQLite db (via search/record.py).
# The GPU is used one model at a time. Called by search/round.mjs each round; exits when the
# queue is drained. Parameterized by env: NEURO_DATASET / NEURO_ROI / NEURO_IMSIZE / NEURO_STUDY.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATASET="${NEURO_DATASET:-nsd}"
ROI="${NEURO_ROI:-ventral}"
IMSIZE="${NEURO_IMSIZE:-128}"
STUDY="${NEURO_STUDY:-NsdSubj01}"
export NEURO_DATASET="$DATASET" NEURO_ROI="$ROI" NEURO_STUDY="$STUDY"   # read by record.py
STATE="search/state/${STUDY}__${ROI}"                                  # per-dataset namespace
mkdir -p "$STATE/queue" "$STATE/done"

shopt -s nullglob
queued=("$STATE"/queue/*.py)
if [ ${#queued[@]} -eq 0 ]; then echo "[evaluate] queue empty"; exit 0; fi
echo "[evaluate] ${#queued[@]} models queued (dataset=$DATASET roi=$ROI imsize=$IMSIZE study=$STUDY)"

TASK="Evaluate every model in ${STATE}/queue/*.py SEQUENTIALLY, ONE AT A TIME (the GPU must never be shared). cwd is the repo root ${REPO}. For each file ${STATE}/queue/<id>.py, in filename order:
1. id=\$(basename <file> .py); cp the file to runs/lab-eval/model.py
2. run it WITH A TIME CAP (some models are slow/heavy): (cd runs/lab-eval && timeout 420 env NEURO_DATASET=${DATASET} NEURO_ROI=${ROI} NEURO_IMSIZE=${IMSIZE} NEURO_RUN=\$id python -m harness.run)  -- this writes runs/lab-eval/results/${STUDY}__${ROI}__\$id.csv
3. if it raises (import/shape/CUDA bug) or records status!=ok, READ the traceback and fix the small bug in BOTH runs/lab-eval/model.py and ${STATE}/queue/<id>.py, then rerun -- AT MOST TWICE. If it times out (exit 124, no result row) or is still broken after two fixes, leave it as status=error and MOVE ON -- never get stuck on one model.
4. record it: NEURO_DATASET=${DATASET} NEURO_ROI=${ROI} NEURO_STUDY=${STUDY} python search/record.py ${STATE}/queue/<id>.py
5. mv ${STATE}/queue/<id>.py ${STATE}/done/
Do them strictly one after another -- never run two models at once. When ${STATE}/queue/ is empty you are finished; stop. Do not propose new models; only evaluate the queue."

exec timeout 7200 copilot -p "$TASK" --allow-all --model claude-opus-4.8 --add-dir "$REPO"
