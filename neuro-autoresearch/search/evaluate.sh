#!/usr/bin/env bash
# search/evaluate.sh -- the SINGLE GPU evaluator loop (one of the two always-on loops search.sh
# supervises). Thin wrapper over search/evaluator.py, which draws work straight from the run's
# search.db (the queue = status='queued' rows; no queue directory, no wave barrier). One model on
# the GPU at a time; claim -> score with the fixed harness -> write the result back in the DB.
#   LAB_RUN=runs/nsd__2026-06-24 NEURO_DATASET=nsd search/evaluate.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DATASET="${NEURO_DATASET:-nsd}"
IMSIZE="${NEURO_IMSIZE:-128}"
echo "[evaluate] DB-backed evaluator (run=${LAB_RUN:-?} dataset=$DATASET imsize=$IMSIZE)"
exec python search/evaluator.py --dataset "$DATASET" --imsize "$IMSIZE"
