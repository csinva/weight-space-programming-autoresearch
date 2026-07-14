#!/usr/bin/env bash
# search/evaluate.sh -- the GPU evaluator loop (one of the two always-on loops search.sh
# supervises). Thin wrapper over search/evaluator.py, which draws work straight from the run's
# search.db (the queue = status='queued' rows; no queue directory, no wave barrier). Each
# evaluator claims one model at a time (atomic queued->running) and scores it with the fixed
# harness in an isolated subprocess. NEURO_EVALUATORS (default 3) fans out that many evaluators
# against the SAME queue -- the atomic claim + WAL busy_timeout keep them from colliding, so they
# run models concurrently on the GPU (set NEURO_EVAL_GPUS to spread the workers across cards). If
# any one dies, `wait` returns and search.sh restarts the
# whole wrapper; requeue_orphans recovers any in-flight `running` row losslessly.
# Image resolution is a dataset-PRESET attribute (harness/datasets.py), not a search knob.
#   LAB_RUN=runs/nsd__2026-06-24 NEURO_DATASET=nsd NEURO_EVALUATORS=3 search/evaluate.sh
#   NEURO_EVALUATORS=8 NEURO_EVAL_GPUS=0,1,2,3 ...  # 8 workers, round-robin 2 per GPU across 4 cards
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DATASET="${NEURO_DATASET:-nsd}"
N="${NEURO_EVALUATORS:-3}"
case "$N" in ''|*[!0-9]*|0) echo "[evaluate] NEURO_EVALUATORS must be a positive integer (got '$N')" >&2; exit 2;; esac
echo "[evaluate] $N concurrent DB-backed evaluator(s) (run=${LAB_RUN:-?} dataset=$DATASET)"
if [ "$N" -eq 1 ]; then
  exec python search/evaluator.py --dataset "$DATASET"
fi
# Fan out N evaluators against the same queue; if any exits, terminate the rest so the wrapper
# exits and search.sh restarts a clean set.
pids=""
term() { trap '' INT TERM; for p in $pids; do kill "$p" 2>/dev/null; done; }
trap term INT TERM
# NEURO_EVAL_GPUS (comma list, e.g. "0,1,2,3"): pin each worker to a physical GPU round-robin so
# the fan-out spreads across cards instead of all landing on cuda:0. Empty = inherit CUDA_VISIBLE_DEVICES.
IFS=',' read -r -a CARDS <<< "${NEURO_EVAL_GPUS:-}"
for i in $(seq 0 $((N - 1))); do
  pin=""; [ "${#CARDS[@]}" -gt 0 ] && pin="CUDA_VISIBLE_DEVICES=${CARDS[$((i % ${#CARDS[@]}))]}"
  env $pin python search/evaluator.py --dataset "$DATASET" & pids="$pids $!"
done
if help wait 2>/dev/null | grep -q '\-n'; then
  wait -n || true          # returns when the FIRST evaluator exits (status irrelevant)
else
  wait || true             # old bash: blocks until all exit
fi
term
wait 2>/dev/null || true
