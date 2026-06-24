#!/usr/bin/env bash
# search/propose.sh -- the N-PROPOSER POOL loop (one of the two always-on loops search.sh
# supervises). Launches NEURO_FLEET headless proposers (search/proposer.sh p00..p<N-1>), one per
# sandbox, and keeps the pool full: a worker that dies is restarted until the wall-clock deadline.
# It owns only pool size + liveness, NOT the stop-condition (search.sh signals this whole group).
#
# Per-slot model mix: by DEFAULT the first half of the fleet drives on Opus 4.8 (high effort) and
# the second half on GPT-5.5 (medium) -- so the pool blends a strong and a different model. Tune
# the split with NEURO_OPUS_FRACTION (0..1, default 0.5) or override the two models/efforts via
# NEURO_MODEL_A/EFFORT_A (first half) and NEURO_MODEL_B/EFFORT_B (second half).
#   LAB_RUN=... NEURO_DATASET=... NEURO_FLEET=10 NEURO_DEADLINE=<epoch> search/propose.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="$REPO/search/proposer.sh"
FLEET="${NEURO_FLEET:-10}"
DEADLINE="${NEURO_DEADLINE:-0}"
[ -f "$WORKER" ] || { echo "[propose] missing $WORKER" >&2; exit 1; }
: "${LAB_RUN:?LAB_RUN must be set}"

MODEL_A="${NEURO_MODEL_A:-claude-opus-4.8}"; EFFORT_A="${NEURO_EFFORT_A:-medium}"   # first half
MODEL_B="${NEURO_MODEL_B:-gpt-5.5}";         EFFORT_B="${NEURO_EFFORT_B:-medium}"   # second half
FRACTION="${NEURO_OPUS_FRACTION:-0.5}"
# Number of slots on model A (the first split), rounded; the rest go to model B.
SPLIT=$(awk -v n="$FLEET" -v f="$FRACTION" 'BEGIN { printf "%d", (n * f) + 0.5 }')
echo "[propose] pool of $FLEET (run=$LAB_RUN dataset=${NEURO_DATASET:-nsd}) -- slots 0..$((SPLIT-1)): $MODEL_A/$EFFORT_A | $SPLIT..$((FLEET-1)): $MODEL_B/$EFFORT_B"

declare -a PIDS
start_slot() {
  local i="$1" m e
  if [ "$i" -lt "$SPLIT" ]; then m="$MODEL_A"; e="$EFFORT_A"; else m="$MODEL_B"; e="$EFFORT_B"; fi
  NEURO_PROPOSER_MODEL="$m" NEURO_PROPOSER_EFFORT="$e" bash "$WORKER" "$(printf 'p%02d' "$i")" &
  PIDS[$i]=$!
  echo "[propose] $(printf 'p%02d' "$i") up (pid ${PIDS[$i]}, $m/$e)"
}
alive() { kill -0 "$1" 2>/dev/null; }
past_deadline() { [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; }
shutdown() { trap '' INT TERM; echo "[propose] stopping ${#PIDS[@]} workers..."; for pid in "${PIDS[@]:-}"; do [ -n "${pid:-}" ] && kill -TERM "$pid" 2>/dev/null; done; wait 2>/dev/null || true; }
trap 'exit 130' INT TERM
trap shutdown EXIT

for i in $(seq 0 $((FLEET - 1))); do start_slot "$i"; done
while ! past_deadline; do
  for i in $(seq 0 $((FLEET - 1))); do
    if ! alive "${PIDS[$i]:-0}"; then
      wait "${PIDS[$i]:-0}" 2>/dev/null || true
      past_deadline && break
      echo "[propose] $(printf 'p%02d' "$i") exited; restarting" >&2
      start_slot "$i"
    fi
  done
  sleep 5
done
echo "[propose] deadline reached; pool draining."
