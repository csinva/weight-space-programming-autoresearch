#!/usr/bin/env bash
# search/proposer/pool.sh -- the N-PROPOSER POOL loop (one of the two always-on loops run.sh
# supervises). Launches NEURO_FLEET headless proposers (search/proposer/worker.sh p00..p<N-1>), one per
# sandbox, and keeps the pool full: a worker that dies is restarted until the wall-clock deadline.
# It owns only pool size + liveness, NOT the stop-condition (run.sh signals this whole group).
#
# Per-slot MODEL mix: by DEFAULT the fleet is spread round-robin across a ROSTER of distinct models
# (Opus 4.8 / 4.7 / 4.6, GPT-5.5, Gemini 3.1 Pro) -- so for a pool of 10 each model drives 2 slots,
# blending strong and architecturally-different proposers. Override the roster via NEURO_MODEL_ROSTER
# (space-separated, each entry "model" or "model:effort"). A per-entry ":effort" pins that model's
# thinking effort (Gemini is pinned to high by default); every other slot inherits the pool-wide
# NEURO_EFFORT (set centrally by run.sh's --effort; default xhigh).
#   LAB_RUN=... NEURO_DATASET=... NEURO_FLEET=10 NEURO_DEADLINE=<epoch> search/proposer/pool.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/search/lib.sh"
WORKER="$REPO/search/proposer/worker.sh"
FLEET="${NEURO_FLEET:-10}"
DEADLINE="${NEURO_DEADLINE:-0}"
[ -f "$WORKER" ] || { echo "[propose] missing $WORKER" >&2; exit 1; }
: "${LAB_RUN:?LAB_RUN must be set}"

EFFORT="${NEURO_EFFORT:-xhigh}"              # the ONE pool-wide effort knob (run.sh --effort)
# The model roster: round-robin across the fleet. Each entry is "model" or "model:effort"; a
# ":effort" suffix pins that model's reasoning effort (else it inherits the pool-wide $EFFORT).
DEFAULT_ROSTER="claude-opus-4.8 claude-opus-4.7 claude-opus-4.6 gpt-5.5 gpt-5.6-sol"
read -r -a ROSTER <<< "${NEURO_MODEL_ROSTER:-$DEFAULT_ROSTER}"
NROSTER=${#ROSTER[@]}
echo "[propose] pool of $FLEET (run=$LAB_RUN dataset=${NEURO_DATASET:-nsd}) base-effort=$EFFORT -- roster (round-robin): ${ROSTER[*]}"

declare -a PIDS
start_slot() {
  local i="$1" entry m e
  entry="${ROSTER[$((i % NROSTER))]}"   # round-robin assignment
  m="${entry%%:*}"                       # model id (before any ":effort")
  if [ "$entry" = "$m" ]; then e="$EFFORT"; else e="${entry#*:}"; fi
  NEURO_PROPOSER_MODEL="$m" NEURO_PROPOSER_EFFORT="$e" bash "$WORKER" "$(printf 'p%02d' "$i")" &
  PIDS[$i]=$!
  echo "[propose] $(printf 'p%02d' "$i") up (pid ${PIDS[$i]}, $m/$e)"
}
shutdown() { trap '' INT TERM; echo "[propose] stopping ${#PIDS[@]} workers..."; for pid in "${PIDS[@]:-}"; do term "$pid"; done; wait 2>/dev/null || true; }
install_shutdown_trap shutdown

for i in $(seq 0 $((FLEET - 1))); do start_slot "$i"; done
while ! past_deadline "$DEADLINE"; do
  for i in $(seq 0 $((FLEET - 1))); do
    if ! alive "${PIDS[$i]:-0}"; then
      wait "${PIDS[$i]:-0}" 2>/dev/null || true
      past_deadline "$DEADLINE" && break
      echo "[propose] $(printf 'p%02d' "$i") exited; restarting" >&2
      start_slot "$i"
    fi
  done
  sleep 5
done
echo "[propose] deadline reached; pool draining."
