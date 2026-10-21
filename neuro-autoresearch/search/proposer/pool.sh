#!/usr/bin/env bash
# search/proposer/pool.sh -- the N-PROPOSER POOL loop (one of the two always-on loops run.sh
# supervises). Launches NEURO_FLEET headless proposers (search/proposer/worker.sh p00..p<N-1>), one per
# sandbox, and keeps the pool full: a worker that dies is restarted until the wall-clock deadline.
# It owns only pool size + liveness, NOT the stop-condition (run.sh signals this whole group).
#
# Slots rotate across provider:model[:effort] entries, default Codex/Astra and agy/Argon 4.
# The optional third field overrides NEURO_EFFORT for that slot.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/search/lib.sh"
use_project_python || exit 2
WORKER="$REPO/search/proposer/worker.sh"
FLEET="${NEURO_FLEET:-10}"
DEADLINE="${NEURO_DEADLINE:-0}"
[ -f "$WORKER" ] || { echo "[propose] missing $WORKER" >&2; exit 1; }
: "${LAB_RUN:?LAB_RUN must be set}"

EFFORT="${NEURO_EFFORT:-high}"              # the ONE pool-wide effort knob (run.sh --effort)
read -r -a ROSTER <<< "${NEURO_MODEL_ROSTER:-codex:astra agy:argon4}"
[ "${#ROSTER[@]}" -gt 0 ] || { echo "[propose] empty model roster" >&2; exit 2; }
python "$REPO/search/providers.py" preflight --agents "${ROSTER[@]}" --effort "$EFFORT" >/dev/null || exit 2
NROSTER=${#ROSTER[@]}
echo "[propose] pool of $FLEET (run=$LAB_RUN dataset=${NEURO_DATASET:-laion}) base-effort=$EFFORT -- roster (round-robin): ${ROSTER[*]}"

declare -a PIDS
start_slot() {
  local i="$1" entry
  entry="${ROSTER[$((i % NROSTER))]}"   # round-robin assignment
  NEURO_PROPOSER_MODEL="$entry" NEURO_PROPOSER_EFFORT="$EFFORT" bash "$WORKER" "$(printf 'p%02d' "$i")" &
  PIDS[$i]=$!
  echo "[propose] $(printf 'p%02d' "$i") up (pid ${PIDS[$i]}, $entry/$EFFORT)"
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
