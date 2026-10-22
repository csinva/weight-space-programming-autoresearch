#!/usr/bin/env bash
# search/proposer/pool.sh -- the N-PROPOSER POOL loop (one of the two always-on loops run.sh
# supervises). Launches NEURO_FLEET headless proposers (search/proposer/worker.sh p00..p<N-1>), one per
# sandbox, and restarts workers until the deadline, except terminal provider failures (exit 78).
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

EFFORT="${NEURO_EFFORT:-high}"              # base effort; a roster entry can override it
read -r -a ROSTER <<< "${NEURO_MODEL_ROSTER:-codex:astra agy:argon4}"
[ "${#ROSTER[@]}" -gt 0 ] || { echo "[propose] empty model roster" >&2; exit 2; }
"$PY" "$REPO/search/providers.py" preflight --agents "${ROSTER[@]}" --effort "$EFFORT" >/dev/null || exit 78
NROSTER=${#ROSTER[@]}
echo "[propose] pool of $FLEET (run=$LAB_RUN dataset=${NEURO_DATASET:-laion}) base-effort=$EFFORT -- roster (round-robin): ${ROSTER[*]}"

# Each slot owns a process group, including the provider CLI and its tools.
# A dead worker must not leave tools running alongside its replacement.
set -m
declare -a PIDS DISABLED
ACTIVE="$FLEET"
start_slot() {
  local i="$1" entry
  entry="${ROSTER[$((i % NROSTER))]}"   # round-robin assignment
  NEURO_PROPOSER_MODEL="$entry" NEURO_PROPOSER_EFFORT="$EFFORT" bash "$WORKER" "$(printf 'p%02d' "$i")" &
  PIDS[$i]=$!
  echo "[propose] $(printf 'p%02d' "$i") up (pid ${PIDS[$i]}, $entry/$EFFORT)"
}
shutdown() {
  trap '' INT TERM
  echo "[propose] stopping ${#PIDS[@]} worker groups..."
  for pid in "${PIDS[@]:-}"; do term_group "$pid"; done
  # Finish before the launcher's outer shutdown grace expires, including tools
  # that ignore TERM after their CLI has already exited.
  sleep 0.2
  for pid in "${PIDS[@]:-}"; do kill_group "$pid"; done
  wait 2>/dev/null || true
}
install_shutdown_trap shutdown

for i in $(seq 0 $((FLEET - 1))); do start_slot "$i"; done
while ! past_deadline "$DEADLINE"; do
  for i in $(seq 0 $((FLEET - 1))); do
    [ "${DISABLED[$i]:-0}" = 1 ] && continue
    if ! alive "${PIDS[$i]:-0}"; then
      # The CLI is gone; its remaining tools cannot belong to a new session.
      kill_group "${PIDS[$i]:-0}"
      status=0
      wait "${PIDS[$i]:-0}" 2>/dev/null || status=$?
      if [ "$status" -eq 78 ]; then
        DISABLED[$i]=1
        PIDS[$i]=0
        ACTIVE=$((ACTIVE - 1))
        echo "[propose] $(printf 'p%02d' "$i") disabled: terminal provider failure (exit 78); inspect $LAB_RUN/proposers/$(printf 'p%02d' "$i")/proposer.log; $ACTIVE active slots remain" >&2
        if [ "$ACTIVE" -eq 0 ]; then
          echo "[propose] all proposer slots disabled; fix provider configuration/permissions before relaunching" >&2
          exit 78
        fi
        continue
      fi
      past_deadline "$DEADLINE" && break
      echo "[propose] $(printf 'p%02d' "$i") exited; restarting" >&2
      start_slot "$i"
    fi
  done
  sleep 5
done
echo "[propose] deadline reached; pool draining."
