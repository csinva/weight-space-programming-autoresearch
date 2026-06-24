#!/usr/bin/env bash
# search/conduct.sh -- the CONDUCTOR loop (optional 3rd always-on loop search.sh supervises).
# Every NEURO_EPOCH_MINS minutes it runs ONE headless Copilot pass (search/conductor.md) that reads
# the shared search.db and writes a BROAD standing direction per proposer slot into the `steering`
# table (`lab.py steer`). ADVISORY only: proposers read their direction (`lab.py direction`) to bias
# what they explore; nothing in the queue or scoring gates on it. Runs once at startup to seed
# directions immediately, then sleeps between epochs.
#   LAB_RUN=runs/nsd__2026-06-24 NEURO_DATASET=nsd NEURO_EPOCH_MINS=30 search/conduct.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RUN="${LAB_RUN:?LAB_RUN must point at the run dir}"
DATASET="${NEURO_DATASET:-nsd}"
MODEL="${NEURO_CONDUCTOR_MODEL:-claude-opus-4.8}"
EFFORT="${NEURO_CONDUCTOR_EFFORT:-medium}"
MINS="${NEURO_EPOCH_MINS:-30}"
case "$MINS" in ''|*[!0-9.]*|*.*.*) echo "[conduct] NEURO_EPOCH_MINS must be a positive number (got '$MINS')" >&2; exit 2;; esac
LOG="$RUN/conductor.log"
PROMPT="$(cat "$REPO/search/conductor.md")"
export LAB_RUN="$RUN" NEURO_DATASET="$DATASET"
echo "[conduct] conductor up (epoch ${MINS}m, $MODEL/$EFFORT, dataset=$DATASET)"
secs="$(awk -v m="$MINS" 'BEGIN{printf "%d", m*60}')"
while :; do
  echo "[conduct] epoch pass $(date -u +%FT%TZ)" >> "$LOG"
  copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
    --add-dir "$REPO" -C "$REPO" >> "$LOG" 2>&1 || true
  sleep "$secs"
done
