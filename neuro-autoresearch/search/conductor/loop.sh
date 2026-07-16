#!/usr/bin/env bash
# search/conductor/loop.sh -- the CONDUCTOR loop (optional 3rd always-on loop run.sh supervises).
# Every NEURO_EPOCH_MINS minutes it runs ONE headless Copilot pass (search/conductor/prompt.md) that reads
# the shared search.db and writes a BROAD, high-level standing direction per proposer slot into that
# slot's sandbox file `proposers/pNN/direction.txt` (one short line, via `echo ... > direction.txt`).
# ADVISORY only: each proposer reads its own direction.txt to bias what it explores; nothing in the
# queue or scoring gates on it. Runs once at startup to seed directions immediately, then sleeps.
#   LAB_RUN=runs/nsd__2026-06-24 NEURO_DATASET=nsd NEURO_EPOCH_MINS=30 search/conductor/loop.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUN="${LAB_RUN:?LAB_RUN must point at the run dir}"
DATASET="${NEURO_DATASET:-nsd}"
MODEL="${NEURO_CONDUCTOR_MODEL:-claude-opus-4.8}"
EFFORT="${NEURO_CONDUCTOR_EFFORT:-${NEURO_EFFORT:-xhigh}}"
MINS="${NEURO_EPOCH_MINS:-60}"
case "$MINS" in ''|*[!0-9.]*|*.*.*) echo "[conduct] NEURO_EPOCH_MINS must be a positive number (got '$MINS')" >&2; exit 2;; esac
LOG="$RUN/conductor.log"
export LAB_RUN="$RUN" NEURO_DATASET="$DATASET"
echo "[conduct] conductor up (epoch ${MINS}m, $MODEL/$EFFORT, dataset=$DATASET)"
secs="$(awk -v m="$MINS" 'BEGIN{printf "%d", m*60}')"
while :; do
  echo "[conduct] epoch pass $(TZ=America/New_York date +%FT%T%Z)" >> "$LOG"
  # Re-read the prompt each epoch so edits go live without a restart.
  copilot -p "$(cat "$REPO/search/conductor/prompt.md")" --allow-all --model "$MODEL" --effort "$EFFORT" \
    --add-dir "$REPO" -C "$REPO" >> "$LOG" 2>&1 || true
  sleep "$secs"
done
