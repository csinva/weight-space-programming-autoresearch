#!/usr/bin/env bash
# search/conductor/loop.sh -- the CONDUCTOR loop (optional 3rd always-on loop run.sh supervises).
# Every NEURO_EPOCH_MINS minutes it runs ONE headless agent pass (search/conductor/prompt.md) that reads
# the shared search.db and writes a BROAD, high-level standing direction per proposer slot into that
# slot's sandbox file `proposers/pNN/direction.txt` (one short line, via `echo ... > direction.txt`).
# ADVISORY only: each proposer reads its own direction.txt to bias what it explores; nothing in the
# queue or scoring gates on it. Runs once at startup to seed directions immediately, then sleeps.
#   LAB_RUN=runs/laion__2026-10-21 NEURO_DATASET=laion NEURO_EPOCH_MINS=30 search/conductor/loop.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/search/lib.sh"
use_project_python || exit 2
cd "$REPO"
RUN="${LAB_RUN:?LAB_RUN must point at the run dir}"
DATASET="${NEURO_DATASET:-laion}"
MODEL="${NEURO_CONDUCTOR_MODEL:-codex:astra}"
EFFORT="${NEURO_CONDUCTOR_EFFORT:-${NEURO_EFFORT:-high}}"
MINS="${NEURO_EPOCH_MINS:-60}"
case "$MINS" in ''|*[!0-9.]*|*.*.*) echo "[conduct] NEURO_EPOCH_MINS must be a positive number (got '$MINS')" >&2; exit 2;; esac
LOG="$RUN/conductor.log"
export LAB_RUN="$RUN" NEURO_DATASET="$DATASET"
echo "[conduct] conductor up (epoch ${MINS}m, $MODEL/$EFFORT, dataset=$DATASET)"
secs="$(awk -v m="$MINS" 'BEGIN{printf "%d", m*60}')"
while :; do
  echo "[conduct] epoch pass $(TZ=America/New_York date +%FT%T%Z)" >> "$LOG"
  # Keep writes inside this run. The fixed hub is invoked by its absolute path.
  PROMPT="$(cat "$REPO/search/conductor/prompt.md")"
  HUB_COMMAND="python \"$REPO/hub.py\""
  PROMPT="${PROMPT//python hub.py/$HUB_COMMAND}"
  python "$REPO/search/providers.py" run --agent "$MODEL" --effort "$EFFORT" \
    --cwd "$RUN" --write-dir "$RUN" <<< "$PROMPT" >> "$LOG" 2>&1 || true
  sleep "$secs"
done
