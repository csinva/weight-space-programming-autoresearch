#!/usr/bin/env bash
# Launch the open-ended search CONDUCTOR (a persistent Copilot session) and keep it going
# until the time budget. Each invocation = ONE round (the conductor does one round then
# stops); we relaunch with --continue so it keeps a thin cross-round context. All durable
# state is on disk (results.db / notebook.md / strategy.md), so a killed run resumes losslessly.
#
#   search/search.sh [hours]      # default 8h
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
HOURS="${1:-8}"

# one-time: ensure the db exists (migrate the CSV history if present and db is missing)
if [ ! -f search/state/results.db ]; then
  if [ -f search/state/results.csv ]; then python search/db.py migrate search/state/results.csv; else python search/db.py init; fi
fi

PROMPT="$(cat search/conductor.md)"
CONTINUE="Do the next round, following your standing instructions. Keep going; never stop on your own."

deadline=$(( $(date +%s) + ${HOURS%.*} * 3600 ))
first=1
while [ "$(date +%s)" -lt "$deadline" ]; do
  remaining=$(( deadline - $(date +%s) ))
  [ "$remaining" -lt 120 ] && break
  echo "[search] launching conductor ($(( remaining / 60 )) min left)"
  if [ "$first" = 1 ]; then
    timeout "$remaining" copilot -p "$PROMPT" --allow-all --model claude-opus-4.8 --add-dir "$REPO" || true
    first=0
  else
    timeout "$remaining" copilot --continue -p "$CONTINUE" --allow-all --model claude-opus-4.8 --add-dir "$REPO" || true
  fi
done
echo "[search] budget reached; done."
