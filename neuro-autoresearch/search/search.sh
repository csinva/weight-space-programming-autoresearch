#!/usr/bin/env bash
# Launch the open-ended search CONDUCTOR (a persistent Copilot session) for one dataset/roi
# and keep it going until the time budget. Each invocation = ONE round (the conductor does
# one round then stops); we relaunch with --continue so it keeps a thin cross-round context.
# All durable state is on disk under the per-dataset namespace, so a kill resumes losslessly.
#
#   search/search.sh <dataset> <roi> [imsize] [hours]
#   e.g.  search/search.sh tvsd IT 128 8        # TVSD monkey-F IT, 128px, 8h
#         search/search.sh nsd ventral 128 8    # the original NSD-ventral search
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DATASET="${1:-nsd}"; ROI="${2:-ventral}"; IMSIZE="${3:-128}"; HOURS="${4:-8}"

case "$DATASET" in
  nsd) STUDY=NsdSubj01;; tvsd) STUDY=TvsdMonkeyF;; nnn) STUDY=NnnV1;; *) STUDY="$DATASET";;
esac
KEY="${STUDY}__${ROI}"
STATE="search/state/$KEY"
mkdir -p "$STATE/queue" "$STATE/done"
python search/db.py init "$STATE/results.db"   # ensure schema (idempotent; cold start of a new dataset)

PREAMBLE="THIS RUN -- dataset=$DATASET roi=$ROI imsize=$IMSIZE. Namespace = $STATE.
Use these EXACT dataset/roi/imsize in round-input.json. Your files for this run:
- write round-input.json    -> $STATE/round-input.json
- read previous result.json -> $STATE/result.json
- notebook (shared memory)  -> $STATE/notebook.md
- strategy (your plan)      -> $STATE/strategy.md
- full history db           -> $STATE/results.db
Run the round with:  node search/round.mjs --input $STATE/round-input.json"

PROMPT="$PREAMBLE

$(cat search/conductor.md)"
CONTINUE="Do the next round for this run, following your standing instructions and THIS RUN's paths. Keep going; never stop on your own."

deadline=$(( $(date +%s) + ${HOURS%.*} * 3600 ))
first=1
while [ "$(date +%s)" -lt "$deadline" ]; do
  remaining=$(( deadline - $(date +%s) ))
  [ "$remaining" -lt 120 ] && break
  echo "[search] $KEY -- launching conductor ($(( remaining / 60 )) min left)"
  if [ "$first" = 1 ]; then
    timeout "$remaining" copilot -p "$PROMPT" --allow-all --model claude-opus-4.8 --add-dir "$REPO" || true
    first=0
  else
    timeout "$remaining" copilot --continue -p "$CONTINUE" --allow-all --model claude-opus-4.8 --add-dir "$REPO" || true
  fi
done
echo "[search] $KEY -- budget reached; done."
