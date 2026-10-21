#!/usr/bin/env bash
# Launch one headless proposer in its own working directory; the supervisor owns its lifetime.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/search/lib.sh"
use_project_python || exit 2
PNAME="${1:?usage: worker.sh <pNN> (LAB_RUN and NEURO_DATASET in environment)}"
RUN="${LAB_RUN:?LAB_RUN must point at the run dir}"
DATASET="${NEURO_DATASET:-laion}"
MODEL="${NEURO_PROPOSER_MODEL:-codex:astra}"
EFFORT="${NEURO_PROPOSER_EFFORT:-${NEURO_EFFORT:-high}}"
SANDBOX="$RUN/proposers/$PNAME"
[ -d "$SANDBOX" ] || { echo "[proposer] missing workspace: $SANDBOX" >&2; exit 1; }
LOG="$SANDBOX/proposer.log"
case "$DATASET" in
  laion) LABEL="LAION-fMRI: human fMRI responses to natural images, averaged across ALL subjects";;
  triplen) LABEL="Triple-N: macaque spiking responses to natural images";;
  *) echo "[proposer] invalid search dataset: $DATASET" >&2; exit 2;;
esac
PREAMBLE="THIS RUN -- dataset=${DATASET}. You are proposer ${PNAME}, one of ${NEURO_FLEET:-several}
concurrent proposers sharing search.db. Reward = RSA on ${LABEL}: alignment of your model's
representational geometry with neural responses. Maximize that verifiable RSA feedback.
The database field test_corr is a compatibility alias for the RSA search reward, NOT an encoding score.
Final transfer evaluations (LAION-fMRI -> NSD and Triple-N -> TVSD) use encoding scores and are
unavailable during search. Never run, request, or inspect final evaluation data or scores.
Parameter budget: at most ${NEURO_MAX_PARAMS:-100000} fixed weights counted across all tensors.
Access the shared search.db only through python hub.py from your proposer directory."
PROMPT="${PREAMBLE}

$(cat "$REPO/search/proposer/prompt.md")"
export LAB_RUN="$RUN" LAB_PROPOSER="$PNAME" NEURO_DATASET="$DATASET" VANILLA_AUTORESEARCH="${VANILLA_AUTORESEARCH:-0}"
echo "[proposer] $PNAME session start $(TZ=America/New_York date +%FT%T%Z) ($MODEL/$EFFORT)" >> "$LOG"
# The run directory is writable for hub's queue/model snapshots. This is not an OS read-isolation boundary.
exec python "$REPO/search/providers.py" run --agent "$MODEL" --effort "$EFFORT" \
  --cwd "$SANDBOX" --write-dir "$RUN" <<< "$PROMPT" >> "$LOG" 2>&1
