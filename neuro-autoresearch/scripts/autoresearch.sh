#!/usr/bin/env bash
# Launch an autoresearch loop. ALWAYS uses Copilot CLI with the Opus model; the ONLY
# configurable agent knob is the reasoning effort (default: medium).
#
# The agent is run NONSTOP via `copilot-keepgoing`: the interactive Copilot CLI runs
# inside a pty and is re-nudged to "keep going" whenever it falls idle, so a run fills
# its whole wall-clock budget instead of stopping the moment the agent thinks a turn is
# done. ALWAYS bound a run with `timeout` -- keepgoing never stops on its own:
#   NEURO_DATASET=nsd NEURO_ROI=FFA NEURO_RUN=$(date +%s) timeout 90m scripts/autoresearch.sh high
#
# Reasoning effort: pass as $1 or via the EFFORT env var (none|low|medium|high|xhigh|max).
# Choose the experiment via env vars before running (the harness reads them):
#   NEURO_DATASET  which preset      (text | nsd | meg; default text)
#   NEURO_ROI      reward region     (NSD: V1/hV4/FFA/PPA/EBA/VWFA/early/ventral)
#   NEURO_RUN      unique run id      (each run gets its own results CSV)
# Other knobs:
#   WORKDIR        working dir for the agent (default: repo root). Point this at an
#                  isolated run folder (runs/<tag>) so parallel runs never collide.
#   KEEPGOING_IDLE seconds of inactivity before a nudge is sent (default: 90)
#
# Example:
#   NEURO_DATASET=nsd NEURO_ROI=FFA NEURO_RUN=$(date +%s) timeout 90m scripts/autoresearch.sh
#   NEURO_DATASET=nsd NEURO_ROI=PPA timeout 90m scripts/autoresearch.sh high   # higher effort
set -euo pipefail

MODEL="claude-opus-4.8"                 # FIXED: always Opus -- not configurable.
EFFORT="${1:-${EFFORT:-medium}}"        # the one knob; defaults to medium.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-$REPO}"             # where the agent works (a run folder, ideally).
IDLE="${KEEPGOING_IDLE:-90}"            # seconds idle before keepgoing nudges.

# The kickoff doubles as the recurring nudge: keepgoing's first message starts the
# agent on the task, and every later idle nudge tells it to keep proposing models.
MESSAGE="Follow prompts/program.md: run the neural-encoding autoresearch loop, \
proposing and evaluating new models to maximise test_corr for the reward region. \
Read this run's results CSV first and hill-climb past best_so_far. Keep going -- \
propose and evaluate the next model now; never stop on your own."

echo "autoresearch: model=$MODEL effort=$EFFORT idle=${IDLE}s workdir=$WORKDIR" \
     "dataset=${NEURO_DATASET:-text} roi=${NEURO_ROI:-<preset>} run=${NEURO_RUN:-<shared>}" >&2

exec copilot-keepgoing --idle "$IDLE" --message "$MESSAGE" -- \
  copilot --model "$MODEL" --effort "$EFFORT" \
  --autopilot --allow-all -C "$WORKDIR"
