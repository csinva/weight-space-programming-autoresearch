#!/usr/bin/env bash
# Launch an autoresearch loop. ALWAYS uses Copilot CLI with the Opus model; the ONLY
# configurable agent knob is the reasoning effort (default: medium).
#
# Reasoning effort: pass as $1 or via the EFFORT env var (none|low|medium|high|xhigh|max).
# Choose the experiment via env vars before running (the harness reads them):
#   NEURO_DATASET  which preset      (text | nsd | meg; default text)
#   NEURO_ROI      reward region     (NSD: V1/hV4/FFA/PPA/EBA/VWFA/early/ventral)
#   NEURO_RUN      unique run id      (each run gets its own results CSV)
#
# Example:
#   NEURO_DATASET=nsd NEURO_ROI=FFA NEURO_RUN=$(date +%s) scripts/autoresearch.sh
#   NEURO_DATASET=nsd NEURO_ROI=PPA scripts/autoresearch.sh high   # higher effort
set -euo pipefail

MODEL="claude-opus-4.8"                 # FIXED: always Opus -- not configurable.
EFFORT="${1:-${EFFORT:-medium}}"        # the one knob; defaults to medium.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "autoresearch: model=$MODEL effort=$EFFORT" \
     "dataset=${NEURO_DATASET:-text} roi=${NEURO_ROI:-<preset>} run=${NEURO_RUN:-<shared>}" >&2

exec copilot -p "Follow prompts/program.md" \
  --model "$MODEL" --effort "$EFFORT" \
  --autopilot --allow-all -C "$REPO"
