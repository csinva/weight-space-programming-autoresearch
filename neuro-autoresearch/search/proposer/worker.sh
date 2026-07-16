#!/usr/bin/env bash
# search/proposer/worker.sh -- launch ONE persistent headless Copilot proposer inside its own sandbox
# (runs/<dataset>__<date>/proposers/<pNN>/). It loops forever: query the shared search.db, edit
# its OWN model.py, propose it (queued), wait for its OWN score, repeat. The pool/launcher owns
# its lifetime; this is a single worker and does not stop on its own.
#
#   LAB_RUN=runs/nsd__2026-06-24 NEURO_DATASET=nsd search/proposer/worker.sh p03
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PNAME="${1:?usage: proposer.sh <pNN> (LAB_RUN + NEURO_DATASET in env)}"
RUN="${LAB_RUN:?LAB_RUN must point at the run dir}"
DATASET="${NEURO_DATASET:-nsd}"
MODEL="${NEURO_PROPOSER_MODEL:-claude-opus-4.8}"
EFFORT="${NEURO_PROPOSER_EFFORT:-${NEURO_EFFORT:-xhigh}}"
SANDBOX="$RUN/proposers/$PNAME"
[ -d "$SANDBOX" ] || { echo "[proposer] missing sandbox: $SANDBOX" >&2; exit 1; }
LOG="$SANDBOX/proposer.log"   # this worker's OWN log -> per-proposer AI-credit attribution at run end

# Shared aspiration (same scientific goal for every dataset; carries NO eval mechanics so the
# proposer can't game the ruler -- only the target's identity is dataset-specific, below).
CEIL="it should be possible for an untrained model to explain a large fraction of the explainable variance in neural predictivity -- push that as far as you can"
# Per-dataset framing: just the target's identity (dataset + modality + ROI).
# Deliberately NO subject count, voxel count, or metric detail.
case "$DATASET" in
  nsd)  LABEL="the Natural Scenes Dataset (NSD): human fMRI responses to natural images, across the early and ventral visual stream";;
  tvsd) LABEL="TVSD: macaque multi-unit spiking to natural images, across early visual cortex through IT";;
  nnn)  LABEL="Triple-N: macaque spiking to natural images, across early visual cortex and IT";;
  ieeg) LABEL="the 'Podcast' ECoG dataset: human intracranial (iEEG) high-gamma responses to naturalistic spoken language, across one patient's cortical electrodes";;
  huth) LABEL="the Huth natural-language fMRI dataset: human fMRI (BOLD) responses to naturalistic spoken narrative stories, across one subject's cortical voxels";;
  *)    LABEL="$DATASET";;
esac

PREAMBLE="THIS RUN -- dataset=${DATASET}. You are proposer ${PNAME}, ONE of ${NEURO_FLEET:-several}
proposers running CONCURRENTLY right now -- all of you read from and write models into the SAME
search.db at the same time, with no coordinator. Reward =
a held-out encoding score on ${LABEL} (a fixed ridge readout from your model's activations to the
recorded responses). For reference, ${CEIL}. Parameter budget: your model may materialise at most
${NEURO_MAX_PARAMS:-100000} fixed weights (counted robustly across ALL tensors you build; over it
errors at score time). The shared brain is search.db in your run; reach it ONLY through \`python hub.py\`
(run from this sandbox)."

PROMPT="${PREAMBLE}

$(cat "$REPO/search/proposer/prompt.md")"

export LAB_RUN="$RUN" LAB_PROPOSER="$PNAME" NEURO_DATASET="$DATASET" VANILLA_AUTORESEARCH="${VANILLA_AUTORESEARCH:-0}"
cd "$SANDBOX"
# No wall-clock here: the proposer stays alive for the whole run; the launcher owns the stop.
# Confined to its OWN sandbox: --add-dir is ONLY the sandbox, so the agent cannot browse the repo
# (docs/, harness source, other proposers). It still RUNS `python hub.py` / `import harness` via the
# symlinks (OS resolves those for subprocesses regardless of the agent's path allow-list).
# Output is APPENDED to this proposer's OWN log (kept across restarts) so the launcher can sum this
# session's "AI Credits" at shutdown. `exec` is preserved so a TERM still reaches copilot directly
# and it flushes its final credit summary.
echo "[proposer] $PNAME session start $(TZ=America/New_York date +%FT%T%Z) ($MODEL/$EFFORT)" >> "$LOG"
exec copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
     --add-dir "$SANDBOX" -C "$SANDBOX" >> "$LOG" 2>&1
