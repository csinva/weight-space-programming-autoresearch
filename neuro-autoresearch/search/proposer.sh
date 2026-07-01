#!/usr/bin/env bash
# search/proposer.sh -- launch ONE persistent headless Copilot proposer inside its own sandbox
# (runs/<dataset>__<date>/proposers/<pNN>/). It loops forever: query the shared search.db, edit
# its OWN model.py, propose it (queued), wait for its OWN score, repeat. The pool/launcher owns
# its lifetime; this is a single worker and does not stop on its own.
#
#   LAB_RUN=runs/nsd__2026-06-24 NEURO_DATASET=nsd search/proposer.sh p03
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
# Per-dataset framing: just the target's identity (dataset + modality + ROI) and any output-dim cap
# (the ridge SVD limit at this dataset). Deliberately NO subject count, voxel count, or metric detail.
case "$DATASET" in
  nsd)  LABEL="the Natural Scenes Dataset (NSD): human fMRI responses to natural images, across the early and ventral visual stream"; CAP="";;
  tvsd) LABEL="TVSD: macaque multi-unit spiking to natural images, across early visual cortex through IT"; CAP="";;
  nnn)  LABEL="Triple-N: macaque spiking to natural images, across early visual cortex and IT"; CAP="";;
  *)    LABEL="$DATASET"; CAP="";;
esac
CAPNOTE=""
[ -n "$CAP" ] && CAPNOTE="DATASET LIMIT: keep the model OUTPUT dim D_out under ~${CAP} -- the ridge readout's GPU SVD fails above that at this dataset's row count (scored as an error). "

PREAMBLE="THIS RUN -- dataset=${DATASET}. You are proposer ${PNAME}, ONE of ${NEURO_FLEET:-several}
proposers running CONCURRENTLY right now -- all of you read from and write models into the SAME
search.db at the same time, with no coordinator. Reward =
a held-out encoding score on ${LABEL} (a fixed ridge readout from your model's activations to the
recorded responses). For reference, ${CEIL}. ${CAPNOTE}The shared brain is search.db in your run; reach it ONLY through \`python hub.py\`
(run from this sandbox)."

PROMPT="${PREAMBLE}

$(cat "$REPO/search/proposer.md")"

export LAB_RUN="$RUN" LAB_PROPOSER="$PNAME" NEURO_DATASET="$DATASET" LAB_SOLO="${LAB_SOLO:-}"
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
