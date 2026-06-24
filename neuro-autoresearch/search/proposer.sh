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
IMSIZE="${NEURO_IMSIZE:-128}"
MODEL="${NEURO_PROPOSER_MODEL:-claude-opus-4.8}"
EFFORT="${NEURO_PROPOSER_EFFORT:-high}"
SANDBOX="$RUN/proposers/$PNAME"
[ -d "$SANDBOX" ] || { echo "[proposer] missing sandbox: $SANDBOX" >&2; exit 1; }

# Per-dataset framing: the reward target + any output-dim cap (the ridge SVD limit at this dataset).
case "$DATASET" in
  nsd)  LABEL="NSD subject-1 ventral visual-stream fMRI";
        CEIL="a trained ImageNet AlexNet (fc7) read out the same way scores ~0.284 on ventral; the noise ceiling is ~0.50. The TARGET for this search is test_corr >= 0.40 on the ventral stream -- well above AlexNet. Untrained models already match AlexNet in early/mid visual cortex; the open frontier is this high-level ventral stream"; CAP="";;
  tvsd) LABEL="TVSD macaque IT spiking (monkey F)";  CEIL="split-half noise ceiling ~0.55-0.63; real headroom"; CAP="30000";;
  nnn)  LABEL="Triple-N macaque IT spiking";         CEIL="an untrained model already reaches ~0.23 on IT"; CAP="";;
  *)    LABEL="$DATASET";                             CEIL=""; CAP="";;
esac
CAPNOTE=""
[ -n "$CAP" ] && CAPNOTE="DATASET LIMIT: keep the model OUTPUT dim D_out under ~${CAP} -- the ridge readout's GPU SVD fails above that at this dataset's row count (scored as an error). "

PREAMBLE="THIS RUN -- dataset=${DATASET} imsize=${IMSIZE}px. You are proposer ${PNAME}. Reward =
held-out ${LABEL} test_corr (mean per-target Pearson r of a fixed ridge readout). For reference,
${CEIL}. ${CAPNOTE}The shared brain is search.db in your run; reach it ONLY through \`python lab.py\`
(run from this sandbox)."

PROMPT="${PREAMBLE}

$(cat "$REPO/search/proposer.md")"

export LAB_RUN="$RUN" LAB_PROPOSER="$PNAME" NEURO_DATASET="$DATASET" NEURO_IMSIZE="$IMSIZE"
cd "$SANDBOX"
# No wall-clock here: the proposer stays alive for the whole run; the launcher owns the stop.
exec copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
     --add-dir "$SANDBOX" --add-dir "$REPO" -C "$SANDBOX"
