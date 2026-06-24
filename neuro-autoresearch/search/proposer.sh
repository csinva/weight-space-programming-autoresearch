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
EFFORT="${NEURO_PROPOSER_EFFORT:-high}"
SANDBOX="$RUN/proposers/$PNAME"
[ -d "$SANDBOX" ] || { echo "[proposer] missing sandbox: $SANDBOX" >&2; exit 1; }
LOG="$SANDBOX/proposer.log"   # this worker's OWN log -> per-proposer AI-credit attribution at run end

# Per-dataset framing: the reward target + any output-dim cap (the ridge SVD limit at this dataset).
case "$DATASET" in
  nsd)  LABEL="NSD subject-1 ventral visual-stream fMRI (prepared GLMdenoise betas)";
        CEIL="the reward is the FINE-GRAINED ventral stream of NSD's prepared GLMdenoise betas (the standard NSD encoding target, ~8.6k voxels). The random-projection baseline floor is ~0.08; the TARGET for this search is test_corr >= 0.40. A strong UNTRAINED computational model of the high-level ventral representation is the open frontier"; CAP="";;
  tvsd) LABEL="TVSD macaque IT spiking (monkey F)";  CEIL="split-half noise ceiling ~0.55-0.63; real headroom"; CAP="";;
  nnn)  LABEL="Triple-N macaque IT spiking";         CEIL="an untrained model already reaches ~0.23 on IT"; CAP="";;
  *)    LABEL="$DATASET";                             CEIL=""; CAP="";;
esac
CAPNOTE=""
[ -n "$CAP" ] && CAPNOTE="DATASET LIMIT: keep the model OUTPUT dim D_out under ~${CAP} -- the ridge readout's GPU SVD fails above that at this dataset's row count (scored as an error). "

PREAMBLE="THIS RUN -- dataset=${DATASET}. You are proposer ${PNAME}. Reward =
held-out ${LABEL} test_corr (mean per-target Pearson r of a fixed ridge readout). For reference,
${CEIL}. ${CAPNOTE}The shared brain is search.db in your run; reach it ONLY through \`python lab.py\`
(run from this sandbox)."

PROMPT="${PREAMBLE}

$(cat "$REPO/search/proposer.md")"

export LAB_RUN="$RUN" LAB_PROPOSER="$PNAME" NEURO_DATASET="$DATASET"
cd "$SANDBOX"
# No wall-clock here: the proposer stays alive for the whole run; the launcher owns the stop.
# Confined to its OWN sandbox: --add-dir is ONLY the sandbox, so the agent cannot browse the repo
# (docs/, harness source, other proposers). It still RUNS `python lab.py` / `import harness` via the
# symlinks (OS resolves those for subprocesses regardless of the agent's path allow-list).
# Output is APPENDED to this proposer's OWN log (kept across restarts) so the launcher can sum this
# session's "AI Credits" at shutdown. `exec` is preserved so a TERM still reaches copilot directly
# and it flushes its final credit summary.
echo "[proposer] $PNAME session start $(date -u +%FT%TZ) ($MODEL/$EFFORT)" >> "$LOG"
exec copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
     --add-dir "$SANDBOX" -C "$SANDBOX" >> "$LOG" 2>&1
