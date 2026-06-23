#!/usr/bin/env bash
# search/proposer.sh -- launch ONE persistent headless Copilot proposer for the open-ended
# search. The proposer is a member of a fixed pool that stays ALIVE for the whole run (never
# killed or restarted between waves): it loops forever -- barrier-wait, query the shared DB,
# invent ONE untrained model, write it to disk, enqueue one queued row -- and the wave number
# self-increments through DB inserts alone. The model and reasoning effort are configurable
# PER worker, so a pool can mix strong and cheap proposers. The launcher (chunk 03) starts N of
# these alongside one search/evaluator.py; this script is a single worker and does not stop on
# its own.
#
#   search/proposer.sh <dataset> <roi> [imsize] [model] [effort]
#   e.g.  search/proposer.sh nsd ventral 128 claude-opus-4.8 high
#         search/proposer.sh tvsd IT 128 claude-sonnet-4.6 medium
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DATASET="${1:-nsd}"; ROI="${2:-ventral}"; IMSIZE="${3:-128}"
MODEL="${4:-claude-opus-4.8}"; EFFORT="${5:-high}"

# dataset -> study name + the per-dataset framing (the bar/ceiling + any output-dim cap),
# mirroring harness/datasets.py presets and search/lib/prompts.mjs targetSpec.
case "$DATASET" in
  nsd)  STUDY=NsdSubj01;   LABEL="NSD subject-1 ${ROI}-region fMRI";
        CEIL="a trained ImageNet AlexNet (fc7) read out the same way is the per-region bar to BEAT (e.g. ventral ~0.284, FFA ~0.396, PPA ~0.403); the noise ceiling is ~0.50"; CAP="";;
  tvsd) STUDY=TvsdMonkeyF; LABEL="TVSD macaque ${ROI} spiking (monkey F)";
        CEIL="the release's split-half noise ceiling averages ~0.55-0.63 (no trained-AlexNet bar for tvsd) -- there is real headroom"; CAP="30000";;
  nnn)  STUDY=NnnV1;       LABEL="Triple-N macaque ${ROI} spiking";
        CEIL="an untrained hand-designed model already reaches ~0.23 on IT; the noise ceiling is not published"; CAP="";;
  *)    STUDY="$DATASET";  LABEL="${DATASET} ${ROI}"; CEIL=""; CAP="";;
esac

KEY="${STUDY}__${ROI}"
STATE="search/state/${KEY}"
DB="${STATE}/results.db"
MODELS="${STATE}/models"
mkdir -p "$MODELS"
python search/db.py init "$DB"   # ensure the namespace schema (idempotent; cold start of a dataset)

SCHEMA="$(python search/dbaccess.py schema)"
CAPNOTE=""
[ -n "$CAP" ] && CAPNOTE="DATASET LIMIT: keep the model's OUTPUT dimension D_out under ~${CAP} -- the harness ridge readout's GPU SVD fails above that at this dataset's row count, so a larger D_out is scored as an error, not a result. "

PREAMBLE="THIS RUN -- dataset=${DATASET} roi=${ROI} imsize=${IMSIZE}px. Target = held-out ${LABEL} test_corr. For reference, ${CEIL}. ${CAPNOTE}

YOUR NAMESPACE (one search, shared by the whole pool):
- shared history+queue DB : ${DB}
- write each model.py to  : ${MODELS}/<id>.py   (a STABLE path; the file is never moved -- the DB row points here)
- run all commands from the repo root: ${REPO}
- helpers:
    barrier  : python search/barrier.py wait ${DB}          # blocks; prints ITER=<n> to stamp
    dbaccess : python search/dbaccess.py <cmd> ${DB} ...     # top_n | random_n | by_id | blueprint | list | register | call | newid | enqueue

DB SCHEMA (the shared memory -- exactly the columns you can query):
${SCHEMA}"

PROMPT="${PREAMBLE}

$(cat search/proposer.md)"

# No wall-clock here on purpose: the proposer must stay alive for the full run; the launcher
# (chunk 03) owns the pool's lifetime and the search stop-condition.
exec copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" --add-dir "$REPO" -C "$REPO"
