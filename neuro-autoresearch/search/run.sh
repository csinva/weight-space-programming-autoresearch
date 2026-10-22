#!/usr/bin/env bash
# search/run.sh -- LAUNCHER for the always-on open-ended search.
#
# Scaffolds one dated run and supervises the decentralized loops against it:
#   - the N-proposer pool  (search/proposer/pool.sh)   -- N provider CLIs invent models, queue them
#   - the GPU evaluator    (search/evaluator/pool.sh)  -- scores queued models concurrently on the GPU
#   - the conductor        (search/conductor/loop.sh)  -- OPTIONAL steering overlay: every --epoch-mins a
#                                                   agent pass reads the DB and writes one BROAD
#                                                   standing direction per proposer (advisory only)
#
# The run's SQLite DB (runs/<dataset>__<date>/search.db) is the ONLY shared memory. Each proposer is
# self-paced -- it waits only for its OWN model's score, then proposes again; the conductor never
# gates the pool (proposers just read their direction and bias toward it). The launcher owns:
# scaffolding, the wall-clock STOP CONDITION (--hours), and graceful shutdown that leaves the DB
# consistent (orphaned `running` rows reset to `queued`).
#
#   search/run.sh <laion|triplen> [--preflight] [--hours H] [--fleet N] [--epoch-mins M] [--effort E] [--gpu G] [--date YYYY-MM-DD]
#   e.g.  search/run.sh laion --hours 12                 # LAION-fMRI, 12h, 10 proposers, steer every 1h
#         search/run.sh laion --hours 8 --epoch-mins 15  # steer every 15 minutes
#         search/run.sh laion --hours 12 --effort high   # base effort; per-role/spec overrides take precedence
#         search/run.sh triplen --hours 6 --gpu 1 --epoch-mins 0     # pin the run to GPU 1 (one dataset per card), conductor OFF
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/search/lib.sh"
use_project_python
cd "$REPO" || { echo "[search] cannot cd to repo root" >&2; exit 1; }

# Operator dials (evaluator count, agent models/effort, timeouts) live in ONE file; values there are
# DEFAULTS, so a pre-set env var or a CLI flag below still wins. Exported here -> inherited by every
# child loop (propose/evaluate/conduct).
[ -f "$REPO/search/config.env" ] && . "$REPO/search/config.env"

# Generic flag parser. Keep scalar option variables so this works with macOS's Bash 3.2.
# The first bare argument is the dataset; precedence is flag > environment > default.
DATASET=""; HOURS_OPT=""; FLEET_OPT=""; DATE_OPT=""; EPOCH_OPT=""; EFFORT_OPT=""; GPU_OPT=""; PREFLIGHT_ONLY=0; pos=0
while [ $# -gt 0 ]; do
  case "$1" in
    --preflight|--dry-run) PREFLIGHT_ONLY=1; shift; continue;;
    -h|--help) sed -n '2,21p' "$0"; exit 0;;
    --*=*) k="${1#--}"; k="${k%%=*}"; v="${1#*=}"; shift;;
    --*)   k="${1#--}"; v="${2:?$1 needs a value}"; shift 2;;
    *)     [ "$pos" = 0 ] && { DATASET="$1"; pos=1; shift; continue; }; echo "[search] unexpected arg: $1" >&2; exit 2;;
  esac
  case "$k" in
    hours) HOURS_OPT="$v";; fleet) FLEET_OPT="$v";; date) DATE_OPT="$v";;
    epoch-mins) EPOCH_OPT="$v";; effort) EFFORT_OPT="$v";; gpu) GPU_OPT="$v";;
    *) echo "[search] unknown flag: --$k" >&2; exit 2;;
  esac
done
DATASET="${DATASET:-laion}"; HOURS="${HOURS_OPT:-12}"; FLEET="${FLEET_OPT:-10}"; EPOCH="${EPOCH_OPT:-60}"
# --effort sets the base effort (NEURO_EFFORT); role/spec overrides take precedence. --gpu pins the run
# to one CUDA device (multi-GPU: a dataset per card); empty = current default GPU. Both: flag > env > default.
EFFORT="${EFFORT_OPT:-${NEURO_EFFORT:-high}}"; GPU="${GPU_OPT:-${CUDA_VISIBLE_DEVICES:-}}"
DATE="${DATE_OPT:-$(TZ=America/New_York date +%Y-%m-%d)}"   # run rolls over at ET midnight

case "$FLEET" in ''|*[!0-9]*|0) echo "[search] --fleet must be a positive integer" >&2; exit 2;; esac
case "$HOURS" in ''|*[!0-9.]*|*.*.*) echo "[search] --hours must be a positive number (got '$HOURS')" >&2; exit 2;; esac
awk -v h="$HOURS" 'BEGIN { exit !(h + 0 > 0) }' || { echo "[search] --hours must be > 0" >&2; exit 2; }
case "$EPOCH" in ''|*[!0-9.]*|*.*.*) echo "[search] --epoch-mins must be a number >= 0 (got '$EPOCH'; 0 disables the conductor)" >&2; exit 2;; esac
case "$EFFORT" in low|medium|high|xhigh|max) ;; *) echo "[search] --effort must be one of: low medium high xhigh max (got '$EFFORT')" >&2; exit 2;; esac
case "$GPU" in *[!0-9,]*) echo "[search] --gpu must be a GPU index or comma list (got '$GPU')" >&2; exit 2;; esac
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"   # torch in the evaluator reads this; no harness code changes
# VANILLA_AUTORESEARCH=1 : vanilla baseline -- closes every cross-island channel (peek-island +
# random-models in hub.py) and forces the conductor OFF, so each proposer is an independent loop.
# Default 0 = the full system (cross-island pollination + conductor).
[ "${VANILLA_AUTORESEARCH:-0}" = 1 ] && { EPOCH=0; export VANILLA_AUTORESEARCH; echo "[search] VANILLA_AUTORESEARCH: cross-island OFF, conductor OFF (N independent loops)"; }
# Validate the scientific protocol and every provider before creating a run directory or DB.
DATASET="$("$PY" -c 'import sys;from harness.protocols import validate_search_dataset;print(validate_search_dataset(sys.argv[1]))' "$DATASET")" || exit 2
[ "${NEURO_METRIC:-rsa}" = rsa ] && [ "${NEURO_STAGE:-search}" = search ] || {
  echo "[search] search requires NEURO_METRIC=rsa and NEURO_STAGE=search; NSD/TVSD encoding is final evaluation only" >&2; exit 2;
}
if [ -z "${NEURO_SUBJECT:-}" ]; then
  _subjects="$("$PY" -c 'import sys;from harness.protocols import search_subjects_csv;print(search_subjects_csv(sys.argv[1]))' "$DATASET")" || exit 2
  export NEURO_SUBJECT="$_subjects"
fi
export NEURO_DATASET="$DATASET" NEURO_METRIC=rsa NEURO_STAGE=search NEURO_EFFORT="$EFFORT"
"$PY" -c 'import sys;from datetime import datetime;from search.setup_run import evaluation_environment;datetime.strptime(sys.argv[2], "%Y-%m-%d");evaluation_environment(sys.argv[1])' "$DATASET" "$DATE" || exit 2
"$PY" -m search.preflight "$DATASET" || exit 2
read -r -a PREFLIGHT_AGENTS <<< "$NEURO_MODEL_ROSTER"
[ "${#PREFLIGHT_AGENTS[@]}" -gt 0 ] || { echo "[search] empty model roster" >&2; exit 2; }
"$PY" search/providers.py preflight --agents "${PREFLIGHT_AGENTS[@]}" --effort "$EFFORT" || exit 2
"$PY" search/providers.py preflight --agents "$NEURO_REPAIR_MODEL" --effort "$NEURO_REPAIR_EFFORT" || exit 2
if awk -v m="$EPOCH" 'BEGIN { exit !(m + 0 > 0) }'; then
  "$PY" search/providers.py preflight --agents "$NEURO_CONDUCTOR_MODEL" --effort "${NEURO_CONDUCTOR_EFFORT:-$EFFORT}" || exit 2
fi
if [ "$PREFLIGHT_ONLY" = 1 ]; then
  echo "[search] preflight passed: dataset=$DATASET subjects=$NEURO_SUBJECT metric=rsa final=encoding fleet=$FLEET"
  exit 0
fi

# Scaffold (or idempotently complete) runs/<dataset>__<date>/ : search.db + models/ + N proposer
# sandboxes + eval/. Re-launching the same day RESUMES that run losslessly.
RUN="$REPO/runs/${DATASET}__${DATE}"
"$PY" search/setup_run.py "$DATASET" --date "$DATE" --n "$FLEET" >/dev/null
KEY="${DATASET}__${DATE}"
export LAB_RUN="$RUN" NEURO_DATASET="$DATASET" NEURO_FLEET="$FLEET" NEURO_EPOCH_MINS="$EPOCH" NEURO_EFFORT="$EFFORT"

# --- DB consistency (delegated to the hub CLI; LAB_RUN targets this run) ------------------------
# reconcile: reset any orphaned `running` row back to `queued` (startup + shutdown).
reconcile() { local n; n="$("$PY" hub.py requeue)"; [ "${n:-0}" -gt 0 ] && echo "[search] reconciled $n orphaned running -> queued"; return 0; }
log_summary() { echo "[search] $("$PY" hub.py summary)"; }

# Provider event logs retain usage in each CLI's native schema. No invented USD conversion:
# these sessions may be subscription-funded and the CLIs do not expose a common price ledger.
write_usage_index() {
  "$PY" - "$RUN" <<'PY_USAGE'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
logs = sorted(str(p.relative_to(root)) for p in root.glob("logs/p*.jsonl"))
if (root / "conductor.log").exists():
    logs.append("conductor.log")
(root / "usage.json").write_text(json.dumps({"logs": logs, "usd": None,
    "note": "Native provider events are retained in logs; subscription/API cost is not inferred."}, indent=2))
PY_USAGE
}
reconcile

PROPOSE="$REPO/search/proposer/pool.sh"; EVALUATE="$REPO/search/evaluator/pool.sh"; CONDUCT="$REPO/search/conductor/loop.sh"
for s in "$PROPOSE" "$EVALUATE"; do [ -f "$s" ] || { echo "[search] missing loop script: $s" >&2; exit 1; }; done
# The conductor is OPTIONAL: --epoch-mins 0 (or a missing conductor loop) runs the pure decentralized
# pool with no steering overlay.
USE_CONDUCT=0
awk -v m="$EPOCH" 'BEGIN { exit !(m + 0 > 0) }' && [ -f "$CONDUCT" ] && USE_CONDUCT=1

deadline=$(deadline_in "$HOURS")
export NEURO_DEADLINE="$deadline"

set -m
PROP_PID=0; EVAL_PID=0; CONDUCT_PID=0
start_propose()  { bash "$PROPOSE"  & PROP_PID=$!; echo "[search] proposer pool up (pid $PROP_PID, fleet $FLEET)"; }
start_evaluate() { bash "$EVALUATE" & EVAL_PID=$!; echo "[search] evaluator up (pid $EVAL_PID)"; }
start_conduct()  { bash "$CONDUCT"  & CONDUCT_PID=$!; echo "[search] conductor up (pid $CONDUCT_PID, epoch ${EPOCH}m)"; }

shutdown() {
  trap '' INT TERM
  echo "[search] $KEY -- stopping loops..."
  term_group "$PROP_PID"; term_group "$EVAL_PID"; term_group "$CONDUCT_PID"
  for _ in $(seq 1 "${GRACE:-20}"); do alive "$PROP_PID" || alive "$EVAL_PID" || alive "$CONDUCT_PID" || break; sleep 1; done
  kill_group "$PROP_PID"; kill_group "$EVAL_PID"; kill_group "$CONDUCT_PID"
  wait "$PROP_PID" "$EVAL_PID" "$CONDUCT_PID" 2>/dev/null || true
  reconcile; log_summary; write_usage_index
  echo "[search] $KEY -- done."
}
install_shutdown_trap shutdown

echo "[search] $KEY -- always-on search (dataset=$DATASET fleet=$FLEET${GPU:+ gpu=$GPU}) for ${HOURS}h"
echo "[search] run dir: $RUN"
start_propose
start_evaluate
[ "$USE_CONDUCT" -eq 1 ] && start_conduct || echo "[search] conductor disabled (--epoch-mins $EPOCH); pure decentralized pool"

CHECK=5
while :; do
  now=$(date +%s); [ "$now" -ge "$deadline" ] && break
  if ! alive "$PROP_PID"; then
    status=0
    reap_group "$PROP_PID" || status=$?
    if [ "$status" -eq 78 ]; then
      echo "[search] proposer pool stopped with terminal provider failures; stopping run without retrying" >&2
      exit 78
    fi
    echo "[search] proposer pool exited (status $status); restarting" >&2
    start_propose
  fi
  alive "$EVAL_PID" || { reap_group "$EVAL_PID" || true; echo "[search] evaluator exited; restarting" >&2; start_evaluate; }
  if [ "$USE_CONDUCT" -eq 1 ] && ! alive "$CONDUCT_PID"; then
    status=0
    reap_group "$CONDUCT_PID" || status=$?
    if [ "$status" -eq 78 ]; then
      USE_CONDUCT=0
      CONDUCT_PID=0
      echo "[search] conductor disabled after terminal provider failure; proposer/evaluator loops continue" >&2
    else
      echo "[search] conductor exited (status $status); restarting" >&2
      start_conduct
    fi
  fi
  rem=$(( deadline - now )); s=$CHECK; [ "$rem" -lt "$s" ] && s="$rem"; [ "$s" -lt 1 ] && s=1
  sleep "$s"
done
echo "[search] $KEY -- ${HOURS}h budget reached."
exit 0
