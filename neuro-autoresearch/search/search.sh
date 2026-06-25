#!/usr/bin/env bash
# search/search.sh -- LAUNCHER for the always-on open-ended search.
#
# Scaffolds one dated run and supervises the decentralized loops against it:
#   - the N-proposer pool  (search/propose.sh)   -- N Copilot CLIs invent models, queue them
#   - the GPU evaluator    (search/evaluate.sh)  -- scores queued models concurrently on the GPU
#   - the conductor        (search/conduct.sh)   -- OPTIONAL steering overlay: every --epoch-mins a
#                                                   Copilot pass reads the DB and writes one BROAD
#                                                   standing direction per proposer (advisory only)
#
# The run's SQLite DB (runs/<dataset>__<date>/search.db) is the ONLY shared memory. Each proposer is
# self-paced -- it waits only for its OWN model's score, then proposes again; the conductor never
# gates the pool (proposers just read their direction and bias toward it). The launcher owns:
# scaffolding, the wall-clock STOP CONDITION (--hours), and graceful shutdown that leaves the DB
# consistent (orphaned `running` rows reset to `queued`).
#
#   search/search.sh <dataset> [--hours H] [--fleet N] [--epoch-mins M] [--effort E] [--date YYYY-MM-DD]
#   e.g.  search/search.sh nsd --hours 12                 # NSD, 12h, 10 proposers, steer every 30m
#         search/search.sh nsd --hours 8 --epoch-mins 15  # steer every 15 minutes
#         search/search.sh nsd --hours 12 --effort high   # ONE knob: thinking effort for ALL proposers + the conductor
#         search/search.sh tvsd --hours 6 --fleet 6 --epoch-mins 0   # conductor OFF (decentralized)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || { echo "[search] cannot cd to repo root" >&2; exit 1; }
source "$REPO/search/lib.sh"

DATASET=""; HOURS=""; FLEET=""; DATE=""; EPOCH=""; EFFORT=""; pos=0
while [ $# -gt 0 ]; do
  case "$1" in
    --hours)   HOURS="${2:?--hours needs a value}"; shift 2;;
    --hours=*) HOURS="${1#*=}"; shift;;
    --fleet)   FLEET="${2:?--fleet needs a value}"; shift 2;;
    --fleet=*) FLEET="${1#*=}"; shift;;
    --date)    DATE="${2:?--date needs a value}"; shift 2;;
    --date=*)  DATE="${1#*=}"; shift;;
    --epoch-mins)   EPOCH="${2:?--epoch-mins needs a value}"; shift 2;;
    --epoch-mins=*) EPOCH="${1#*=}"; shift;;
    --effort)   EFFORT="${2:?--effort needs a value}"; shift 2;;
    --effort=*) EFFORT="${1#*=}"; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    --*)       echo "[search] unknown flag: $1" >&2; exit 2;;
    *) case "$pos" in 0) DATASET="$1";; *) echo "[search] unexpected arg: $1" >&2; exit 2;; esac; pos=$((pos+1)); shift;;
  esac
done
DATASET="${DATASET:-nsd}"; HOURS="${HOURS:-12}"; FLEET="${FLEET:-10}"; EPOCH="${EPOCH:-30}"
# The ONE thinking-effort knob: applies to every proposer AND the conductor (children inherit
# NEURO_EFFORT below). Precedence: --effort > $NEURO_EFFORT > default.
EFFORT="${EFFORT:-${NEURO_EFFORT:-xhigh}}"
DATE="${DATE:-$(TZ=America/New_York date +%Y-%m-%d)}"   # run rolls over at ET midnight

case "$HOURS" in ''|*[!0-9.]*|*.*.*) echo "[search] --hours must be a positive number (got '$HOURS')" >&2; exit 2;; esac
awk -v h="$HOURS" 'BEGIN { exit !(h + 0 > 0) }' || { echo "[search] --hours must be > 0" >&2; exit 2; }
case "$EPOCH" in ''|*[!0-9.]*|*.*.*) echo "[search] --epoch-mins must be a number >= 0 (got '$EPOCH'; 0 disables the conductor)" >&2; exit 2;; esac
case "$EFFORT" in low|medium|high|xhigh|max) ;; *) echo "[search] --effort must be one of: low medium high xhigh max (got '$EFFORT')" >&2; exit 2;; esac

# Scaffold (or idempotently complete) runs/<dataset>__<date>/ : search.db + models/ + N proposer
# sandboxes + eval/. Re-launching the same day RESUMES that run losslessly.
RUN="$REPO/runs/${DATASET}__${DATE}"
python scripts/setup_run.py "$DATASET" --date "$DATE" --n "$FLEET" >/dev/null
KEY="${DATASET}__${DATE}"
export LAB_RUN="$RUN" NEURO_DATASET="$DATASET" NEURO_FLEET="$FLEET" NEURO_EPOCH_MINS="$EPOCH" NEURO_EFFORT="$EFFORT"

# --- DB consistency (delegated to the hub CLI; LAB_RUN targets this run) ------------------------
# reconcile: reset any orphaned `running` row back to `queued` (startup + shutdown).
reconcile() { local n; n="$(python hub.py requeue)"; [ "${n:-0}" -gt 0 ] && echo "[search] reconciled $n orphaned running -> queued"; return 0; }
log_summary() { echo "[search] $(python hub.py summary)"; }

# Per-proposer API cost: sum every "AI Credits <n>" line in each proposer's own log (one line per
# copilot session, so restarts accumulate), convert to USD (credits / 100), and write a costs table
# into the run dir. Called once at shutdown, when every proposer has flushed its final credit line.
write_costs() {
  local out="$RUN/costs.txt" total=0 pn log c usd
  { printf '# per-proposer API cost for %s (USD = AI Credits / 100)\n' "$KEY"
    printf '# written %s\n' "$(TZ=America/New_York date +%FT%T%Z)"
    printf '%-8s %12s %12s\n' "proposer" "credits" "usd"; } > "$out"
  for d in "$RUN"/proposers/p*/; do
    [ -d "$d" ] || continue
    pn="$(basename "$d")"; log="$d/proposer.log"
    [ -f "$log" ] || continue
    c="$(grep -oE 'AI Credits[[:space:]]+[0-9.]+' "$log" | grep -oE '[0-9.]+$' \
         | awk '{s+=$1} END{printf "%.1f", s+0}')"
    usd="$(awk -v x="$c" 'BEGIN{printf "%.4f", x/100}')"
    printf '%-8s %12s %12s\n' "$pn" "$c" "$usd" >> "$out"
    total="$(awk -v t="$total" -v x="$c" 'BEGIN{printf "%.1f", t+x}')"
  done
  printf '%-8s %12s %12s\n' "TOTAL" "$total" \
    "$(awk -v x="$total" 'BEGIN{printf "%.4f", x/100}')" >> "$out"
  echo "[search] per-proposer costs -> $out (total ${total} credits = \$$(awk -v x="$total" 'BEGIN{printf "%.2f", x/100}'))"
}
reconcile

PROPOSE="$REPO/search/propose.sh"; EVALUATE="$REPO/search/evaluate.sh"; CONDUCT="$REPO/search/conduct.sh"
for s in "$PROPOSE" "$EVALUATE"; do [ -f "$s" ] || { echo "[search] missing loop script: $s" >&2; exit 1; }; done
# The conductor is OPTIONAL: --epoch-mins 0 (or a missing conduct.sh) runs the pure decentralized
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
  reconcile; log_summary; write_costs
  echo "[search] $KEY -- done."
}
install_shutdown_trap shutdown

echo "[search] $KEY -- always-on search (dataset=$DATASET fleet=$FLEET) for ${HOURS}h"
echo "[search] run dir: $RUN"
start_propose
start_evaluate
[ "$USE_CONDUCT" -eq 1 ] && start_conduct || echo "[search] conductor disabled (--epoch-mins $EPOCH); pure decentralized pool"

CHECK=5
while :; do
  now=$(date +%s); [ "$now" -ge "$deadline" ] && break
  alive "$PROP_PID" || { kill_group "$PROP_PID"; wait "$PROP_PID" 2>/dev/null || true; echo "[search] proposer pool exited; restarting" >&2; start_propose; }
  alive "$EVAL_PID" || { kill_group "$EVAL_PID"; wait "$EVAL_PID" 2>/dev/null || true; echo "[search] evaluator exited; restarting" >&2; start_evaluate; }
  [ "$USE_CONDUCT" -eq 1 ] && { alive "$CONDUCT_PID" || { kill_group "$CONDUCT_PID"; wait "$CONDUCT_PID" 2>/dev/null || true; echo "[search] conductor exited; restarting" >&2; start_conduct; }; }
  rem=$(( deadline - now )); s=$CHECK; [ "$rem" -lt "$s" ] && s="$rem"; [ "$s" -lt 1 ] && s=1
  sleep "$s"
done
echo "[search] $KEY -- ${HOURS}h budget reached."
exit 0
