#!/usr/bin/env bash
# search/search.sh -- LAUNCHER for the always-on open-ended search.
#
# Scaffolds one dated run and supervises the two decentralized loops against it:
#   - the N-proposer pool  (search/propose.sh)   -- 10 Copilot CLIs invent models, queue them
#   - the single evaluator (search/evaluate.sh)  -- scores queued models one at a time on the GPU
#
# There is NO conductor and NO wave barrier: the run's SQLite DB
# (runs/<dataset>__<date>/search.db) is the ONLY shared memory. Each proposer is self-paced --
# it waits only for its OWN model's score, then proposes again. The launcher owns just three
# things: scaffolding, the wall-clock STOP CONDITION (--hours), and graceful shutdown that leaves
# the DB consistent (orphaned `running` rows reset to `queued`).
#
#   search/search.sh <dataset> [imsize] [--hours H] [--fleet N] [--date YYYY-MM-DD]
#   e.g.  search/search.sh nsd 128 --hours 12          # NSD, 128px, 12h, 10 proposers
#         search/search.sh tvsd 128 --hours 6 --fleet 6
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || { echo "[search] cannot cd to repo root" >&2; exit 1; }

DATASET=""; IMSIZE=""; HOURS=""; FLEET=""; DATE=""; pos=0
while [ $# -gt 0 ]; do
  case "$1" in
    --hours)   HOURS="${2:?--hours needs a value}"; shift 2;;
    --hours=*) HOURS="${1#*=}"; shift;;
    --fleet)   FLEET="${2:?--fleet needs a value}"; shift 2;;
    --fleet=*) FLEET="${1#*=}"; shift;;
    --date)    DATE="${2:?--date needs a value}"; shift 2;;
    --date=*)  DATE="${1#*=}"; shift;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    --*)       echo "[search] unknown flag: $1" >&2; exit 2;;
    *) case "$pos" in 0) DATASET="$1";; 1) IMSIZE="$1";; *) echo "[search] unexpected arg: $1" >&2; exit 2;; esac; pos=$((pos+1)); shift;;
  esac
done
DATASET="${DATASET:-nsd}"; IMSIZE="${IMSIZE:-128}"; HOURS="${HOURS:-12}"; FLEET="${FLEET:-10}"
DATE="${DATE:-$(date -u +%Y-%m-%d)}"

case "$HOURS" in ''|*[!0-9.]*|*.*.*) echo "[search] --hours must be a positive number (got '$HOURS')" >&2; exit 2;; esac
awk -v h="$HOURS" 'BEGIN { exit !(h + 0 > 0) }' || { echo "[search] --hours must be > 0" >&2; exit 2; }

# Scaffold (or idempotently complete) runs/<dataset>__<date>/ : search.db + models/ + N proposer
# sandboxes + eval/. Re-launching the same day RESUMES that run losslessly.
RUN="$REPO/runs/${DATASET}__${DATE}"
python scripts/setup_run.py "$DATASET" --date "$DATE" --n "$FLEET" >/dev/null
KEY="${DATASET}__${DATE}"
export LAB_RUN="$RUN" NEURO_DATASET="$DATASET" NEURO_IMSIZE="$IMSIZE" NEURO_FLEET="$FLEET"

# --- DB consistency: reset any orphaned `running` row back to `queued` (startup + shutdown) -----
reconcile() {
  python - "$RUN" <<'PY'
import sys; sys.path.insert(0, ".")
import lab
n = lab.requeue_orphans(sys.argv[1])
if n: print(f"[search] reconciled {n} orphaned running -> queued")
PY
}
summary() {
  python - "$RUN" <<'PY'
import sys; sys.path.insert(0, ".")
import lab
con = lab.connect(sys.argv[1])
rows = dict(con.execute("SELECT COALESCE(status,'?'), COUNT(*) FROM models GROUP BY 1").fetchall())
best = con.execute("SELECT id, test_corr FROM models WHERE status='ok' AND test_corr IS NOT NULL "
                   "ORDER BY test_corr DESC LIMIT 1").fetchone()
con.close()
tally = " ".join(f"{k}={v}" for k, v in sorted(rows.items())) or "(empty)"
champ = f"{best['id']} @ {best['test_corr']:.4f}" if best else "(none)"
print(f"[search] DB: {tally} | champion: {champ}")
PY
}
reconcile

PROPOSE="$REPO/search/propose.sh"; EVALUATE="$REPO/search/evaluate.sh"
for s in "$PROPOSE" "$EVALUATE"; do [ -f "$s" ] || { echo "[search] missing loop script: $s" >&2; exit 1; }; done

now=$(date +%s)
deadline=$(awk -v n="$now" -v h="$HOURS" 'BEGIN { printf "%d", n + h * 3600 }')
export NEURO_DEADLINE="$deadline"

set -m
PROP_PID=0; EVAL_PID=0
alive() { kill -0 "$1" 2>/dev/null; }
start_propose()  { bash "$PROPOSE"  & PROP_PID=$!; echo "[search] proposer pool up (pid $PROP_PID, fleet $FLEET)"; }
start_evaluate() { bash "$EVALUATE" & EVAL_PID=$!; echo "[search] evaluator up (pid $EVAL_PID)"; }
term_group()     { [ "$1" -gt 0 ] && kill -TERM -"$1" 2>/dev/null; return 0; }
kill_group()     { [ "$1" -gt 0 ] && kill -KILL -"$1" 2>/dev/null; return 0; }

shutdown() {
  trap '' INT TERM
  echo "[search] $KEY -- stopping loops..."
  term_group "$PROP_PID"; term_group "$EVAL_PID"
  for _ in $(seq 1 "${GRACE:-20}"); do alive "$PROP_PID" || alive "$EVAL_PID" || break; sleep 1; done
  kill_group "$PROP_PID"; kill_group "$EVAL_PID"
  wait "$PROP_PID" "$EVAL_PID" 2>/dev/null || true
  reconcile; summary
  echo "[search] $KEY -- done."
}
trap 'exit 130' INT TERM
trap shutdown EXIT

echo "[search] $KEY -- always-on search (dataset=$DATASET imsize=$IMSIZE fleet=$FLEET) for ${HOURS}h"
echo "[search] run dir: $RUN"
start_propose
start_evaluate

CHECK=5
while :; do
  now=$(date +%s); [ "$now" -ge "$deadline" ] && break
  alive "$PROP_PID" || { kill_group "$PROP_PID"; wait "$PROP_PID" 2>/dev/null || true; echo "[search] proposer pool exited; restarting" >&2; start_propose; }
  alive "$EVAL_PID" || { kill_group "$EVAL_PID"; wait "$EVAL_PID" 2>/dev/null || true; echo "[search] evaluator exited; restarting" >&2; start_evaluate; }
  rem=$(( deadline - now )); s=$CHECK; [ "$rem" -lt "$s" ] && s="$rem"; [ "$s" -lt 1 ] && s=1
  sleep "$s"
done
echo "[search] $KEY -- ${HOURS}h budget reached."
exit 0
