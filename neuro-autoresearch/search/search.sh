#!/usr/bin/env bash
# search/search.sh -- LAUNCHER for the always-on open-ended search.
#
# Starts and supervises the two decentralized loops against ONE namespace:
#   - the N-proposer pool   (search/propose.sh)   -- invents models, inserts `queued` rows
#   - the single evaluator  (search/evaluate.sh)  -- scores `queued` rows one at a time on the GPU
#
# There is NO conductor and NO notebook: the per-namespace SQLite DB
# (search/state/<study>__<roi>/results.db) is the ONLY shared memory -- each proposer
# self-directs from the ranked DB. The launcher owns just three things: namespace setup, the
# wall-clock STOP CONDITION (--hours), and graceful shutdown that leaves the DB consistent.
#
#   search/search.sh <dataset> <roi> [imsize] [hours] [--hours H] [--fleet N]
#   ([hours] as the 4th positional is back-compat for --hours; if both are given, the later wins)
#   e.g.  search/search.sh tvsd IT 128 --hours 12   # TVSD monkey-F IT, 128px, 12h, 10 proposers
#         search/search.sh nsd ventral              # NSD-ventral, defaults (128px, 12h, fleet 10)
#
# Both loops run until the wall-clock deadline. Killing the launcher (Ctrl-C / SIGTERM) -- or
# either loop dying -- leaves the DB consistent: orphaned `running` rows are reset to `queued`
# on shutdown AND on the next launch, so a kill resumes losslessly.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || { echo "[search] cannot cd to repo root: $REPO" >&2; exit 1; }

# --- args: positional dataset/roi/imsize/hours, plus --hours/--fleet flags (last value wins) -
DATASET=""; ROI=""; IMSIZE=""; HOURS=""; FLEET=""; pos=0
while [ $# -gt 0 ]; do
  case "$1" in
    --hours)   HOURS="${2:?--hours needs a value}"; shift 2;;
    --hours=*) HOURS="${1#*=}"; shift;;
    --fleet)   FLEET="${2:?--fleet needs a value}"; shift 2;;
    --fleet=*) FLEET="${1#*=}"; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    --*)       echo "[search] unknown flag: $1" >&2; exit 2;;
    *) case "$pos" in
         0) DATASET="$1";; 1) ROI="$1";; 2) IMSIZE="$1";; 3) HOURS="$1";;
         *) echo "[search] unexpected arg: $1" >&2; exit 2;;
       esac; pos=$((pos + 1)); shift;;
  esac
done
DATASET="${DATASET:-nsd}"; ROI="${ROI:-ventral}"; IMSIZE="${IMSIZE:-128}"
HOURS="${HOURS:-12}"; FLEET="${FLEET:-10}"

# Validate HOURS: awk would silently coerce a typo (e.g. "12h", "1.2.3") to 0 and a "-5" stays
# negative, yielding deadline<=now -> an instant teardown that exits 0 as if it had run. Reject up
# front so an unattended/budgeted launch never quietly does nothing.
case "$HOURS" in
  ''|*[!0-9.]*|*.*.*) echo "[search] --hours must be a positive number (got: '$HOURS')" >&2; exit 2;;
esac
awk -v h="$HOURS" 'BEGIN { exit !(h + 0 > 0) }' || {
  echo "[search] --hours must be > 0 (got: '$HOURS')" >&2; exit 2; }

case "$DATASET" in
  nsd) STUDY=NsdSubj01;; tvsd) STUDY=TvsdMonkeyF;; nnn) STUDY=NnnV1;; *) STUDY="$DATASET";;
esac
KEY="${STUDY}__${ROI}"
STATE="search/state/$KEY"        # repo-relative namespace (matches evaluate.sh / record.py)
DB="$REPO/$STATE/results.db"     # absolute, for the launcher's own DB calls
mkdir -p "$STATE/queue" "$STATE/done"
python search/db.py init "$DB" >/dev/null   # ensure the namespace schema (idempotent)

# --- DB consistency helpers (reuse db.py's WAL-safe connect, so we never race the loops) ----
# Reset any orphaned `running` row back to `queued`. A row is `running` only while the
# evaluator holds it on the GPU; if that loop is killed mid-score the row would otherwise be
# stuck forever. Called at startup (clean up a previously killed run) and after shutdown.
reconcile() {
  PYTHONPATH="$REPO/search" python - "$DB" <<'PY'
import sys, db
con = db.connect(sys.argv[1])
try:
    n = con.execute("UPDATE models SET status='queued' WHERE status='running'").rowcount
    con.commit()
    if n:
        print(f"[search] reconciled {n} orphaned running -> queued")
except Exception as e:
    print(f"[search] reconcile failed (orphaned 'running' rows may remain): {e}", file=sys.stderr)
finally:
    con.close()
PY
}

# Final status tally + champion -- a quick confirmation the DB is left consistent.
summary() {
  PYTHONPATH="$REPO/search" python - "$DB" <<'PY'
import sys, db
con = db.connect(sys.argv[1])
try:
    rows = dict(con.execute(
        "SELECT COALESCE(status,'?') s, COUNT(*) FROM models GROUP BY s").fetchall())
    best = con.execute(
        "SELECT id, test_corr FROM models WHERE status='ok' AND test_corr IS NOT NULL "
        "ORDER BY test_corr DESC LIMIT 1").fetchone()
finally:
    con.close()
tally = " ".join(f"{k}={v}" for k, v in sorted(rows.items())) or "(empty)"
champ = f"{best['id']} @ {best['test_corr']:.4f}" if best else "(none)"
print(f"[search] DB: {tally} | champion: {champ}")
PY
}

reconcile   # startup: clean up any orphans left by a previously killed run

# --- the two loops the launcher supervises (their bodies live in the runtime, chunk 02) -----
PROPOSE="$REPO/search/propose.sh"   # always-on N-proposer pool
EVALUATE="$REPO/search/evaluate.sh" # always-on single evaluator
for s in "$PROPOSE" "$EVALUATE"; do
  [ -f "$s" ] || { echo "[search] missing loop script: $s (provided by the search runtime)" >&2; exit 1; }
done

# Both loops are configured purely by env -- one namespace, plus the wall-clock deadline they
# MAY honor to avoid starting new work near the end (the launcher enforces the hard stop).
now=$(date +%s)
deadline=$(awk -v n="$now" -v h="$HOURS" 'BEGIN { printf "%d", n + h * 3600 }')
export NEURO_DATASET="$DATASET" NEURO_ROI="$ROI" NEURO_IMSIZE="$IMSIZE" NEURO_STUDY="$STUDY"
export NEURO_STATE="$STATE" NEURO_DB="$DB" NEURO_FLEET="$FLEET" NEURO_DEADLINE="$deadline"

# Job control: each background loop becomes its OWN process-group leader (PGID == its PID), so
# one group signal (`kill -SIGNAL -<pid>`) reaches the whole subtree -- the loop AND the
# copilot children it spawns -- for a clean teardown.
set -m
PROP_PID=0; EVAL_PID=0
alive()          { kill -0 "$1" 2>/dev/null; }
start_propose()  { bash "$PROPOSE"  & PROP_PID=$!; echo "[search] proposer pool up (pid $PROP_PID, fleet $FLEET)"; }
start_evaluate() { bash "$EVALUATE" & EVAL_PID=$!; echo "[search] evaluator up (pid $EVAL_PID)"; }
term_group()     { [ "$1" -gt 0 ] && kill -TERM -"$1" 2>/dev/null; return 0; }
kill_group()     { [ "$1" -gt 0 ] && kill -KILL -"$1" 2>/dev/null; return 0; }

# Graceful shutdown: TERM both groups, wait up to GRACE seconds, KILL any survivor, then
# reconcile the DB (a model the evaluator was mid-scoring is reset queued -> consistent).
# Cleanup lives ONLY in the EXIT trap, so it runs exactly once on every path -- normal exit or
# signal (the INT/TERM trap just exits, which fires EXIT). shutdown() masks INT/TERM up front so a
# signal arriving mid-teardown (e.g. during the grace wait on the normal-exit path) can't abort it.
shutdown() {
  trap '' INT TERM   # teardown is non-interruptible: a signal here must not abort cleanup
  echo "[search] $KEY -- stopping loops..."
  term_group "$PROP_PID"; term_group "$EVAL_PID"
  for _ in $(seq 1 "${GRACE:-20}"); do
    alive "$PROP_PID" || alive "$EVAL_PID" || break
    sleep 1
  done
  kill_group "$PROP_PID"; kill_group "$EVAL_PID"
  wait "$PROP_PID" "$EVAL_PID" 2>/dev/null || true
  reconcile
  summary
  echo "[search] $KEY -- done."
}
trap 'exit 130' INT TERM
trap shutdown EXIT

echo "[search] $KEY -- always-on search (dataset=$DATASET roi=$ROI imsize=$IMSIZE fleet=$FLEET) for ${HOURS}h"
start_propose
start_evaluate

# Supervise until the wall-clock deadline: restart either loop if it exits early so the search
# stays always-on; the poll interval doubles as restart backoff (no hot-spin).
CHECK=5
while :; do
  now=$(date +%s)
  [ "$now" -ge "$deadline" ] && break
  alive "$PROP_PID" || { kill_group "$PROP_PID"; wait "$PROP_PID" 2>/dev/null || true; echo "[search] proposer pool exited; restarting" >&2; start_propose; }
  alive "$EVAL_PID" || { kill_group "$EVAL_PID"; wait "$EVAL_PID" 2>/dev/null || true; echo "[search] evaluator exited; restarting"     >&2; start_evaluate; }
  rem=$(( deadline - now )); s=$CHECK; [ "$rem" -lt "$s" ] && s="$rem"; [ "$s" -lt 1 ] && s=1
  sleep "$s"
done

echo "[search] $KEY -- ${HOURS}h budget reached."
exit 0   # the EXIT trap runs shutdown exactly once
