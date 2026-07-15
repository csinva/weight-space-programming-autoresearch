#!/usr/bin/env bash
# scripts/distill/distill.sh -- distill a search run's CHAMPION model.py into its clearest
# computational principle, WITHOUT losing predictivity.
#
# One headless Copilot agent gets a fixed wall-clock budget (default 15 min) to rewrite the champion
# into the simplest, clearest computation it can, under ONE hard constraint: the held-out reward must
# stay within --tolerance (default 0.5%) of the champion's re-scored value. Scoring is the SAME fixed
# harness the search used -- a dedicated WARM evaluator (search/evaluator.py) drains a PRIVATE distill
# queue (runs/<run>/distill/search.db), so the split loads ONCE and the agent gets many evals inside
# its budget. Nothing under harness/ or hub.py is editable: only model.py evolves.
#
#   scripts/distill/distill.sh <run-dir> [--minutes 15] [--tolerance 0.005] \
#                              [--effort high] [--model claude-opus-4.8]
#   e.g.  scripts/distill/distill.sh runs/nsd__2026-06-24
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || { echo "[distill] cannot cd to repo root" >&2; exit 1; }
PY="${PYTHON:-$REPO/.venv/bin/python}"; [ -x "$PY" ] || PY=python
[ -x "$REPO/.venv/bin/python" ] && export PATH="$REPO/.venv/bin:$PATH"   # the agent's bare `python` = venv

# --- args ---------------------------------------------------------------------------------------
RUN=""; declare -A OPT
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    --*=*) k="${1#--}"; k="${k%%=*}"; OPT[$k]="${1#*=}"; shift;;
    --*)   k="${1#--}"; OPT[$k]="${2:?$1 needs a value}"; shift 2;;
    *)     [ -z "$RUN" ] && { RUN="$1"; shift; continue; }; echo "[distill] unexpected arg: $1" >&2; exit 2;;
  esac
done
[ -n "$RUN" ] || { echo "[distill] usage: distill.sh <run-dir> [--minutes N] [--tolerance T] [--effort E] [--model M]" >&2; exit 2; }
RUN="$(cd "$RUN" 2>/dev/null && pwd)" || { echo "[distill] no such run dir: $1" >&2; exit 2; }
[ -f "$RUN/search.db" ] || { echo "[distill] $RUN has no search.db (not a search run)" >&2; exit 2; }
MINUTES="${OPT[minutes]:-15}"; TOL="${OPT[tolerance]:-0.005}"
EFFORT="${OPT[effort]:-high}"; MODEL="${OPT[model]:-claude-opus-4.8}"
DATASET="$(basename "$RUN" | sed 's/__.*//')"

# --- eval env: read back EXACTLY what the search scored with -------------------------------------
if [ -f "$RUN/env.json" ]; then
  while IFS=$'\t' read -r k v; do [ -n "$k" ] && export "$k=$v"; done < <(
    "$PY" -c "import json;[print(f'{k}\t{v}') for k,v in json.load(open('$RUN/env.json')).items()]")
  echo "[distill] eval env <- $RUN/env.json"
else
  # runs scaffolded before env.json existed: fall back to search.sh's per-dataset subject defaults.
  case "$DATASET" in
    nsd)  export NEURO_SUBJECT="${NEURO_SUBJECT:-1,2}";;
    ieeg) export NEURO_SUBJECT="${NEURO_SUBJECT:-01,02,03}";;
    nnn)  export NEURO_SUBJECT="${NEURO_SUBJECT:-1,2,3}";;
    huth) export NEURO_SUBJECT="${NEURO_SUBJECT:-UTS01,UTS02,UTS03}";;
  esac
  echo "[distill] no env.json -> dataset defaults (NEURO_SUBJECT=${NEURO_SUBJECT:-unset})"
fi
export NEURO_DATASET="$DATASET"
echo "[distill] run=$RUN dataset=$DATASET subject=${NEURO_SUBJECT:-default} tol=$TOL budget=${MINUTES}m"

# --- resolve the champion -----------------------------------------------------------------------
CHAMP="$(LAB_RUN="$RUN" "$PY" hub.py champion)"
read -r CHAMP_ID CHAMP_CORR CHAMP_PATH <<EOF
$(printf '%s' "$CHAMP" | "$PY" -c "import json,sys; r=json.load(sys.stdin) or {}; print(r.get('id') or '', r.get('test_corr') if r.get('test_corr') is not None else '', r.get('path') or '')")
EOF
[ -n "$CHAMP_ID" ] && [ -f "$CHAMP_PATH" ] || { echo "[distill] no scored champion in $RUN" >&2; exit 2; }
echo "[distill] champion=$CHAMP_ID recorded_test_corr=${CHAMP_CORR:-?}"

# --- scaffold the PRIVATE distill run (its own search.db, separate from the parent run's) --------
DISTILL="$RUN/distill"; WORK="$DISTILL/work"; RESULT="$DISTILL/result"; LOG="$DISTILL/distill.log"
"$PY" -c "import sys; sys.path.insert(0,'$REPO'); import hub; hub.init('$DISTILL')"
"$PY" -c "import sys; sys.path.insert(0,'$REPO'); from scripts.setup_run import make_sandbox; make_sandbox('$WORK','$REPO')"
rm -f "$WORK/CLAUDE.md"                    # drop the proposer orientation; the distiller uses its own prompt
mkdir -p "$DISTILL/eval" "$RESULT"
cp -f "$CHAMP_PATH" "$WORK/model.py"       # seed the agent with the CHAMPION, not the baseline

# --- start the WARM evaluator (single worker, NO repair so it never rewrites a variant) ----------
LAB_RUN="$DISTILL" NEURO_DATASET="$DATASET" "$PY" search/evaluator.py \
    --dataset "$DATASET" --max-repairs 0 >"$DISTILL/eval/evaluator.log" 2>&1 &
EVAL_PID=$!
cleanup() { kill "$EVAL_PID" 2>/dev/null; wait "$EVAL_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "[distill] warm evaluator up (pid $EVAL_PID) -> $DISTILL/eval/evaluator.log"

# --- re-score the champion through the SAME harness -> the reference reward ----------------------
REF_ID="$(LAB_RUN="$DISTILL" "$PY" hub.py propose-model "$WORK/model.py" --idea "champion (reference)")"
echo "[distill] reference proposal $REF_ID -- scoring (loads the split once) ..."
REF_ROW="$(LAB_RUN="$DISTILL" "$PY" hub.py wait-for-score "$REF_ID" --timeout 1200)"
REF_CORR="$(printf '%s' "$REF_ROW" | "$PY" -c "import json,sys; v=json.load(sys.stdin).get('test_corr'); print(v if v is not None else '')")"
[ -n "$REF_CORR" ] || { echo "[distill] champion failed to re-score: $REF_ROW" >&2; exit 1; }
FLOOR="$("$PY" -c "print(round(float($REF_CORR)*(1-float($TOL)),4))")"
"$PY" -c "
ref=float('$REF_CORR'); rec='$CHAMP_CORR'
if rec:
    d=abs(ref-float(rec))
    print(f'[distill] WARNING: reference {ref:.4f} differs from recorded champion {float(rec):.4f} by {d:.4f} -- eval env may not match the search' if d>0.01 else f'[distill] reference matches recorded champion ({ref:.4f})')
"
echo "[distill] reference test_corr=$REF_CORR  ->  FLOOR (>= within $TOL) = $FLOOR"

# --- the distiller agent: a time-boxed headless Copilot confined to WORK -------------------------
PROMPT="$(cat "$REPO/scripts/distill/distiller.md")

## THIS DISTILLATION
- Dataset: ${DATASET}. Champion reference test_corr = ${REF_CORR}.
- HARD FLOOR: every model you KEEP must score test_corr >= ${FLOOR} (within ${TOL} of the reference).
  A model below the floor is a FAILURE -- revert it.
- Budget: ~${MINUTES} minutes of wall clock, then you are stopped. Pace yourself and ALWAYS leave
  your best clear, PASSING model in model.py.
- Score with:  id=\$(python hub.py propose-model model.py --idea \"...\"); python hub.py wait-for-score \$id"

export LAB_RUN="$DISTILL" NEURO_DATASET="$DATASET"   # NOTE: no LAB_PROPOSER -> not island-blinded
echo "[distill] agent start ($MODEL / $EFFORT, ${MINUTES}m budget) -> $LOG"
echo "[distill] === agent session $(date +%FT%T) ===" >>"$LOG"
timeout "${MINUTES}m" copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
    --add-dir "$WORK" -C "$WORK" >>"$LOG" 2>&1
echo "[distill] agent stopped (exit ${PIPESTATUS[0]:-$?})"

# --- finalize: distilled model + full variant ledger + summary ----------------------------------
cp -f "$WORK/model.py" "$RESULT/distilled_model.py"
LAB_RUN="$DISTILL" "$PY" - "$REF_CORR" "$FLOOR" "$DISTILL" <<'PYEOF' | tee "$RESULT/summary.txt"
import json, os, sqlite3, sys
ref, floor, distill = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
con = sqlite3.connect(os.path.join(distill, "search.db")); con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(
    "SELECT id, idea, test_corr, n_params, model_loc, status FROM models "
    "ORDER BY (test_corr IS NULL), test_corr DESC")]
con.close()
json.dump(rows, open(os.path.join(distill, "result", "ledger.json"), "w"), indent=2, default=str)
ok = [r for r in rows if r["test_corr"] is not None]
passing = [r for r in ok if r["test_corr"] >= floor]
print(f"champion reference test_corr = {ref:.4f}   floor (>= {floor:.4f}, {len(passing)}/{len(ok)} scored variants pass)")
print(f"{'id':<7}{'test_corr':>10}{'n_params':>11}{'loc':>6}  idea")
for r in ok[:25]:
    tag = "PASS" if r["test_corr"] >= floor else "fail"
    print(f"{r['id']:<7}{r['test_corr']:>10.4f}{(r['n_params'] or 0):>11}{(r['model_loc'] or 0):>6}  [{tag}] {(r['idea'] or '')[:58]}")
print("\ndistilled model -> " + os.path.join(distill, "result", "distilled_model.py"))
print("full ledger     -> " + os.path.join(distill, "result", "ledger.json"))
PYEOF
echo "[distill] done -> $RESULT"
