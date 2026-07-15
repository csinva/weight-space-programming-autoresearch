#!/usr/bin/env bash
# scripts/simplify/simplify.sh -- behavior-preserving refactor of a search run's CHAMPION model.py.
#
# Unlike scripts/distill (which ABLATES to move toward a principle, bounded by test_corr), this keeps
# the computation EXACTLY the same and only cleans up the CODE. The hard bar is OUTPUT EQUIVALENCE:
# the refactored model must produce activations numerically equal (torch.allclose) to the frozen
# original on a fixed synthetic batch. That check needs no ridge readout, no dataset, no GPU, so it
# returns sub-second and the agent gets many fast iterations. One headless Copilot agent gets a fixed
# wall-clock budget (default 15 min); optionally we record ONE final ridge test_corr as a backstop.
#
#   scripts/simplify/simplify.sh <run-dir> [--minutes 15] [--rtol 1e-4] [--atol 1e-6] \
#                                [--no-final-score] [--effort high] [--model claude-opus-4.8]
#   e.g.  scripts/simplify/simplify.sh runs/nsd__2026-06-24
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || { echo "[simplify] cannot cd to repo root" >&2; exit 1; }
PY="${PYTHON:-$REPO/.venv/bin/python}"; [ -x "$PY" ] || PY=python
[ -x "$REPO/.venv/bin/python" ] && export PATH="$REPO/.venv/bin:$PATH"   # the agent's bare `python` = venv

# --- args ---------------------------------------------------------------------------------------
RUN=""; declare -A OPT; FINAL_SCORE=1
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    --no-final-score) FINAL_SCORE=0; shift;;
    --*=*) k="${1#--}"; k="${k%%=*}"; OPT[$k]="${1#*=}"; shift;;
    --*)   k="${1#--}"; OPT[$k]="${2:?$1 needs a value}"; shift 2;;
    *)     [ -z "$RUN" ] && { RUN="$1"; shift; continue; }; echo "[simplify] unexpected arg: $1" >&2; exit 2;;
  esac
done
[ -n "$RUN" ] || { echo "[simplify] usage: simplify.sh <run-dir> [--minutes N] [--rtol R] [--atol A] [--no-final-score] [--effort E] [--model M]" >&2; exit 2; }
RUN="$(cd "$RUN" 2>/dev/null && pwd)" || { echo "[simplify] no such run dir: $1" >&2; exit 2; }
[ -f "$RUN/search.db" ] || { echo "[simplify] $RUN has no search.db (not a search run)" >&2; exit 2; }
MINUTES="${OPT[minutes]:-15}"; RTOL="${OPT[rtol]:-1e-4}"; ATOL="${OPT[atol]:-1e-6}"
EFFORT="${OPT[effort]:-high}"; MODEL="${OPT[model]:-claude-opus-4.8}"
DATASET="$(basename "$RUN" | sed 's/__.*//')"

# --- eval env: read back EXACTLY what the search scored with (for capture_spec + the final score) --
if [ -f "$RUN/env.json" ]; then
  while IFS=$'\t' read -r k v; do [ -n "$k" ] && export "$k=$v"; done < <(
    "$PY" -c "import json;[print(f'{k}\t{v}') for k,v in json.load(open('$RUN/env.json')).items()]")
  echo "[simplify] eval env <- $RUN/env.json"
else
  case "$DATASET" in
    nsd)  export NEURO_SUBJECT="${NEURO_SUBJECT:-1,2}";;
    ieeg) export NEURO_SUBJECT="${NEURO_SUBJECT:-01,02,03}";;
    nnn)  export NEURO_SUBJECT="${NEURO_SUBJECT:-1,2,3}";;
    huth) export NEURO_SUBJECT="${NEURO_SUBJECT:-UTS01,UTS02,UTS03}";;
  esac
  echo "[simplify] no env.json -> dataset defaults (NEURO_SUBJECT=${NEURO_SUBJECT:-unset})"
fi
export NEURO_DATASET="$DATASET"
echo "[simplify] run=$RUN dataset=$DATASET subject=${NEURO_SUBJECT:-default} rtol=$RTOL atol=$ATOL budget=${MINUTES}m"

# --- resolve the champion -----------------------------------------------------------------------
CHAMP="$(LAB_RUN="$RUN" "$PY" hub.py champion)"
read -r CHAMP_ID CHAMP_CORR CHAMP_PATH <<EOF
$(printf '%s' "$CHAMP" | "$PY" -c "import json,sys; r=json.load(sys.stdin) or {}; print(r.get('id') or '', r.get('test_corr') if r.get('test_corr') is not None else '', r.get('path') or '')")
EOF
[ -n "$CHAMP_ID" ] && [ -f "$CHAMP_PATH" ] || { echo "[simplify] no scored champion in $RUN" >&2; exit 2; }
echo "[simplify] champion=$CHAMP_ID recorded_test_corr=${CHAMP_CORR:-?}"

# --- scaffold the PRIVATE simplify run (its own search.db, separate from the parent run's) --------
SIMP="$RUN/simplify"; WORK="$SIMP/work"; RESULT="$SIMP/result"; LOG="$SIMP/simplify.log"
SPEC="$WORK/input_spec.json"
"$PY" -c "import sys; sys.path.insert(0,'$REPO'); import hub; hub.init('$SIMP')"
"$PY" -c "import sys; sys.path.insert(0,'$REPO'); from scripts.setup_run import make_sandbox; make_sandbox('$WORK','$REPO')"
rm -f "$WORK/CLAUDE.md"                    # drop the proposer orientation; the simplifier uses its own prompt
mkdir -p "$SIMP/eval" "$RESULT"
cp -f "$CHAMP_PATH" "$WORK/model.py"       # seed the agent with the CHAMPION
cp -f "$CHAMP_PATH" "$WORK/reference.py"   # FROZEN reference: defines "same output" (agent must not edit)

# --- capture the synthetic-batch input spec (loads the real split ONCE, then discards the data) ---
echo "[simplify] capturing input spec (shape/dtype/range) from the real split ..."
"$PY" -m scripts.simplify.capture_spec "$SPEC" >/dev/null || { echo "[simplify] capture_spec failed" >&2; exit 1; }
echo "[simplify] input spec -> $SPEC : $(cat "$SPEC" | tr -d '\n' | tr -s ' ')"

# --- sanity: the untouched champion must be equivalent to itself (guards the harness/spec) --------
SANITY="$(LAB_RUN="$SIMP" "$PY" hub.py check-equivalence "$WORK/model.py" --reference "$WORK/reference.py" --spec "$SPEC" --rtol "$RTOL" --atol "$ATOL")"
echo "[simplify] self-equivalence sanity: $(printf '%s' "$SANITY" | tr -d '\n' | tr -s ' ')"
printf '%s' "$SANITY" | "$PY" -c "import json,sys; assert json.load(sys.stdin).get('equivalent') is True, 'champion is not equivalent to itself -- spec/harness problem'" \
  || { echo "[simplify] ABORT: self-equivalence check failed" >&2; exit 1; }

# --- the simplifier agent: a time-boxed headless Copilot confined to WORK -------------------------
PROMPT="$(cat "$REPO/scripts/simplify/simplifier.md")

## THIS SIMPLIFICATION
- Dataset: ${DATASET}. Champion recorded test_corr = ${CHAMP_CORR:-?} (unchanged: you preserve behavior).
- HARD BAR: after EVERY edit, \`python hub.py check-equivalence model.py --reference reference.py --spec input_spec.json\`
  must report \"equivalent\": true (torch.allclose rtol=${RTOL}, atol=${ATOL}). If false/error, REVERT that edit.
- Budget: ~${MINUTES} minutes of wall clock, then you are stopped. The check is cheap -- take MANY small
  verified steps and ALWAYS leave an equivalent model.py on disk."

export LAB_RUN="$SIMP" NEURO_DATASET="$DATASET"   # not island-blinded (no LAB_PROPOSER)
echo "[simplify] agent start ($MODEL / $EFFORT, ${MINUTES}m budget) -> $LOG"
echo "[simplify] === agent session $(date +%FT%T) ===" >>"$LOG"
timeout "${MINUTES}m" copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
    --add-dir "$WORK" -C "$WORK" >>"$LOG" 2>&1
echo "[simplify] agent stopped (exit ${PIPESTATUS[0]:-$?})"

# --- finalize: verify equivalence of the on-disk result, then optionally record one ridge score ---
FINAL_EQ="$(LAB_RUN="$SIMP" "$PY" hub.py check-equivalence "$WORK/model.py" --reference "$WORK/reference.py" --spec "$SPEC" --rtol "$RTOL" --atol "$ATOL")"
EQ_OK="$(printf '%s' "$FINAL_EQ" | "$PY" -c "import json,sys; print('yes' if json.load(sys.stdin).get('equivalent') is True else 'no')")"
echo "[simplify] final on-disk equivalence: $EQ_OK -- $(printf '%s' "$FINAL_EQ" | tr -d '\n' | tr -s ' ')"

if [ "$EQ_OK" = "yes" ]; then
  cp -f "$WORK/model.py" "$RESULT/simplified_model.py"
else
  echo "[simplify] WARNING: on-disk model.py is NOT equivalent -- restoring the frozen reference as the result" >&2
  cp -f "$WORK/reference.py" "$RESULT/simplified_model.py"
fi

REF_LOC="$("$PY" -c "print(sum(1 for l in open('$WORK/reference.py') if l.strip()))")"
OUT_LOC="$("$PY" -c "print(sum(1 for l in open('$RESULT/simplified_model.py') if l.strip()))")"
FINAL_CORR="n/a (skipped)"
if [ "$FINAL_SCORE" = 1 ] && [ "$EQ_OK" = "yes" ]; then
  echo "[simplify] recording one final ridge test_corr (backstop) via a warm evaluator ..."
  LAB_RUN="$SIMP" NEURO_DATASET="$DATASET" "$PY" search/evaluator.py --dataset "$DATASET" --max-repairs 0 \
      >"$SIMP/eval/evaluator.log" 2>&1 &
  EVAL_PID=$!; trap 'kill "$EVAL_PID" 2>/dev/null; wait "$EVAL_PID" 2>/dev/null || true' EXIT INT TERM
  FID="$(LAB_RUN="$SIMP" "$PY" hub.py propose-model "$RESULT/simplified_model.py" --idea "simplified champion (behavior-preserving)")"
  FROW="$(LAB_RUN="$SIMP" "$PY" hub.py wait-for-score "$FID" --timeout 1200)"
  FINAL_CORR="$(printf '%s' "$FROW" | "$PY" -c "import json,sys; v=json.load(sys.stdin).get('test_corr'); print(v if v is not None else 'error')")"
fi

# --- summary ------------------------------------------------------------------------------------
{
  echo "== simplify summary =="
  echo "run             : $RUN"
  echo "champion        : $CHAMP_ID  recorded_test_corr=${CHAMP_CORR:-?}"
  echo "equivalence bar : torch.allclose(rtol=$RTOL, atol=$ATOL) on a synthetic batch ($(basename "$SPEC"))"
  echo "final equivalent: $EQ_OK"
  echo "code size (non-blank loc): reference $REF_LOC -> simplified $OUT_LOC  (delta $((OUT_LOC - REF_LOC)))"
  echo "final recorded test_corr : $FINAL_CORR (champion ${CHAMP_CORR:-?}; should match -- behavior preserved)"
  echo "simplified model -> $RESULT/simplified_model.py"
} | tee "$RESULT/summary.txt"
echo "[simplify] done -> $RESULT"
