#!/usr/bin/env bash
# Simplify a search run's champion while preserving either its activations or its neural score.
#
#   scripts/simplify/simplify.sh RUN --mode exact
#   scripts/simplify/simplify.sh RUN --mode approximate [--tolerance 0.005]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
PY="${PYTHON:-$REPO/.venv/bin/python}"; [ -x "$PY" ] || PY=python
[ -x "$REPO/.venv/bin/python" ] && export PATH="$REPO/.venv/bin:$PATH"
[ -f search/config.env ] && . search/config.env

RUN=""; MODE=exact; FINAL_SCORE=1
MINUTES=15; TOL=0.005; RTOL=1e-4; ATOL=1e-6
EFFORT=high; MODEL=claude-opus-4.8
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) sed -n '2,5p' "$0"; exit 0;;
    --no-final-score) FINAL_SCORE=0; shift;;
    --mode|--minutes|--tolerance|--rtol|--atol|--effort|--model)
      name="${1#--}"; value="${2:?$1 needs a value}"; shift 2
      case "$name" in
        mode) MODE="$value";; minutes) MINUTES="$value";; tolerance) TOL="$value";;
        rtol) RTOL="$value";; atol) ATOL="$value";; effort) EFFORT="$value";; model) MODEL="$value";;
      esac;;
    --*) echo "unknown option: $1" >&2; exit 2;;
    *) [ -z "$RUN" ] && { RUN="$1"; shift; continue; }; echo "unexpected arg: $1" >&2; exit 2;;
  esac
done
case "$MODE" in exact|approximate) ;; *) echo "--mode must be exact or approximate" >&2; exit 2;; esac
[ -n "$RUN" ] || { echo "usage: simplify.sh RUN --mode exact|approximate" >&2; exit 2; }
RUN="$(cd "$RUN" 2>/dev/null && pwd)" || { echo "no such run: $RUN" >&2; exit 2; }
[ -f "$RUN/search.db" ] || { echo "$RUN is not a search run" >&2; exit 2; }

DATASET="$(basename "$RUN" | sed 's/__.*//')"

# Restore the environment used by the original search.
if [ -f "$RUN/env.json" ]; then
  while IFS=$'\t' read -r k v; do [ -n "$k" ] && export "$k=$v"; done < <(
    "$PY" -c "import json;[print(f'{k}\t{v}') for k,v in json.load(open('$RUN/env.json')).items()]")
elif [ -z "${NEURO_SUBJECT:-}" ]; then
  subject="$("$PY" -c "import sys;sys.path.insert(0,'scripts/eval');from splits import search_subjects_csv;print(search_subjects_csv('$DATASET'))")" || exit 1
  [ -n "$subject" ] && export NEURO_SUBJECT="$subject"
fi
export NEURO_DATASET="$DATASET"

CHAMP="$(LAB_RUN="$RUN" "$PY" hub.py champion)"
read -r CHAMP_ID CHAMP_CORR CHAMP_PATH <<EOF
$(printf '%s' "$CHAMP" | "$PY" -c "import json,sys;r=json.load(sys.stdin) or {};print(r.get('id') or '',r.get('test_corr') if r.get('test_corr') is not None else '',r.get('path') or '')")
EOF
[ -n "$CHAMP_ID" ] && [ -f "$CHAMP_PATH" ] || { echo "no scored champion in $RUN" >&2; exit 2; }

SIMP="$RUN/simplify/$MODE"; WORK="$SIMP/work"; RESULT="$SIMP/result"; LOG="$SIMP/simplify.log"
"$PY" -c "import sys;sys.path.insert(0,'$REPO');import hub;hub.init('$SIMP')"
"$PY" -c "import sys;sys.path.insert(0,'$REPO');from scripts.setup_run import make_sandbox;make_sandbox('$WORK','$REPO')"
rm -f "$WORK/CLAUDE.md"
mkdir -p "$SIMP/eval" "$RESULT"
cp -f "$CHAMP_PATH" "$WORK/model.py"

EVAL_PID=""
start_evaluator() {
  LAB_RUN="$SIMP" "$PY" search/evaluator.py --dataset "$DATASET" --max-repairs 0 >"$SIMP/eval/evaluator.log" 2>&1 &
  EVAL_PID=$!
}
cleanup() { [ -n "$EVAL_PID" ] && kill "$EVAL_PID" 2>/dev/null; wait "$EVAL_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [ "$MODE" = exact ]; then
  SPEC="$WORK/input_spec.json"
  cp -f "$CHAMP_PATH" "$WORK/reference.py"
  "$PY" -m scripts.simplify.capture_spec "$SPEC" >/dev/null || exit 1
  CHECK="python hub.py check-equivalence model.py --reference reference.py --spec input_spec.json --rtol $RTOL --atol $ATOL"
  SANITY="$(LAB_RUN="$SIMP" "$PY" hub.py check-equivalence "$WORK/model.py" --reference "$WORK/reference.py" --spec "$SPEC" --rtol "$RTOL" --atol "$ATOL")"
  printf '%s' "$SANITY" | "$PY" -c "import json,sys;assert json.load(sys.stdin).get('equivalent') is True" || exit 1
  RULES="$(printf 'Mode: EXACT. Refactor code without changing activations. After every edit run:\n\n    %s\n\nKeep an edit only when it reports equivalent=true. Never edit reference.py.' "$CHECK")"
else
  start_evaluator
  REF_ID="$(LAB_RUN="$SIMP" "$PY" hub.py propose-model "$WORK/model.py" --idea "champion reference")"
  REF_ROW="$(LAB_RUN="$SIMP" "$PY" hub.py wait-for-score "$REF_ID" --timeout 1200)"
  REF_CORR="$(printf '%s' "$REF_ROW" | "$PY" -c "import json,sys;v=json.load(sys.stdin).get('test_corr');print(v if v is not None else '')")"
  [ -n "$REF_CORR" ] || { echo "champion failed to re-score" >&2; exit 1; }
  FLOOR="$("$PY" -c "print(float('$REF_CORR')*(1-float('$TOL')))" )"
  RULES="$(printf 'Mode: APPROXIMATE. You may remove or replace computation. Keep a change only when neural predictivity stays above %s (reference %s, tolerance %s). Score each edit with:\n\n    id=$(python hub.py propose-model model.py --idea "short label")\n    python hub.py wait-for-score $id\n\nRevert any edit below the floor.' "$FLOOR" "$REF_CORR" "$TOL")"
fi

PROMPT="$(cat scripts/simplify/simplifier.md)

## This run
$RULES

Dataset: $DATASET. Budget: about $MINUTES minutes. Always leave the best passing model in model.py."
export LAB_RUN="$SIMP"
echo "[simplify] mode=$MODE run=$RUN champion=$CHAMP_ID budget=${MINUTES}m"
timeout "${MINUTES}m" copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
  --add-dir "$WORK" -C "$WORK" >>"$LOG" 2>&1
echo "[simplify] agent stopped"

FINAL_OK=yes; FINAL_CORR="n/a"
if [ "$MODE" = exact ]; then
  FINAL="$(LAB_RUN="$SIMP" "$PY" hub.py check-equivalence "$WORK/model.py" --reference "$WORK/reference.py" --spec "$SPEC" --rtol "$RTOL" --atol "$ATOL")"
  FINAL_OK="$(printf '%s' "$FINAL" | "$PY" -c "import json,sys;print('yes' if json.load(sys.stdin).get('equivalent') is True else 'no')")"
  if [ "$FINAL_OK" != yes ]; then
    cp -f "$WORK/reference.py" "$WORK/model.py"
    FINAL_OK="restored reference"
  fi
  if [ "$FINAL_SCORE" = 1 ] && [ "$FINAL_OK" = yes ]; then
    start_evaluator
    ID="$(LAB_RUN="$SIMP" "$PY" hub.py propose-model "$WORK/model.py" --idea "exact simplification")"
    ROW="$(LAB_RUN="$SIMP" "$PY" hub.py wait-for-score "$ID" --timeout 1200)"
    FINAL_CORR="$(printf '%s' "$ROW" | "$PY" -c "import json,sys;print(json.load(sys.stdin).get('test_corr'))")"
  fi
else
  ID="$(LAB_RUN="$SIMP" "$PY" hub.py propose-model "$WORK/model.py" --idea "final approximate simplification")"
  ROW="$(LAB_RUN="$SIMP" "$PY" hub.py wait-for-score "$ID" --timeout 1200)"
  read -r FINAL_CORR FINAL_OK <<EOF
$(printf '%s' "$ROW" | "$PY" -c "import json,sys;v=json.load(sys.stdin).get('test_corr');print(v if v is not None else 'error','yes' if v is not None and v>=float('$FLOOR') else 'no')")
EOF
  if [ "$FINAL_OK" = no ]; then
    FALLBACK="$("$PY" -c "import sqlite3;c=sqlite3.connect('$SIMP/search.db');r=c.execute(\"select path from models where status='ok' and test_corr>=? order by created_at desc limit 1\",(float('$FLOOR'),)).fetchone();print(r[0])")"
    cp -f "$FALLBACK" "$WORK/model.py"
    FINAL_OK="restored latest passing snapshot"
  fi
fi

cp -f "$WORK/model.py" "$RESULT/simplified_model.py"
REF_LOC="$("$PY" -c "print(sum(bool(x.strip()) for x in open('$CHAMP_PATH')))" )"
OUT_LOC="$("$PY" -c "print(sum(bool(x.strip()) for x in open('$RESULT/simplified_model.py')))" )"
{
  echo "mode: $MODE"
  echo "champion: $CHAMP_ID (test_corr=${CHAMP_CORR:-?})"
  [ "$MODE" = exact ] && echo "equivalent: $FINAL_OK (rtol=$RTOL, atol=$ATOL)"
  [ "$MODE" = exact ] && echo "final test_corr: $FINAL_CORR"
  [ "$MODE" = approximate ] && echo "final: $FINAL_OK (test_corr=$FINAL_CORR, floor=$FLOOR)"
  echo "non-blank lines: $REF_LOC -> $OUT_LOC"
  echo "model: $RESULT/simplified_model.py"
} | tee "$RESULT/summary.txt"
