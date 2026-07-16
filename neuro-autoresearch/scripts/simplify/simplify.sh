#!/usr/bin/env bash
# Simplify a run's champion while preserving either its activations or its neural score.
#
#   scripts/simplify/simplify.sh RUN --mode exact
#   scripts/simplify/simplify.sh RUN --mode approximate [--tolerance 0.005]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
PY="${PYTHON:-$REPO/.venv/bin/python}"; [ -x "$PY" ] || PY=python
[ -x "$REPO/.venv/bin/python" ] && export PATH="$REPO/.venv/bin:$PATH"
[ -f search/config.env ] && . search/config.env

case "${1:-}" in -h|--help) sed -n '2,5p' "$0"; exit 0;; "") sed -n '2,5p' "$0"; exit 2;; esac
RUN="$1"; shift
MODE=exact; MINUTES=15; TOLERANCE=0.005; EFFORT=high; MODEL=claude-opus-4.8
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:?--mode needs a value}"; shift 2;;
    --minutes) MINUTES="${2:?--minutes needs a value}"; shift 2;;
    --tolerance) TOLERANCE="${2:?--tolerance needs a value}"; shift 2;;
    --effort) EFFORT="${2:?--effort needs a value}"; shift 2;;
    --model) MODEL="${2:?--model needs a value}"; shift 2;;
    -h|--help) sed -n '2,5p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
case "$MODE" in exact|approximate) ;; *) echo "--mode must be exact or approximate" >&2; exit 2;; esac
RUN="$(cd "$RUN" 2>/dev/null && pwd)" || { echo "no such run: $RUN" >&2; exit 2; }
[ -f "$RUN/search.db" ] || { echo "$RUN is not a search run" >&2; exit 2; }
DATASET="$(basename "$RUN" | sed 's/__.*//')"

# Restore the environment used by the search.
if [ -f "$RUN/env.json" ]; then
  while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(
    "$PY" -c "import json;[print(f'{k}\t{v}') for k,v in json.load(open('$RUN/env.json')).items()]")
elif [ -z "${NEURO_SUBJECT:-}" ]; then
  subject="$("$PY" -c "import sys;sys.path.insert(0,'scripts/eval');from splits import search_subjects_csv;print(search_subjects_csv('$DATASET'))")" || exit 1
  [ -n "$subject" ] && export NEURO_SUBJECT="$subject"
fi
export NEURO_DATASET="$DATASET"

CHAMPION="$(LAB_RUN="$RUN" "$PY" hub.py champion)"
read -r CHAMPION_ID CHAMPION_SCORE CHAMPION_PATH <<EOF
$(printf '%s' "$CHAMPION" | "$PY" -c "import json,sys;r=json.load(sys.stdin) or {};v=r.get('test_corr');print(r.get('id') or '','' if v is None else v,r.get('path') or '')")
EOF
[ -f "$CHAMPION_PATH" ] || { echo "no scored champion in $RUN" >&2; exit 2; }

OUT="$RUN/simplify/$MODE"; WORK="$OUT/work"; RESULT="$OUT/result"
"$PY" -c "import sys;sys.path.insert(0,'$REPO');import hub;hub.init('$OUT')"
"$PY" -c "import sys;sys.path.insert(0,'$REPO');from scripts.setup_run import make_sandbox;make_sandbox('$WORK','$REPO')"
rm -f "$WORK/CLAUDE.md"
mkdir -p "$OUT/eval" "$RESULT"
cp -f "$CHAMPION_PATH" "$WORK/model.py"

if [ "$MODE" = exact ]; then
  SPEC="$WORK/input_spec.json"
  cp -f "$CHAMPION_PATH" "$WORK/reference.py"
  "$PY" -m scripts.simplify.capture_spec "$SPEC"
  CHECK="python hub.py check-equivalence model.py --reference reference.py --spec input_spec.json"
  LAB_RUN="$OUT" "$PY" hub.py check-equivalence "$WORK/model.py" --reference "$WORK/reference.py" --spec "$SPEC" \
    | "$PY" -c "import json,sys;assert json.load(sys.stdin)['equivalent']" || exit 1
  RULES="$(printf 'EXACT mode: refactor without changing activations. After every edit run:\n\n    %s\n\nKeep only equivalent edits. Never edit reference.py.' "$CHECK")"
else
  LAB_RUN="$OUT" "$PY" search/evaluator.py --dataset "$DATASET" --max-repairs 0 >"$OUT/eval/evaluator.log" 2>&1 &
  EVALUATOR=$!
  trap 'kill "$EVALUATOR" 2>/dev/null; wait "$EVALUATOR" 2>/dev/null || true' EXIT INT TERM
  ID="$(LAB_RUN="$OUT" "$PY" hub.py propose-model "$WORK/model.py" --idea "champion reference")"
  ROW="$(LAB_RUN="$OUT" "$PY" hub.py wait-for-score "$ID" --timeout 1200)"
  REFERENCE_SCORE="$(printf '%s' "$ROW" | "$PY" -c "import json,sys;v=json.load(sys.stdin).get('test_corr');print('' if v is None else v)")"
  [ -n "$REFERENCE_SCORE" ] || { echo "champion failed to re-score" >&2; exit 1; }
  FLOOR="$("$PY" -c "s=float('$REFERENCE_SCORE');print(s-abs(s)*float('$TOLERANCE'))" )"
  RULES="$(printf 'APPROXIMATE mode: simplify the computation itself. Keep a change only when test_corr >= %s. Score every edit with:\n\n    id=$(python hub.py propose-model model.py --idea "short label")\n    python hub.py wait-for-score $id\n\nRevert changes below the floor.' "$FLOOR")"
fi

PROMPT="$(cat scripts/simplify/simplifier.md)

$RULES

Dataset: $DATASET. Budget: $MINUTES minutes. Always leave a passing model.py."
export LAB_RUN="$OUT"
echo "[simplify] mode=$MODE run=$RUN champion=$CHAMPION_ID budget=${MINUTES}m"
timeout "${MINUTES}m" copilot -p "$PROMPT" --allow-all --model "$MODEL" --effort "$EFFORT" \
  --add-dir "$WORK" -C "$WORK" >>"$OUT/simplify.log" 2>&1

if [ "$MODE" = exact ]; then
  FINAL="$(LAB_RUN="$OUT" "$PY" hub.py check-equivalence "$WORK/model.py" --reference "$WORK/reference.py" --spec "$SPEC")"
  if printf '%s' "$FINAL" | "$PY" -c "import json,sys;assert json.load(sys.stdin)['equivalent']"; then
    STATUS="equivalent activations"
  else
    cp -f "$WORK/reference.py" "$WORK/model.py"
    STATUS="failed final check; restored champion"
  fi
else
  ID="$(LAB_RUN="$OUT" "$PY" hub.py propose-model "$WORK/model.py" --idea "final approximate simplification")"
  ROW="$(LAB_RUN="$OUT" "$PY" hub.py wait-for-score "$ID" --timeout 1200)"
  FINAL_SCORE="$(printf '%s' "$ROW" | "$PY" -c "import json,sys;v=json.load(sys.stdin).get('test_corr');print('' if v is None else v)")"
  if "$PY" -c "assert float('$FINAL_SCORE') >= float('$FLOOR')" 2>/dev/null; then
    STATUS="test_corr=$FINAL_SCORE >= $FLOOR"
  else
    FALLBACK="$("$PY" -c "import sqlite3;c=sqlite3.connect('$OUT/search.db');r=c.execute(\"select path from models where status='ok' and test_corr>=? order by created_at desc limit 1\",(float('$FLOOR'),)).fetchone();print(r[0])")"
    cp -f "$FALLBACK" "$WORK/model.py"
    STATUS="failed final check; restored latest passing model"
  fi
fi

cp -f "$WORK/model.py" "$RESULT/simplified_model.py"
{
  echo "mode: $MODE"
  echo "champion: $CHAMPION_ID (test_corr=$CHAMPION_SCORE)"
  echo "validation: $STATUS"
  echo "lines: $(wc -l < "$CHAMPION_PATH" | tr -d ' ') -> $(wc -l < "$RESULT/simplified_model.py" | tr -d ' ')"
  echo "model: $RESULT/simplified_model.py"
} | tee "$RESULT/summary.txt"
