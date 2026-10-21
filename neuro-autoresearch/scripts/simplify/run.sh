#!/usr/bin/env bash
# Simplify a run's champion while preserving either its activations or its neural score.
#
#   scripts/simplify/run.sh RUN --mode exact
#   scripts/simplify/run.sh RUN --mode approximate [--tolerance 0.005]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
source "$REPO/search/lib.sh"
use_project_python || exit 2
[ -f search/config.env ] && . search/config.env

case "${1:-}" in -h|--help) sed -n '2,5p' "$0"; exit 0;; "") sed -n '2,5p' "$0"; exit 2;; esac
RUN="$1"; shift
MODE=exact; MINUTES=15; TOLERANCE=0.005; EFFORT=high; MODEL="${NEURO_SIMPLIFY_MODEL:-codex:astra}"
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
DATASET="$("$PY" -c 'import sys;from harness.protocols import validate_search_dataset;print(validate_search_dataset(sys.argv[1]))' "$DATASET")" || exit 2
"$PY" search/providers.py preflight --agents "$MODEL" --effort "$EFFORT" || exit 2

# Restore only scalar scientific environment values, never protocol/provider metadata.
# Refuse historical runs without an RSA contract, then verify the code/subjects match that contract.
[ -f "$RUN/env.json" ] || { echo "Run has no versioned RSA contract; cannot simplify it under a different reward" >&2; exit 2; }
ENV_LINES="$("$PY" - "$RUN/env.json" <<'PY_ENV'
import json, sys
from search.setup_run import EVAL_ENV_KEYS
with open(sys.argv[1]) as handle:
    snapshot = json.load(handle)
if snapshot.get("NEURO_METRIC") != "rsa" or snapshot.get("NEURO_STAGE") != "search":
    raise SystemExit("Simplification requires an RSA search snapshot")
keys = set(EVAL_ENV_KEYS) | {"NEURO_DATASET", "NEURO_METRIC", "NEURO_STAGE"}
for key in sorted(keys):
    if key in snapshot:
        value = snapshot[key]
        if not isinstance(value, str) or "\n" in value or "\t" in value:
            raise SystemExit(f"Invalid scalar setting {key}")
        print(f"{key}\t{value}")
PY_ENV
)" || exit 2
while IFS=$'\t' read -r key value; do export "$key=$value"; done <<< "$ENV_LINES"
"$PY" - "$RUN" "$DATASET" <<'PY_CHECK'
import sys
from search.setup_run import write_env_json
write_env_json(sys.argv[1], sys.argv[2])
PY_CHECK
[ "$?" = 0 ] || exit 2
export NEURO_DATASET="$DATASET" NEURO_METRIC=rsa NEURO_STAGE=search

CHAMPION="$(LAB_RUN="$RUN" "$PY" hub.py champion)"
read -r CHAMPION_ID CHAMPION_SCORE CHAMPION_PATH <<EOF
$(printf '%s' "$CHAMPION" | "$PY" -c "import json,sys;r=json.load(sys.stdin) or {};v=r.get('test_corr');print(r.get('id') or '','' if v is None else v,r.get('path') or '')")
EOF
[ -f "$CHAMPION_PATH" ] || { echo "no scored champion in $RUN" >&2; exit 2; }

OUT="$RUN/simplify/$MODE"; WORK="$OUT/work"; RESULT="$OUT/result"
"$PY" - "$OUT" "$DATASET" <<'PY_INIT'
import sys
import hub
from search.setup_run import write_env_json
write_env_json(sys.argv[1], sys.argv[2])
hub.init(sys.argv[1])
PY_INIT
[ "$?" = 0 ] || exit 2
"$PY" -c "import sys;sys.path.insert(0,'$REPO/search');from setup_run import make_sandbox;make_sandbox('$WORK','$REPO')"
rm -f "$WORK/CLAUDE.md" "$WORK/AGENTS.md" "$WORK/GEMINI.md"
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
  LAB_RUN="$OUT" "$PY" search/evaluator/worker.py --dataset "$DATASET" --max-repairs 0 >"$OUT/eval/evaluator.log" 2>&1 &
  EVALUATOR=$!
  trap 'kill "$EVALUATOR" 2>/dev/null; wait "$EVALUATOR" 2>/dev/null || true' EXIT INT TERM
  ID="$(LAB_RUN="$OUT" "$PY" hub.py propose-model "$WORK/model.py" --idea "champion reference")"
  ROW="$(LAB_RUN="$OUT" "$PY" hub.py wait-for-score "$ID" --timeout 1200)"
  REFERENCE_SCORE="$(printf '%s' "$ROW" | "$PY" -c "import json,sys;v=json.load(sys.stdin).get('test_corr');print('' if v is None else v)")"
  [ -n "$REFERENCE_SCORE" ] || { echo "champion failed to re-score" >&2; exit 1; }
  FLOOR="$("$PY" -c "s=float('$REFERENCE_SCORE');print(s-abs(s)*float('$TOLERANCE'))" )"
  RULES="$(printf 'APPROXIMATE mode: simplify the computation itself. Keep a change only when test_corr >= %s. Score every edit with:\n\n    id=$(python hub.py propose-model model.py --idea "short label")\n    python hub.py wait-for-score $id\n\nRevert changes below the floor.' "$FLOOR")"
fi

PROMPT="$(cat scripts/simplify/prompt.md)

$RULES

Search dataset: $DATASET. Reward is RSA (test_corr compatibility field). Never access NSD/TVSD final encoding scores. Budget: $MINUTES minutes. Always leave a passing model.py."
export LAB_RUN="$OUT"
echo "[simplify] mode=$MODE run=$RUN champion=$CHAMPION_ID budget=${MINUTES}m"
timeout "${MINUTES}m" "$PY" "$REPO/search/providers.py" run --agent "$MODEL" --effort "$EFFORT" \
  --cwd "$WORK" --write-dir "$OUT" <<< "$PROMPT" >>"$OUT/simplify.log" 2>&1

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
