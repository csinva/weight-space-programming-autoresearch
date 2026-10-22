#!/usr/bin/env bash
# Simplify an RSA champion; see docs/simplification.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/search/lib.sh"
use_project_python || exit 2
cd "$REPO" || exit 1
[ -f search/config.env ] && . search/config.env

usage() {
  cat <<'HELP'
Usage: scripts/simplify/run.sh RUN [options]

Simplify an RSA run's champion using a live agent. Results go to
RUN/simplify/<mode>/result/; the source champion is unchanged.

  --mode exact|approximate  Preserve checked activations or search RSA (default: exact)
  --tolerance FRACTION      Maximum relative RSA loss in approximate mode (default: 0.005)
  --minutes MINUTES         Agent time budget (default: 15)
  --model AGENT             Provider/model selector (default: codex:astra)
  --effort EFFORT           Provider reasoning effort (default: high)
  -h, --help               Show this help
HELP
}

# Extract a scalar without hiding a valid zero score or interpolating data into code.
json_field() {
  "$PY" -c 'import json, sys; value = (json.load(sys.stdin) or {}).get(sys.argv[1]); print("" if value is None else value)' "$1"
}

case "${1:-}" in -h|--help) usage; exit 0;; "") usage; exit 2;; esac
RUN="$1"; shift
MODE=exact; MINUTES=15; TOLERANCE=0.005; EFFORT=high; MODEL="${NEURO_SIMPLIFY_MODEL:-codex:astra}"
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:?--mode needs a value}"; shift 2;;
    --minutes) MINUTES="${2:?--minutes needs a value}"; shift 2;;
    --tolerance) TOLERANCE="${2:?--tolerance needs a value}"; shift 2;;
    --effort) EFFORT="${2:?--effort needs a value}"; shift 2;;
    --model) MODEL="${2:?--model needs a value}"; shift 2;;
    -h|--help) usage; exit 0;;
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

CHAMPION="$(LAB_RUN="$RUN" "$PY" hub.py champion)" || exit 2
CHAMPION_ID="$(json_field id <<< "$CHAMPION")" || exit 2
CHAMPION_SCORE="$(json_field test_corr <<< "$CHAMPION")" || exit 2
CHAMPION_PATH="$(json_field path <<< "$CHAMPION")" || exit 2
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
"$PY" -c 'import sys; from search.setup_run import make_sandbox; make_sandbox(*sys.argv[1:])' "$WORK" "$REPO" || exit 2
rm -f "$WORK/CLAUDE.md" "$WORK/AGENTS.md" "$WORK/GEMINI.md"
mkdir -p "$OUT/eval" "$RESULT"
cp -f "$CHAMPION_PATH" "$WORK/model.py"

if [ "$MODE" = exact ]; then
  SPEC="$WORK/input_spec.json"
  cp -f "$CHAMPION_PATH" "$WORK/reference.py"
  "$PY" -m scripts.simplify.capture_spec "$SPEC" || exit 2
  CHECK='"$NEURO_PYTHON" hub.py check-equivalence model.py --reference reference.py --spec input_spec.json'
  LAB_RUN="$OUT" "$PY" hub.py check-equivalence "$WORK/model.py" --reference "$WORK/reference.py" --spec "$SPEC" \
    | "$PY" -c "import json,sys;assert json.load(sys.stdin)['equivalent']" || exit 1
  RULES="$(printf 'EXACT mode: refactor without changing activations. After every edit run:\n\n    %s\n\nKeep only equivalent edits. Never edit reference.py.' "$CHECK")"
else
  LAB_RUN="$OUT" "$PY" search/evaluator/worker.py --dataset "$DATASET" --max-repairs 0 >"$OUT/eval/evaluator.log" 2>&1 &
  EVALUATOR=$!
  trap 'kill "$EVALUATOR" 2>/dev/null; wait "$EVALUATOR" 2>/dev/null || true' EXIT INT TERM
  ID="$(LAB_RUN="$OUT" "$PY" hub.py propose-model "$WORK/model.py" --idea "champion reference")"
  ROW="$(LAB_RUN="$OUT" "$PY" hub.py wait-for-score "$ID" --timeout 1200)"
  REFERENCE_SCORE="$(json_field test_corr <<< "$ROW")"
  [ -n "$REFERENCE_SCORE" ] || { echo "champion failed to re-score" >&2; exit 1; }
  FLOOR="$("$PY" -c 'import sys; score, tolerance = map(float, sys.argv[1:]); print(score - abs(score) * tolerance)' "$REFERENCE_SCORE" "$TOLERANCE")"
  RULES="$(printf 'APPROXIMATE mode: simplify the computation itself. Keep a change only when test_corr >= %s. Score every edit with:\n\n    id=$("$NEURO_PYTHON" hub.py propose-model model.py --idea "short label")\n    "$NEURO_PYTHON" hub.py wait-for-score $id\n\nRevert changes below the floor.' "$FLOOR")"
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
  FINAL_SCORE="$(json_field test_corr <<< "$ROW")"
  if "$PY" -c 'import sys; assert float(sys.argv[1]) >= float(sys.argv[2])' "$FINAL_SCORE" "$FLOOR" 2>/dev/null; then
    STATUS="test_corr=$FINAL_SCORE >= $FLOOR"
  else
    FALLBACK="$("$PY" - "$OUT/search.db" "$FLOOR" <<'PY_FALLBACK'
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as connection:
    row = connection.execute(
        "SELECT path FROM models WHERE status='ok' AND test_corr>=? ORDER BY created_at DESC LIMIT 1",
        (float(sys.argv[2]),),
    ).fetchone()
if row is None:
    raise SystemExit("No passing model available to restore")
print(row[0])
PY_FALLBACK
)" || exit 1
    cp -f "$FALLBACK" "$WORK/model.py" || exit 1
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
