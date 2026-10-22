#!/usr/bin/env bash
# Reproducible three-proposer search; ordinary run.sh options are forwarded.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOW=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --show-config) SHOW=1;;
    --fleet|--fleet=*|--effort|--effort=*)
      echo "[fixed] fleet=3 and proposer effort=medium are fixed by this script" >&2
      exit 2;;
    *) ARGS+=("$arg");;
  esac
done
export NEURO_MODEL_ROSTER="codex:gpt-6-astra:medium claude:claude-opus-5-5:medium agy:gemini-4-argon-medium:medium"
export NEURO_CONDUCTOR_MODEL="codex:gpt-6-astra:xhigh"
export NEURO_CONDUCTOR_EFFORT=xhigh
export NEURO_EFFORT=medium
export NEURO_REPAIR_MODEL="codex:gpt-6-astra"
export NEURO_REPAIR_EFFORT=medium
export AGY_ADC_AUTH=true
# Use Claude Code's existing login; clear only this launch's conflicting routes.
CLAUDE_LOGIN_ENV=(ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN
  CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY ANTHROPIC_BASE_URL)
ENV_UNSET=()
for name in "${CLAUDE_LOGIN_ENV[@]}"; do
  unset "$name"
  ENV_UNSET+=(-u "$name")
done
# Explicit model IDs and per-slot efforts avoid machine-local alias overrides.
echo "[fixed] p00=GPT-6 Astra/medium; p01=Claude Opus 5.5/medium; p02=Gemini 4 Argon/medium"
echo "[fixed] conductor=GPT-6 Astra/xhigh; all provider CLIs skip permission prompts"
printf '[fixed] command: '
printf '%q ' env "${ENV_UNSET[@]}" "NEURO_MODEL_ROSTER=$NEURO_MODEL_ROSTER" \
  "NEURO_CONDUCTOR_MODEL=$NEURO_CONDUCTOR_MODEL" "NEURO_CONDUCTOR_EFFORT=$NEURO_CONDUCTOR_EFFORT" \
  "NEURO_EFFORT=$NEURO_EFFORT" "NEURO_REPAIR_MODEL=$NEURO_REPAIR_MODEL" \
  "NEURO_REPAIR_EFFORT=$NEURO_REPAIR_EFFORT" "AGY_ADC_AUTH=$AGY_ADC_AUTH" bash "$REPO/search/run.sh" "${ARGS[@]}" --fleet 3 --effort medium
printf '\n'
if [ "$SHOW" -eq 1 ]; then
  source "$REPO/search/lib.sh"
  use_project_python
  # Placeholder paths/prompt are replaced by the normal run scaffold and workers.
  PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY_CONFIG'
import os, shlex
from search.providers import command, resolve
roles = [(f"p{i:02d}", spec, "medium", f"<run>/proposers/p{i:02d}")
         for i, spec in enumerate(os.environ["NEURO_MODEL_ROSTER"].split())]
roles += [("conductor", os.environ["NEURO_CONDUCTOR_MODEL"], os.environ["NEURO_CONDUCTOR_EFFORT"], "<run>"),
          ("repair", os.environ["NEURO_REPAIR_MODEL"], os.environ["NEURO_REPAIR_EFFORT"], "<workspace>")]
for role, spec, effort, cwd in roles:
    print(f"[fixed] {role} cwd={cwd}: {shlex.join(command(resolve(spec, effort), cwd, '<role prompt>', [cwd if role == 'repair' else '<run>']))}")
PY_CONFIG
  exit 0
fi
exec bash "$REPO/search/run.sh" "${ARGS[@]}" --fleet 3 --effort medium
