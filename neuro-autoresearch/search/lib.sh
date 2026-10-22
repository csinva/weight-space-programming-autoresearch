#!/usr/bin/env bash
# search/lib.sh -- shared supervisor primitives, sourced by run.sh and proposer/pool.sh.
# Both scripts run the SAME pattern: spawn child(ren), restart on death until a wall-clock
# deadline, then TERM -> grace -> KILL on shutdown. Define those primitives ONCE here so the
# two supervisors stay in lock-step instead of hand-rolling the loop twice.

now_epoch()     { date +%s; }
# Echo the epoch H hours from now (H may be fractional).
deadline_in()   { awk -v n="$(now_epoch)" -v h="$1" 'BEGIN { printf "%d", n + h*3600 }'; }
past_deadline() { local d="${1:-0}"; [ "$d" -gt 0 ] && [ "$(now_epoch)" -ge "$d" ]; }
alive()         { [ "${1:-0}" -gt 0 ] && kill -0 "$1" 2>/dev/null; }

# Single-PID signals (proposer/pool.sh starts workers in this group).
# A child may already have exited. Guard kill itself so errexit cannot abort cleanup.
term()          { [ "${1:-0}" -gt 0 ] && kill -TERM "$1" 2>/dev/null || true; }
# Process-GROUP signals (run.sh runs each child under `set -m`, one group each).
term_group()    { [ "${1:-0}" -gt 0 ] && kill -TERM -"$1" 2>/dev/null || true; }
kill_group()    { [ "${1:-0}" -gt 0 ] && kill -KILL -"$1" 2>/dev/null || true; }

# Install the standard "TERM on signal, run FUNC on exit" trap pair.
install_shutdown_trap() { trap 'exit 130' INT TERM; trap "$1" EXIT; }

# Select one interpreter for launchers, agents' hub calls, and evaluator subprocesses.
use_project_python() {
  # NEURO_PYTHON is the explicit override; PYTHON remains a legacy fallback.
  if [ -n "${NEURO_PYTHON:-${PYTHON:-}}" ]; then
    PY="$(command -v "${NEURO_PYTHON:-$PYTHON}")" || { echo "[search] Python executable not found" >&2; return 2; }
  elif [ -x "$REPO/.venv/bin/python" ]; then
    PY="$REPO/.venv/bin/python"
  else
    PY="$(command -v python)" || return 2
  fi
  # Children change working directories. Make the path absolute, but do NOT
  # resolve the executable symlink: venv Python depends on its original path.
  local python_dir
  python_dir="$(cd "$(dirname "$PY")" && pwd)" || return 2
  PY="$python_dir/$(basename "$PY")"
  [ -f "$PY" ] && [ -x "$PY" ] || { echo "[search] Invalid Python executable: $PY" >&2; return 2; }
  export NEURO_PYTHON="$PY"
  # Keep sibling tools available; launch Python explicitly instead of relying
  # on PATH (the selected binary may be named python3 or be a custom wrapper).
  export PATH="$(dirname "$PY"):$PATH"
}
