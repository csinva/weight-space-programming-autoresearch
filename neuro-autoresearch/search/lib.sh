#!/usr/bin/env bash
# search/lib.sh -- shared supervisor primitives, sourced by run.sh and proposer/pool.sh.
# Both scripts run the SAME pattern: spawn child(ren), restart on death until a wall-clock
# deadline, then TERM -> grace -> KILL on shutdown. Define those primitives ONCE here so the
# two supervisors stay in lock-step instead of hand-rolling the loop twice.

now_epoch()     { date +%s; }
# Echo the epoch H hours from now (H may be fractional).
deadline_in()   { awk -v n="$(now_epoch)" -v h="$1" 'BEGIN { printf "%d", n + h*3600 }'; }
past_deadline() { local d="${1:-0}"; [ "$d" -gt 0 ] && [ "$(now_epoch)" -ge "$d" ]; }
alive()         { kill -0 "$1" 2>/dev/null; }

# Single-PID signals (proposer/pool.sh starts workers in this group).
term()          { [ "${1:-0}" -gt 0 ] && kill -TERM "$1" 2>/dev/null; return 0; }
# Process-GROUP signals (run.sh runs each child under `set -m`, one group each).
term_group()    { [ "${1:-0}" -gt 0 ] && kill -TERM -"$1" 2>/dev/null; return 0; }
kill_group()    { [ "${1:-0}" -gt 0 ] && kill -KILL -"$1" 2>/dev/null; return 0; }

# Install the standard "TERM on signal, run FUNC on exit" trap pair.
install_shutdown_trap() { trap 'exit 130' INT TERM; trap "$1" EXIT; }
