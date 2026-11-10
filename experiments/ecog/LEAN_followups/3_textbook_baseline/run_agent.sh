#!/bin/bash
# Launch textbook-baseline agent K inside a bubblewrap sandbox.
# Visible to the agent: /work (task, smoke test, example audio; read-write), a fresh /home/agent holding only
# the Claude credentials file, read-only system dirs, Python at /opt/venv (the scoring environment's packages,
# read-only), uv and a pip shim at /opt/bin (installs PyPI packages into /work/pkgs), the Claude CLI at /opt/claude.
# Not visible: /gpfs (projects, results, logs, LEAN code, memory), SLURM (under /gpfs/cmd); ssh/scp/sftp/
# rsync/wget/curl/ncat/telnet masked. Claude runs with no MCP servers (no account connectors), skills disabled,
# and only the Bash, Read, Write and Edit tools (no web tools, no subagents, no messaging other sessions).
# usage: run_agent.sh K [test|run]
set -uo pipefail
K=$1; MODE=${2:-run}
BASE=/path/to/ecog_data/lean_textbook_baseline
A=$BASE/agent_$K
VENV=/path/to/ecog_data/circuit-building-autoresearch/.venv
PYS=/path/to/scratch/ondemand/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu
CLAUDE_PKG=/gpfs/home/user/.npm-global/lib/node_modules/@anthropic-ai/claude-code
CRED=/gpfs/home/user/.claude/.credentials.json
mkdir -p $A/work $A/home/.claude
MASK=()
for b in ssh scp sftp rsync wget curl ncat nc telnet; do [ -f /usr/bin/$b ] && [ ! -L /usr/bin/$b ] && MASK+=(--ro-bind /dev/null /usr/bin/$b); done
BW=(env -i /usr/bin/bwrap --unshare-user --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent --new-session
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/sbin /sbin
    --ro-bind /etc /etc --proc /proc --dev /dev --tmpfs /tmp
    --ro-bind $PYS /opt/python --ro-bind $BASE/sbx/venv /opt/venv
    --ro-bind $VENV/lib/python3.10/site-packages /opt/venv/lib/python3.10/site-packages
    --ro-bind $BASE/sbx/bin /opt/bin --ro-bind $CLAUDE_PKG /opt/claude
    --bind $A/home /home/agent --bind $CRED /home/agent/.claude/.credentials.json --bind $A/work /work
    "${MASK[@]}" --chdir /work
    --setenv HOME /home/agent --setenv USER agent --setenv PATH /opt/bin:/opt/venv/bin:/usr/bin:/bin --setenv PYTHONPATH /work/pkgs
    --setenv UV_CACHE_DIR /work/.uv-cache --setenv UV_NO_PROGRESS 1 --setenv LANG C.UTF-8
    --setenv TMPDIR /tmp --setenv OMP_NUM_THREADS 4)
CL=(/opt/claude/bin/claude.exe --model claude-opus-4-7 --output-format stream-json --verbose
    --dangerously-skip-permissions --strict-mcp-config --mcp-config '{"mcpServers":{}}'
    --tools Bash Read Write Edit --disable-slash-commands)
if [ "$MODE" = test ]; then
  "${BW[@]}" /bin/sh -c 'echo "== /:"; ls /; echo "== /gpfs:"; ls /gpfs 2>&1; echo "== home:"; ls -la /home/agent /home/agent/.claude;
    echo "== python:"; python -c "import numpy, scipy, torch, sys; print(sys.executable, numpy.__version__, scipy.__version__, torch.__version__)";
    echo "== pip:"; pip install tabulate 2>&1 | tail -3; python -c "import tabulate, numpy; print(tabulate.__file__, numpy.__file__, numpy.__version__)"; rm -rf /work/pkgs /work/.uv-cache;
    echo "== masked/absent:"; for b in ssh srun sbatch squeue curl wget; do printf "%s: " $b; command -v $b || echo absent; done; ssh -V 2>&1 | head -1;
    echo "== claude:"; /opt/claude/bin/claude.exe --version'
  "${BW[@]}" "${CL[@]}" -p "Reply with the single word OK. Then stop." < /dev/null > $BASE/logs/sandbox_test_$K.jsonl 2>&1
  python3 - "$BASE/logs/sandbox_test_$K.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    try: e = json.loads(line)
    except Exception: print('RAW', line[:200]); continue
    if e.get('type') == 'system' and e.get('subtype') == 'init':
        print('init: cwd', e.get('cwd'), '| model', e.get('model'), '| mcp_servers', e.get('mcp_servers'), '| tools', sorted(e.get('tools', [])))
    if e.get('type') == 'result':
        print('result:', repr(e.get('result'))[:100], '| is_error', e.get('is_error'))
PY
  exit 0
fi
NONCE=$(python3 -c "import secrets; print(secrets.token_hex(6))")
echo "agent $K nonce $NONCE start $(date -Is)" >> $BASE/logs/agents.txt
"${BW[@]}" "${CL[@]}" -p "Read /work/TASK.md and complete the task described there. Work independently. (session $NONCE)" \
  < /dev/null > $BASE/logs/agent_$K.jsonl 2> $BASE/logs/agent_$K.err
echo "agent $K end $(date -Is) rc=$?" >> $BASE/logs/agents.txt
