"""Terminal provider failures must survive both supervisor layers without retries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def supervisor_repo(tmp_path):
    for name in ("search/lib.sh", "search/config.env", "search/run.sh", "search/proposer/pool.sh"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sleep = bindir / "sleep"
    sleep.write_text("#!/bin/sh\nexec /bin/sleep 0.05\n")
    sleep.chmod(0o755)
    python = bindir / "python3"
    python.write_text(f'''#!{sys.executable}
import sys
if sys.argv[1] == '-c' and 'validate_search_dataset' in sys.argv[2]:
    print('laion')
elif sys.argv[1:] == ['hub.py', 'requeue']:
    print(0)
''')
    python.chmod(0o755)
    env = {**os.environ, "NEURO_PYTHON": str(python), "LAB_RUN": str(tmp_path),
           "NEURO_MODEL_ROSTER": "codex:astra agy:argon4", "NEURO_FLEET": "2",
           "NEURO_SUBJECT": "test", "NEURO_METRIC": "rsa", "NEURO_STAGE": "search",
           "PATH": f"{bindir}:{os.environ['PATH']}", "GRACE": "1"}
    return tmp_path, env


def test_pool_disables_failed_slot_but_keeps_healthy_slot(supervisor_repo):
    root, env = supervisor_repo
    (root / "search/proposer/worker.sh").write_text('''#!/bin/bash
echo "$1" >> "$LAB_RUN/starts"
if [ "$1" = p00 ]; then exit 78; fi
# Healthy sessions may complete and be restarted while the failed slot stays disabled.
/bin/sleep 0.1
''')
    env["NEURO_DEADLINE"] = str(int(time.time()) + 2)
    result = subprocess.run(["bash", str(root / "search/proposer/pool.sh")], env=env,
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    starts = (root / "starts").read_text().splitlines()
    assert starts.count("p00") == 1
    assert starts.count("p01") > 1
    assert "p00 disabled: terminal provider failure" in result.stderr
    assert "1 active slots remain" in result.stderr


def test_pool_retries_transient_failure_then_stops_when_all_slots_terminal(supervisor_repo):
    root, env = supervisor_repo
    (root / "search/proposer/worker.sh").write_text('''#!/bin/bash
echo "$1" >> "$LAB_RUN/starts"
if [ "$1" = p01 ] && [ ! -f "$LAB_RUN/transient" ]; then
    touch "$LAB_RUN/transient"
    exit 1
fi
exit 78
''')
    env["NEURO_DEADLINE"] = str(int(time.time()) + 10)
    result = subprocess.run(["bash", str(root / "search/proposer/pool.sh")], env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 78, result.stderr
    starts = (root / "starts").read_text().splitlines()
    assert starts.count("p00") == 1
    assert starts.count("p01") == 2
    assert "all proposer slots disabled" in result.stderr


def test_launcher_preserves_terminal_pool_exit_and_shuts_down(supervisor_repo):
    root, env = supervisor_repo
    (root / "search/proposer/pool.sh").write_text(f'''#!/bin/bash
echo start >> "{root}/starts"
exit 78
''')
    evaluator = root / "search/evaluator/pool.sh"
    evaluator.parent.mkdir()
    evaluator.write_text(f'''#!/bin/bash
trap 'echo stopped > "{root}/evaluator-stopped"; exit 0' TERM
while :; do sleep 0.05; done
''')
    result = subprocess.run(["bash", str(root / "search/run.sh"), "laion", "--epoch-mins", "0"],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 78, result.stderr
    assert (root / "starts").read_text().splitlines() == ["start"]
    assert (root / "evaluator-stopped").exists()
    assert "stopping run without retrying" in result.stderr
    assert "-- done." in result.stdout


def test_launcher_disables_terminal_conductor_and_keeps_proposers(supervisor_repo):
    root, env = supervisor_repo
    for name in ("proposer/pool.sh", "evaluator/pool.sh"):
        path = root / "search" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('trap "exit 0" TERM\nwhile :; do sleep 0.05; done\n')
    conductor = root / "search/conductor/loop.sh"
    conductor.parent.mkdir()
    conductor.write_text(f'echo start >> "{root}/conductor-starts"\nexit 78\n')
    result = subprocess.run(["bash", str(root / "search/run.sh"), "laion", "--hours", "0.0007"],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert (root / "conductor-starts").read_text().splitlines() == ["start"]
    assert "conductor disabled after terminal provider failure" in result.stderr
    assert "budget reached" in result.stdout


def test_conductor_preserves_terminal_status_and_separate_effort(tmp_path):
    root = tmp_path
    (root / "search/conductor").mkdir(parents=True)
    for name in ("search/lib.sh", "search/conductor/loop.sh", "search/conductor/prompt.md"):
        shutil.copyfile(REPO / name, root / name)
    python = root / "python3"
    python.write_text(f'''#!{sys.executable}
import json, pathlib, sys
pathlib.Path({str(root / 'args.json')!r}).write_text(json.dumps(sys.argv[1:]))
raise SystemExit(78)
''')
    python.chmod(0o755)
    env = {**os.environ, "NEURO_PYTHON": str(python), "LAB_RUN": str(root),
           "NEURO_CONDUCTOR_MODEL": "codex:gpt-6-astra:xhigh", "NEURO_CONDUCTOR_EFFORT": "xhigh",
           "NEURO_EFFORT": "medium"}
    result = subprocess.run(["bash", str(root / "search/conductor/loop.sh")], env=env,
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 78, result.stderr
    args = json.loads((root / "args.json").read_text())
    assert args[args.index("--effort") + 1] == "xhigh"
    assert "stopping without retrying" in result.stderr


def test_slot_restart_and_pool_shutdown_remove_tool_descendants(supervisor_repo):
    root, env = supervisor_repo
    # Fake workers leave TERM-ignoring tool children behind. The first session
    # exits, forcing a restart; the replacement stays alive until pool shutdown.
    worker = root / "search/proposer/worker.sh"
    worker.write_text(f'''#!/bin/bash
exec {sys.executable} - <<'PY_WORKER'
import os, pathlib, signal, time
root = pathlib.Path(os.environ['LAB_RUN'])
first = not (root / 'first-tool').exists()
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / ('first-tool' if first else 'second-tool')).write_text(str(os.getpid()))
    while True:
        time.sleep(1)
while not (root / ('first-tool' if first else 'second-tool')).exists():
    time.sleep(0.01)
if first:
    raise SystemExit(1)
while True:
    time.sleep(1)
PY_WORKER
''')
    env.update(NEURO_FLEET="1", NEURO_DEADLINE="0")
    process = subprocess.Popen(["bash", str(root / "search/proposer/pool.sh")], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def running(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False

    try:
        deadline = time.monotonic() + 5
        while not (root / "second-tool").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (root / "second-tool").exists(), "pool did not replace the failed session"
        first = int((root / "first-tool").read_text())
        second = int((root / "second-tool").read_text())
        assert not running(first), "old tool survived the slot restart"
        assert running(second)
        process.terminate()
        process.communicate(timeout=5)
        assert process.returncode == 130
        assert not running(second), "tool survived whole-pool shutdown"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for name in ("first-tool", "second-tool"):
            if (root / name).exists():
                try:
                    os.kill(int((root / name).read_text()), 9)
                except ProcessLookupError:
                    pass
