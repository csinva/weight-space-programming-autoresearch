"""Queue recovery must finish before any evaluator can hold a live claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

import hub
from search.evaluator import worker

REPO = Path(__file__).resolve().parents[1]


def test_exit_cleanup_survives_already_stopped_process_group():
    script = '''
set -euo pipefail
source "$1/search/lib.sh"
set -m
sleep 30 & child=$!
shutdown() {
    term_group "$child"
    wait "$child" 2>/dev/null || true
    # The normal TERM -> KILL path must tolerate a child that has exited.
    kill_group "$child"
    term_group "$child"
    term "$child"
    term_group 0
    kill_group 0
    term 0
    if alive 0; then exit 99; fi
    echo cleanup-complete
}
install_shutdown_trap shutdown
exit 0
'''
    result = subprocess.run(["bash", "-c", script, "test", str(REPO)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cleanup-complete"


def queue_rows(run, states):
    hub.init(str(run))
    with hub.connect(str(run)) as connection:
        for i, state in enumerate(states):
            connection.execute("INSERT INTO models (id,status,created_at) VALUES (?,?,?)",
                               (f"model{i}", state, f"2026-10-21T00:00:0{i}"))


def test_new_worker_does_not_requeue_peers_live_claim(tmp_path, monkeypatch):
    queue_rows(tmp_path, ["running", "queued"])
    monkeypatch.setenv("LAB_RUN", str(tmp_path))
    monkeypatch.setenv("NEURO_METRIC", "rsa")
    monkeypatch.setenv("NEURO_STAGE", "search")
    claimed = []

    def stop_after_claim(row, *args):
        claimed.append(row["id"])
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "process_model", stop_after_claim)
    worker.main(["--dataset", "laion"])
    assert claimed == ["model1"]
    with hub.connect(str(tmp_path)) as connection:
        assert connection.execute("SELECT status FROM models WHERE id='model0'").fetchone()[0] == "running"


def test_pool_recovers_once_before_starting_workers(tmp_path):
    count = 2
    queue_rows(tmp_path, ["running"] * count)
    events = tmp_path / "events.jsonl"
    bindir = tmp_path / "selected python"
    bindir.mkdir()
    fake_python = bindir / "python3"
    fake_python.write_text(f'''#!{sys.executable}
import json, os, pathlib, sqlite3, subprocess, sys, time
log=pathlib.Path(os.environ["TEST_POOL_EVENTS"])
def emit(event):
    with log.open("a") as file:
        file.write(json.dumps(event)+"\\n")
if sys.argv[1] == "-c":
    emit({{"event":"recovery"}})
    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))
assert sys.argv[1].endswith("search/evaluator/worker.py"), sys.argv
with sqlite3.connect(pathlib.Path(os.environ["LAB_RUN"])/"search.db", timeout=5) as connection:
    row=connection.execute("UPDATE models SET status='running' WHERE id=(SELECT id FROM models WHERE status='queued' ORDER BY id LIMIT 1) RETURNING id").fetchone()
emit({{"event":"worker", "claim": row[0] if row else None}})
# Keep the first worker alive until every peer has recorded its initial claim.
end=time.monotonic()+5
while time.monotonic()<end:
    if sum(json.loads(line)["event"]=="worker" for line in log.read_text().splitlines()) >= int(os.environ["NEURO_EVALUATORS"]):
        break
    time.sleep(0.02)
''')
    fake_python.chmod(0o755)
    env = {**os.environ, "LAB_RUN": str(tmp_path), "NEURO_DATASET": "laion", "NEURO_PYTHON": str(fake_python),
           "NEURO_EVALUATORS": str(count), "TEST_POOL_EVENTS": str(events)}
    result = subprocess.run(["bash", str(REPO / "search/evaluator/pool.sh")], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert records[0] == {"event": "recovery"}
    assert sum(record["event"] == "recovery" for record in records) == 1
    claims = [record["claim"] for record in records if record["event"] == "worker"]
    assert sorted(claims) == [f"model{i}" for i in range(count)]
