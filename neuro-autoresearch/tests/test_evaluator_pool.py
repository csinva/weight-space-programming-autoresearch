"""Queue recovery must finish before any evaluator can hold a live claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import hub
from search.evaluator import worker

REPO = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize("count", [1, 2])
def test_pool_recovers_once_before_starting_workers(tmp_path, count):
    queue_rows(tmp_path, ["running"] * count)
    events = tmp_path / "events.jsonl"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_python = bindir / "python"
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
    env = {**os.environ, "LAB_RUN": str(tmp_path), "NEURO_DATASET": "laion", "PYTHON": str(fake_python),
           "NEURO_EVALUATORS": str(count), "TEST_POOL_EVENTS": str(events)}
    result = subprocess.run(["bash", str(REPO / "search/evaluator/pool.sh")], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert records[0] == {"event": "recovery"}
    assert sum(record["event"] == "recovery" for record in records) == 1
    claims = [record["claim"] for record in records if record["event"] == "worker"]
    assert sorted(claims) == [f"model{i}" for i in range(count)]
