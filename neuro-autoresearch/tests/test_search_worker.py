"""Queue recovery must finish before any evaluator can hold a live claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

from search import queue
from search import worker

REPO = Path(__file__).resolve().parents[1]


def test_exit_cleanup_survives_already_stopped_process_group():
    from search.agents import stop_process
    child = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    child.wait()
    stop_process(child, grace=0)
    stop_process(child, grace=0)


def queue_rows(run, states):
    queue.init(str(run))
    with queue.connect(str(run)) as connection:
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
    with queue.connect(str(tmp_path)) as connection:
        assert connection.execute("SELECT status FROM models WHERE id='model0'").fetchone()[0] == "running"




def test_cleanup_kills_tools_after_leader_exits(tmp_path):
    """A completed provider must not leave a TERM-ignoring tool beside its replacement."""
    import time
    from search.agents import stop_process
    ready = tmp_path / 'ready'
    heartbeat = tmp_path / 'heartbeat'
    tool = ("import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(ready)!r}).write_text(str(os.getpid())); "
            f"f=open({str(heartbeat)!r},'a',buffering=1); "
            "exec('while True:\\n f.write(\"tick\\\\n\"); time.sleep(0.02)')")
    leader = subprocess.Popen([sys.executable, '-c',
        'import subprocess,sys; subprocess.Popen([sys.executable,"-c",sys.argv[1]])', tool],
        start_new_session=True)
    try:
        leader.wait(timeout=5)
        until = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < until:
            time.sleep(.02)
        assert ready.exists()
        stop_process(leader, grace=.05)
        time.sleep(.05)
        size = heartbeat.stat().st_size
        time.sleep(.1)
        assert heartbeat.stat().st_size == size
    finally:
        stop_process(leader, grace=0)
