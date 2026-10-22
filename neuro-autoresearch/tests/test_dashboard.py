"""The monitoring UI discovers search runs and reads their trusted state only."""
import sqlite3

import pytest

from dashboard import _db


def test_discovers_search_runs_without_final_results(tmp_path, monkeypatch):
    runs = tmp_path / "search_runs"
    run = runs / "laion__2026-10-22"
    run.mkdir(parents=True)
    with sqlite3.connect(run / "search.db") as db:
        db.execute("CREATE TABLE models (id TEXT)")
    final = tmp_path / "final_results" / "laion__2026-10-23"
    final.mkdir(parents=True)
    (final / "search.db").touch()
    monkeypatch.setattr(_db, "RUNS_ROOT", runs)
    assert _db.list_run_dirs() == [run]
    assert _db.default_run_dir() == run
    assert _db.run_dir("laion", "2026-10-22") == run
    with _db.connect(run) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("INSERT INTO models VALUES ('abcde')")


def test_direction_comes_from_trusted_state(tmp_path):
    state = tmp_path / "state" / "p00"
    state.mkdir(parents=True)
    (state / "direction.txt").write_text("Explore spatial frequency\n")
    workspace = tmp_path / "proposers" / "p00"
    workspace.mkdir(parents=True)
    (workspace / "direction.txt").write_text("Untrusted workspace text")
    assert _db.proposer_direction(tmp_path, "p00") == "Explore spatial frequency"
    assert _db.proposer_direction(tmp_path, "p01") == ""


def test_provider_logs_drive_activity_and_log_endpoints(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    pytest.importorskip("starlette")
    from dashboard.logs import get_logs, get_logs_list

    (tmp_path / "proposers" / "p00").mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    event = {"type": "item.completed", "item": {"text": "Candidate submitted"}}
    (logs / "p00.jsonl").write_text(json.dumps(event) + "\nA provider error\n")
    (logs / "eval0.jsonl").write_text("Private evaluator traceback")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(run_dir=tmp_path)),
                              path_params={"agent_id": "p00"})
    listed = json.loads(asyncio.run(get_logs_list(request)).body)
    assert set(listed) == {"p00"}
    result = json.loads(asyncio.run(get_logs(request)).body)
    entries = result["turns"][0]["entries"]
    assert json.loads(entries[0]["content"]) == event
    assert entries[1]["content"] == "A provider error"
    assert _db.run_is_alive(tmp_path)
    assert _db.agent_activity(tmp_path, "p00", True, set())[0] == "active"
    request.path_params["agent_id"] = "eval0"
    assert json.loads(asyncio.run(get_logs(request)).body)["turns"] == []
