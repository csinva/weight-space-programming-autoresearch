"""Launcher validation, process cleanup and evaluator ownership without live providers."""
from collections import Counter
import io
from pathlib import Path
import subprocess
import sys

import pytest

from search import agents, queue, run, worker


@pytest.mark.parametrize('args', [
    ['--hours', 'nan'],
    ['--date', 'bad'],
    ['nsd'],
    ['tvsd'],
])
def test_invalid_launch_never_scaffolds(args, monkeypatch):
    monkeypatch.setattr(run, 'scaffold_run', lambda *a, **kw: pytest.fail('created run'))
    with pytest.raises(SystemExit) as result:
        run.main(['--region', 'ventral', *args])
    assert result.value.code == 2


def test_preflight_checks_sources_and_providers_without_creating_run(tmp_path, monkeypatch):
    from neuro_eval import datasets, model
    from neuro_eval.benchmarks import protocol_for
    from search import providers

    source = tmp_path / 'images.h5'
    source.write_bytes(b'data')
    subjects = ','.join(protocol_for('laion').search_subjects)
    monkeypatch.setattr(run, 'evaluation_environment', lambda d: {'NEURO_SUBJECT': subjects})
    monkeypatch.setattr(datasets, 'dataset_source_paths', lambda d: {'images': str(source)})
    monkeypatch.setattr(model, 'check_isolation', lambda: None)
    for name in run.PROVIDER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    observed = []
    monkeypatch.setattr(providers, 'preflight', lambda agents: observed.extend(agents))
    monkeypatch.setattr(run, 'scaffold_run', lambda *a, **kw: pytest.fail('created run'))
    assert run.main(['laion', '--region', 'ventral', '--preflight']) == 0
    # Two Codex proposers plus the coordinator and optional repair agent.
    assert Counter(agent.provider for agent in observed) == {'codex': 4, 'agy': 2}

    source.write_bytes(b'')
    observed.clear()
    with pytest.raises(SystemExit) as result:
        run.main(['laion', '--region', 'ventral', '--preflight'])
    assert result.value.code == 2
    assert observed == []


def simulation(tmp_path, monkeypatch, outcomes, *, evaluators=1, coordinator=0, hours=.0003,
               coordinator_seconds=0, coordinator_starts=None):
    monkeypatch.setenv('NEURO_REGION', 'ventral')
    monkeypatch.setenv('NEURO_METRIC', 'rsa')
    events = []
    counts = Counter()
    clock = [0.0]

    class PromptInput(io.StringIO):
        def write(self, prompt):
            assert "remaining session budget at launch=" in prompt
            assert "minutes" in prompt
            return super().write(prompt)

    class Child:
        def __init__(self, argv, **kwargs):
            if 'search.worker' in argv:
                self.name = 'eval'
                assert '-u' in argv
            elif argv[argv.index('--role') + 1] == 'coordinator':
                self.name = 'coordinator'
            else:
                self.name = Path(argv[argv.index('--cwd') + 1]).name
            self.started = clock[0]
            if self.name == 'coordinator' and coordinator_starts is not None:
                coordinator_starts.append(self.started)
            self.pid = 100 + len(events)
            self.stdin = PromptInput()
            results = outcomes.get(self.name, [None])
            self.code = results[min(counts[self.name], len(results) - 1)]
            counts[self.name] += 1
            events.append(('start', self.name))

        def poll(self):
            if self.name == 'coordinator' and clock[0] - self.started < coordinator_seconds:
                return None
            return self.code

    monkeypatch.setattr(agents.subprocess, 'Popen', Child)
    monkeypatch.setattr(agents, 'stop_process', lambda child, grace=0: events.append(('stop', child.name)))
    monkeypatch.setattr(queue, 'requeue_orphans', lambda run: events.append(('recover', None)))
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    monkeypatch.setattr('time.sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setenv('NEURO_COORDINATOR_MODEL', 'codex:gpt-6-astra:xhigh')
    monkeypatch.setenv('NEURO_COORDINATOR_EFFORT', 'xhigh')
    result = agents.supervise(str(tmp_path), 'laion', ['codex:astra', 'codex:astra'], hours, coordinator, evaluators)
    return result, events, counts


def test_terminal_slot_stays_disabled_while_healthy_restarts(tmp_path, monkeypatch):
    code, events, counts = simulation(tmp_path, monkeypatch, {'p00': [78], 'p01': [0, None], 'coordinator': [78]}, coordinator=1, hours=.04)
    assert code == 0
    assert counts['p00'] == counts['coordinator'] == 1
    assert counts['p01'] == 2
    assert events[0] == ('recover', None)
    assert events[-1] == ('recover', None)


def test_all_terminal_slots_stop_run(tmp_path, monkeypatch):
    code, events, counts = simulation(tmp_path, monkeypatch, {'p00': [78], 'p01': [78]})
    assert code == 78
    assert counts['p00'] == counts['p01'] == 1
    assert ('stop', 'eval') in events


def test_evaluator_peers_stop_before_recovery(tmp_path, monkeypatch):
    code, events, counts = simulation(tmp_path, monkeypatch, {'eval': [70, None]}, evaluators=2)
    recoveries = [i for i, event in enumerate(events) if event[0] == 'recover']
    assert len(recoveries) == 3
    restart = recoveries[1]
    assert events[restart-2:restart] == [('stop', 'eval'), ('stop', 'eval')]
    assert events[restart+1:restart+3] == [('start', 'eval'), ('start', 'eval')]


def queue_rows(run, states):
    queue.init(str(run))
    with queue.connect(str(run)) as connection:
        for i, state in enumerate(states):
            connection.execute("INSERT INTO models (id,status,created_at) VALUES (?,?,?)",
                               (f"model{i}", state, f"2026-10-21T00:00:0{i}"))


def test_new_worker_does_not_requeue_peers_live_claim(run, monkeypatch):
    tmp_path = run
    queue_rows(tmp_path, ["running", "queued"])
    monkeypatch.setenv("LAB_RUN", str(tmp_path))
    monkeypatch.setenv("NEURO_METRIC", "rsa")
    monkeypatch.setenv("NEURO_STAGE", "search")
    claimed = []

    def stop_after_claim(row, *args):
        assert args[3] == 0  # default repair ownership belongs to the proposer
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
    tool = f"""
import signal
import time
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open({str(heartbeat)!r}, 'a', buffering=1) as output:
    output.write('tick\\n')
    Path({str(ready)!r}).touch()
    while True:
        output.write('tick\\n')
        time.sleep(.02)
"""
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


def test_failed_score_reports_completion_without_automatic_repair(tmp_path, monkeypatch, capsys):
    import hashlib
    source = tmp_path / "model.py"
    source.write_text("MODEL = None\n")
    row = {"id": "failed", "path": str(source), "proposer": "p00",
           "model_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    monkeypatch.setattr(worker, "score_guarded", lambda *a: {
        "status": "error", "repair_log": "candidate failed", "log": "candidate failed"})
    monkeypatch.setattr(queue, "record", lambda *a: "error")
    monkeypatch.setattr(worker, "repair", lambda *a: pytest.fail("automatic repair started"))
    worker.process_model(row, str(tmp_path), "laion", 420, 0, "codex:astra", "medium")
    output = capsys.readouterr().out
    assert "failed: started proposer=p00" in output
    assert "failed: completed status=error" in output
    assert "+00:00" in output and "wall_seconds=" in output


def test_clean_proposers_do_not_restart_near_deadline(tmp_path, monkeypatch, capsys):
    import time
    code, events, counts = simulation(tmp_path, monkeypatch, {'p00': [0], 'p01': [0]})
    assert code == 0
    assert counts['p00'] == counts['p01'] == counts['eval'] == 1
    # Clean exits do not stop the evaluator or shorten the search deadline.
    assert time.monotonic() >= .0003 * 3600
    assert events[-2:] == [('stop', 'eval'), ('recover', None)]
    output = capsys.readouterr().out
    assert output.count('no restart in final two minutes') == 2
    assert 'terminal provider failure' not in output


def test_transient_failure_still_restarts_near_deadline(tmp_path, monkeypatch):
    code, _, counts = simulation(tmp_path, monkeypatch, {'p00': [1, None]})
    assert code == 0
    assert counts['p00'] == 2


def test_supervisor_refreshes_usage_on_resume_and_shutdown(tmp_path, monkeypatch):
    from search import costs
    refreshed = []
    monkeypatch.setattr(costs, 'summarize_run', lambda path: refreshed.append(Path(path)))
    code, _, _ = simulation(tmp_path, monkeypatch, {}, hours=.002)
    assert code == 0
    assert refreshed == [tmp_path, tmp_path, tmp_path]


@pytest.mark.parametrize('args', [[], ['--dataset', 'laion'],
                                  ['--region', 'joint'],
                                  ['--region', 'early', '--metric', 'unknown']])
def test_launcher_requires_supported_explicit_configuration(args, monkeypatch):
    monkeypatch.setattr(run, 'scaffold_run', lambda *a, **kw: pytest.fail('created run'))
    with pytest.raises(SystemExit) as result:
        run.main(args)
    assert result.value.code == 2


@pytest.mark.parametrize(('duration', 'expected'), [
    (10, [0, 30, 60, 90, 120]),
    (40, [0, 60, 120]),
])
def test_conductor_uses_fixed_start_slots_without_overlap(tmp_path, monkeypatch, duration, expected):
    starts = []
    code, events, counts = simulation(
        tmp_path, monkeypatch, {'coordinator': [0]}, coordinator=.5, hours=.04,
        coordinator_seconds=duration, coordinator_starts=starts)
    assert code == 0
    assert starts == expected
    active = False
    for action, name in events:
        if name == 'coordinator':
            if action == 'start':
                assert not active
                active = True
            elif action == 'stop':
                assert active
                active = False
    assert not active
