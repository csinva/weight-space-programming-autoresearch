"""Supervisor recovery and terminal failures use fake child processes only."""
import io
from collections import Counter
from pathlib import Path
import pytest
from search import agents, queue


def simulation(tmp_path, monkeypatch, outcomes, *, evaluators=1, coordinator=0):
    events = []
    counts = Counter()
    clock = [0.0]
    class Child:
        def __init__(self, argv, **kwargs):
            if 'search.worker' in argv:
                self.name = 'eval'
            elif argv[argv.index('--role') + 1] == 'coordinator':
                self.name = 'coordinator'
            else:
                self.name = Path(argv[argv.index('--cwd') + 1]).name
            self.pid = 100 + len(events)
            self.stdin = io.StringIO()
            self.code = outcomes.get(self.name, [None])[min(counts[self.name], len(outcomes.get(self.name, [None]))-1)]
            counts[self.name] += 1
            events.append(('start', self.name))
        def poll(self):
            return self.code
    monkeypatch.setattr(agents.subprocess, 'Popen', Child)
    monkeypatch.setattr(agents, 'stop_process', lambda child, grace=0: events.append(('stop', child.name)))
    monkeypatch.setattr(queue, 'requeue_orphans', lambda run: events.append(('recover', None)))
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    monkeypatch.setattr('time.sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setenv('NEURO_COORDINATOR_MODEL', 'codex:gpt-6-astra:xhigh')
    monkeypatch.setenv('NEURO_COORDINATOR_EFFORT', 'xhigh')
    result = agents.supervise(str(tmp_path), 'laion', ['codex:astra', 'codex:astra'], .0003, coordinator, evaluators)
    return result, events, counts


def test_terminal_slot_stays_disabled_while_healthy_restarts(tmp_path, monkeypatch):
    code, events, counts = simulation(tmp_path, monkeypatch, {'p00': [78], 'p01': [0, None]})
    assert code == 0
    assert counts['p00'] == 1 and counts['p01'] == 2
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


def test_coordinator_terminal_failure_does_not_stop_proposers(tmp_path, monkeypatch):
    code, events, counts = simulation(tmp_path, monkeypatch, {'coordinator': [78]}, coordinator=1)
    assert code == 0 and counts['coordinator'] == 1
    assert counts['p00'] == counts['p01'] == 1
