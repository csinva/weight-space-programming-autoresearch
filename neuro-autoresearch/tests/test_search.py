"""Search workflows: restricted feedback, immutable repair, and valid rewards."""
import hashlib
import json
from pathlib import Path

import pytest

from search import queue, worker
from search.api import Broker, request
from search.run import scaffold_run


def test_socket_feedback_submission_lineage_and_roles(run, tmp_path, monkeypatch, search_result):
    socket_path = tmp_path / 'api.sock'
    monkeypatch.setenv('NEURO_BROKER_SOCKET', str(socket_path))
    with Broker(run, 'proposer', 'p00', socket_path):
        parent = request({'command': 'propose-model', 'source': '# parent\n', 'idea': 'parent'})
        assert queue.record(parent, {**search_result, "test_corr": .1,
                                     "subject_scores": {s: .1 for s in search_result["subjects"]}}, run=str(run)) == "ok"
        child = request({'command': 'propose-model', 'source': '# child\n', 'parent_id': parent})
        assert queue.record(child, search_result, run=str(run)) == "ok"
        row = request({'command': 'status', 'id': child})
        assert row['parent_id'] == parent
        assert row['parent_rsa_delta'] == pytest.approx(.1)
        assert row['search_rsa'] == .2
        assert row['rsa_diagnostics']['primary_rsa'] == .2
        assert row['model_sha256']
        assert 'path' not in row and 'evaluation_metadata' not in row
        for payload in ({'command': 'query-db', 'sql': 'SELECT * FROM models'},
                        {'command': 'reset-island', 'proposer': 'p01', 'direction': 'x'},
                        {'command': 'source', 'id': '../../etc/passwd'},
                        {'command': 'history', 'limit': 101}):
            with pytest.raises(RuntimeError):
                request(payload)
    with Broker(run, 'proposer', 'p01', socket_path):
        assert len(request({'command': 'history'})) == 2
        assert request({'command': 'source', 'id': child})['code'] == '# child\n'
        with monkeypatch.context() as scope:
            scope.setenv('VANILLA_AUTORESEARCH', '1')
            assert request({'command': 'history'}) == []
            with pytest.raises(RuntimeError):
                request({'command': 'source', 'id': child})
    with Broker(run, 'coordinator', None, socket_path):
        request({'command': 'set-direction', 'proposer': 'p00', 'direction': 'Explore'})
        assert request({'command': 'direction'})['p00'] == 'Explore\n'
        request({'command': 'reset-island', 'proposer': 'p00', 'direction': 'Start afresh'})
        assert (run / 'state' / 'p00' / 'reset_at').exists()
        assert not (run / 'proposers' / 'p00' / 'reset_at').exists()


def test_error_feedback_never_returns_raw_evaluator_text(run, tmp_path, monkeypatch):
    socket_path = tmp_path / 'api.sock'
    monkeypatch.setenv('NEURO_BROKER_SOCKET', str(socket_path))
    with Broker(run, 'proposer', 'p00', socket_path):
        mid = request({'command': 'propose-model', 'source': '# failure\n'})
        queue.record(mid, {'status': 'error: timeout /private/path target=[SECRET]',
                           'candidate_error': 'x' * 2000 + ' invalid feature shape'},
                   status='error', run=str(run))
        row = request({'command': 'status', 'id': mid})
        assert row['error_reason'] == 'Evaluation timed out'
        assert 'SECRET' not in json.dumps(row) and '/private/path' not in json.dumps(row)
        assert len(row['candidate_error']) == 1500
        assert row['candidate_error'].endswith('invalid feature shape')


def test_repair_creates_child_without_replacing_submission(run, tmp_path, monkeypatch):
    normal_ancestors = 3
    monkeypatch.setenv('LAB_RUN', str(run))
    monkeypatch.setenv('NEURO_DATASET', 'laion')
    monkeypatch.delenv('LAB_PROPOSER', raising=False)
    model = tmp_path / 'model.py'
    original = b'MODEL = None\n'
    model.write_bytes(original)
    parent = None
    for _ in range(normal_ancestors):
        parent = queue.propose(str(model), parent_id=parent)
        queue.record(parent, status="error")
    mid = queue.propose(str(model), parent_id=parent)
    row = queue.get_next_model()
    monkeypatch.setattr(worker, 'score_guarded', lambda *args: {
        'status': 'error: test', 'log': 'private scorer trace',
        'repair_log': 'candidate exception', 'candidate_error': 'candidate exception'})

    def repair(path, log, *args):
        assert Path(log).read_text() == 'candidate exception'
        Path(path).write_text('MODEL = 1\n')
    monkeypatch.setattr(worker, 'repair', repair)
    logs = run / 'eval'
    worker.process_model(row, str(logs), 'laion', 10, 1, 'fake', 'low')
    with queue.connect() as con:
        rows = [dict(r) for r in con.execute('SELECT * FROM models ORDER BY rowid')]
    assert len(rows) == normal_ancestors + 2
    rows = rows[-2:]
    assert rows[0]['id'] == mid and rows[0]['status'] == 'error'
    assert json.loads(rows[0]['evaluation_metadata'])['candidate_error'] == 'candidate exception'
    assert rows[0]['model_sha256'] == hashlib.sha256(original).hexdigest()
    assert Path(rows[0]['path']).read_bytes() == original
    assert rows[1]['parent_id'] == mid and rows[1]['status'] == 'queued'
    assert Path(rows[1]['path']).read_text() == 'MODEL = 1\n'
    assert rows[0]['repair_depth'] == 0 and rows[1]['repair_depth'] == 1
    monkeypatch.setattr(worker, 'repair', lambda *args: pytest.fail('repair budget exceeded'))
    worker.process_model(queue.get_next_model(), str(logs), 'laion', 10, 1, 'fake', 'low')


def test_reset_does_not_follow_agent_symlinks(tmp_path, monkeypatch):
    monkeypatch.delenv('LAB_PROPOSER', raising=False)
    island = tmp_path / 'proposers/p00'
    island.mkdir(parents=True)
    secret = tmp_path / 'private'
    secret.write_text('protected')
    for name in ('reset_at', 'reset_seen', 'direction.txt', 'model.py', 'scratch'):
        (island / name).symlink_to(secret)
    queue.reset_island('p00', 'new direction', str(tmp_path))
    assert secret.read_text() == 'protected'
    assert (tmp_path / 'state/p00/reset_at').is_file()
    assert (tmp_path / 'state/p00/direction.txt').read_text() == 'new direction\n'


def test_repair_cannot_redirect_host_read(tmp_path):
    private = tmp_path / 'private'
    private.write_text('neural data')
    candidate = tmp_path / 'model.py'
    candidate.symlink_to(private)
    with pytest.raises(OSError):
        worker._read_repair(candidate)


@pytest.mark.parametrize("change", ["settings", "data"])
def test_resume_preserves_identity_before_writing(run, clean_environment, monkeypatch, change):
    root = str(run.parents[1])
    before = (run / "env.json").read_bytes()
    assert Path(scaffold_run("laion", "2026-10-21", root, 2)) == run
    if change == "settings":
        monkeypatch.setenv("NEURO_IMSIZE", "256")
    else:
        clean_environment.write_bytes(b"different responses")
    with pytest.raises(ValueError, match="changed evaluation contract"):
        scaffold_run("laion", "2026-10-21", root, 2)
    assert (run / "env.json").read_bytes() == before


@pytest.mark.parametrize("invalid", [
    {"metric": "encoding", "stage": "final"},
    {"subjects": ["sub-01"], "subject_scores": {"sub-01": 0.9}},
])
def test_invalid_reward_cannot_become_champion(run, search_result, invalid):
    model_id = queue.propose(str(run / "proposers/p00/model.py"), run=str(run))
    assert queue.record(model_id, {**search_result, **invalid}, run=str(run)) == "error"
    assert queue.champion(str(run)) is None
    with queue.connect(str(run)) as connection:
        row = connection.execute("SELECT test_corr FROM models WHERE id=?", (model_id,)).fetchone()
    assert row[0] is None
