"""Archive identity survives repairs and attempted state-file redirection."""
import hashlib
import json
from pathlib import Path

import hub
import pytest
from harness.protocols import protocol_for
from search.evaluator import worker


@pytest.mark.parametrize("normal_ancestors", [0, 3])
def test_repair_creates_child_without_replacing_submission(tmp_path, monkeypatch, normal_ancestors):
    protocol = protocol_for('laion')
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'env.json').write_text(json.dumps({
        'NEURO_DATASET': 'laion', 'NEURO_METRIC': 'rsa', 'NEURO_STAGE': 'search',
        'NEURO_SUBJECT': ','.join(protocol.search_subjects), 'protocol': protocol.to_dict()}))
    monkeypatch.setenv('LAB_RUN', str(run))
    monkeypatch.setenv('NEURO_DATASET', 'laion')
    monkeypatch.delenv('LAB_PROPOSER', raising=False)
    model = tmp_path / 'model.py'
    original = b'MODEL = None\n'
    model.write_bytes(original)
    parent = None
    for _ in range(normal_ancestors):
        parent = hub.propose(str(model), parent_id=parent)
        hub.record(parent, status="error")
    mid = hub.propose(str(model), parent_id=parent)
    row = hub.get_next_model()
    monkeypatch.setattr(worker, 'score_guarded', lambda *args: {'status': 'error: test', 'log': 'private scorer trace', 'repair_log': 'candidate exception', 'candidate_error': 'candidate exception'})
    def repair(path, log, *args):
        assert Path(log).read_text() == 'candidate exception'
        Path(path).write_text('MODEL = 1\n')
    monkeypatch.setattr(worker, 'repair', repair)
    logs = run / 'eval'
    logs.mkdir()
    worker.process_model(row, str(logs), 'laion', 10, 1, 'fake', 'low')
    with hub.connect() as con:
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
    worker.process_model(hub.get_next_model(), str(logs), 'laion', 10, 1, 'fake', 'low')


def test_reset_does_not_follow_agent_symlinks(tmp_path, monkeypatch):
    monkeypatch.delenv('LAB_PROPOSER', raising=False)
    island = tmp_path / 'proposers/p00'
    island.mkdir(parents=True)
    secret = tmp_path / 'private'
    secret.write_text('protected')
    for name in ('reset_at', 'reset_seen', 'direction.txt', 'model.py', 'scratch'):
        (island / name).symlink_to(secret)
    hub.reset_island('p00', 'new direction', str(tmp_path))
    assert secret.read_text() == 'protected'
    assert (tmp_path / 'state/p00/reset_at').is_file()
    assert (tmp_path / 'state/p00/direction.txt').read_text() == 'new direction\n'


def test_repair_cannot_redirect_host_read(tmp_path):
    import pytest
    private = tmp_path / 'private'
    private.write_text('neural data')
    candidate = tmp_path / 'model.py'
    candidate.symlink_to(private)
    with pytest.raises(OSError):
        worker._read_repair(candidate)
