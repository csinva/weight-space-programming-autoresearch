"""Broker contract tests use synthetic records and never invoke providers or data."""
import json

import pytest

import hub
from harness.protocols import SOURCE_REVISION, protocol_for
from search.broker import Broker
from search.client import request


@pytest.fixture
def broker_run(tmp_path, monkeypatch):
    for key in ('LAB_PROPOSER', 'NEURO_DATASET', 'VANILLA_AUTORESEARCH'):
        monkeypatch.delenv(key, raising=False)
    run = tmp_path / 'run'
    hub.init(str(run))
    protocol = protocol_for('laion')
    (run / 'env.json').write_text(json.dumps({
        'NEURO_DATASET': 'laion', 'NEURO_METRIC': 'rsa', 'NEURO_STAGE': 'search',
        'NEURO_SUBJECT': ','.join(protocol.search_subjects), 'protocol': protocol.to_dict()}))
    for name in ('p00', 'p01'):
        (run / 'proposers' / name).mkdir(parents=True)
    return run


def score(run, mid, value):
    protocol = protocol_for('laion')
    result = {'status': 'ok', 'test_corr': value, 'metric': 'rsa', 'stage': 'search',
              'protocol_version': protocol.version, 'dataset': 'laion',
              'source_revision': SOURCE_REVISION, 'subjects': list(protocol.search_subjects),
              'subject_scores': {subject: value for subject in protocol.search_subjects},
              'rsa_diagnostics': {'schema_version': 1, 'primary_rsa': value}}
    assert hub.record(mid, result, run=str(run)) == 'ok'


def test_socket_feedback_submission_lineage_and_roles(broker_run, tmp_path, monkeypatch):
    socket_path = tmp_path / 'api.sock'
    monkeypatch.setenv('NEURO_BROKER_SOCKET', str(socket_path))
    with Broker(broker_run, 'proposer', 'p00', socket_path):
        parent = request({'command': 'propose-model', 'source': '# parent\n', 'idea': 'parent'})
        score(broker_run, parent, .1)
        child = request({'command': 'propose-model', 'source': '# child\n', 'parent_id': parent})
        score(broker_run, child, .2)
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
    with Broker(broker_run, 'proposer', 'p01', socket_path):
        assert len(request({'command': 'history'})) == 2
        assert request({'command': 'source', 'id': child})['code'] == '# child\n'
        with monkeypatch.context() as scope:
            scope.setenv('VANILLA_AUTORESEARCH', '1')
            assert request({'command': 'history'}) == []
            with pytest.raises(RuntimeError):
                request({'command': 'source', 'id': child})
    with Broker(broker_run, 'conductor', None, socket_path):
        request({'command': 'set-direction', 'proposer': 'p00', 'direction': 'Explore'})
        assert request({'command': 'direction'})['p00'] == 'Explore\n'
        request({'command': 'reset-island', 'proposer': 'p00', 'direction': 'Start afresh'})
        assert (broker_run / 'state' / 'p00' / 'reset_at').exists()
        assert not (broker_run / 'proposers' / 'p00' / 'reset_at').exists()


def test_error_feedback_never_returns_raw_evaluator_text(broker_run, tmp_path, monkeypatch):
    socket_path = tmp_path / 'api.sock'
    monkeypatch.setenv('NEURO_BROKER_SOCKET', str(socket_path))
    with Broker(broker_run, 'proposer', 'p00', socket_path):
        mid = request({'command': 'propose-model', 'source': '# failure\n'})
        hub.record(mid, {'status': 'error: timeout /private/path target=[SECRET]'},
                   status='error', run=str(broker_run))
        row = request({'command': 'status', 'id': mid})
        assert row['error_reason'] == 'Evaluation timed out'
        assert 'SECRET' not in json.dumps(row) and '/private/path' not in json.dumps(row)


def test_equivalence_uses_frozen_trusted_inputs(broker_run, tmp_path, monkeypatch):
    reference, spec = tmp_path / 'reference.py', tmp_path / 'input_spec.json'
    reference.write_text('# frozen reference\n')
    spec.write_text('{"batch_shape": [2, 12]}')
    broker = Broker(broker_run, 'simplifier', 'p00', tmp_path / 'api.sock',
                    reference=reference, input_spec=spec)
    reference.write_text('# tampered after startup\n')

    def check(candidate, frozen, specification):
        from pathlib import Path
        assert Path(candidate).read_text() == '# candidate\n'
        assert Path(frozen).read_text() == '# frozen reference\n'
        assert json.loads(Path(specification).read_text())['batch_shape'] == [2, 12]
        return {'equivalent': False, 'error': 'private stderr', 'trace': '/private/path'}

    monkeypatch.setattr(hub, 'check_equivalence', check)
    monkeypatch.setenv('NEURO_BROKER_SOCKET', str(tmp_path / 'api.sock'))
    with broker:
        result = request({'command': 'check-equivalence', 'source': '# candidate\n',
                          'reference': '/etc/passwd', 'spec': '/private/path'})
        assert result == {'equivalent': False, 'reason': 'Isolated equivalence check failed'}
    with Broker(broker_run, 'proposer', 'p00', tmp_path / 'api.sock'):
        with pytest.raises(RuntimeError):
            request({'command': 'check-equivalence', 'source': '# candidate\n'})


def test_default_feedback_is_compact_and_details_are_single_model_only(broker_run, tmp_path):
    broker = Broker(broker_run, 'proposer', 'p00', tmp_path / 'api.sock')
    mid = broker.dispatch({'command': 'propose-model', 'source': '# diagnostics\n'})
    score(broker_run, mid, .2)
    subjects = list(protocol_for('laion').search_subjects)
    diagnostics = {
        'schema_version': 1, 'primary_rsa': .2,
        'subject_results': {subject: {'n_stimuli': 1252} for subject in subjects},
        'regions': {'early': {'rsa': .3, 'subjects': subjects, 'n_subjects': 5}},
        'source_groups': {'laion': {'status': 'ok', 'rsa': .1, 'subjects': subjects, 'n_subjects': 5,
            'n_stimuli_by_subject': {subject: 641 for subject in subjects},
            'regions': {'early': {'status': 'ok', 'rsa': .15, 'subjects': subjects, 'n_subjects': 5}},
            'subject_results': {subject: {'rsa': .1} for subject in subjects}}}}
    with hub.connect(str(broker_run)) as con:
        con.execute('UPDATE models SET evaluation_metadata=? WHERE id=?',
                    (json.dumps({'rsa_diagnostics': diagnostics}), mid))
    for command in ('status', 'history', 'top-models'):
        row = broker.dispatch({'command': command, 'id': mid})
        row = row[0] if isinstance(row, list) else row
        compact = row['rsa_diagnostics']
        assert 'subject_results' not in compact
        assert compact['primary_rsa'] == .2
        assert compact['regions']['early']['subjects'] == subjects
        assert compact['source_groups']['laion'] == {
            'status': 'ok', 'rsa': .1, 'n_subjects': 5, 'n_stimuli': 641,
            'regions': {'early': {'status': 'ok', 'rsa': .15, 'n_subjects': 5}}}
    detailed = broker.dispatch({'command': 'status', 'id': mid, 'details': True})
    assert detailed['rsa_diagnostics'] == diagnostics
    assert 'evaluation_metadata' not in detailed and 'path' not in detailed
    with pytest.raises(ValueError, match='one model ID'):
        broker.dispatch({'command': 'history', 'details': True})


@pytest.mark.parametrize('command', ['status', 'wait-for-score'])
def test_client_details_flag_requests_single_model(command, monkeypatch, capsys):
    from search import client
    calls = []

    def fake_request(payload):
        calls.append(dict(payload))
        return {'status': 'ok'}

    monkeypatch.setattr(client, 'request', fake_request)
    client.main([command, 'abc12', '--details'])
    assert calls == [{'command': 'status', 'id': 'abc12', 'details': True}]
    assert json.loads(capsys.readouterr().out) == {'status': 'ok'}


def test_candidate_error_is_bounded_and_separate_from_private_scorer_status():
    from search.broker import public_row

    row = public_row({'id': 'abc12', 'status': 'error', 'repair_depth': 1,
                      'evaluation_metadata': json.dumps({
                          'status': 'error: scorer traceback contains PRIVATE_TARGET',
                          'candidate_error': 'x' * 2000 + ' invalid feature shape'})})
    assert len(row['candidate_error']) == 1500
    assert row['candidate_error'].endswith('invalid feature shape')
    assert row['repair_depth'] == 1
    assert 'PRIVATE_TARGET' not in json.dumps(row)
