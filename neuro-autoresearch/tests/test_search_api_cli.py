"""CLI feedback is readable without filtering and JSON remains machine-readable."""
import json

import pytest

from search import api


@pytest.mark.parametrize('command', ['status', 'wait-for-score'])
def test_readable_feedback_and_explicit_json(command, monkeypatch, capsys):
    row = {
        'id': 'abc12', 'status': 'ok', 'search_score': .23456789,
        'parent_id': 'abc11', 'parent_score_delta': -.01234567,
        'subregion_scores': {
            'V1': {'status': 'ok', 'score': .12345678, 'parent_score_delta': .02345678},
            'V2': {'status': 'unavailable', 'score': None},
        },
        'seconds': 12.34, 'n_params': 123, 'metric': 'rsa', 'region': 'early',
        'dataset': 'laion', 'proposer': 'p00',
    }
    requests = []
    def request(payload):
        requests.append(dict(payload))
        return row
    monkeypatch.setattr(api, 'request', request)
    api.main([command, 'abc12'])
    output = capsys.readouterr().out
    assert output.splitlines()[0] == 'Pooled search score: 0.234568 | parent delta: -0.012346'
    assert 'ROI V1: score 0.123457 | parent delta: +0.023457' in output
    assert 'ROI V2: score unavailable | parent delta: unavailable' in output
    assert 'Model abc12 | status ok | parent abc11' in output
    assert 'runtime 12.3s' in output
    api.main([command, 'abc12', '--json'])
    assert json.loads(capsys.readouterr().out) == row
    assert all('json' not in payload for payload in requests)
    api.main([command, 'abc12', '--details'])
    assert json.loads(capsys.readouterr().out) == row
    assert requests[-1]['details'] is True


def test_wait_polls_before_displaying_terminal_feedback(monkeypatch, capsys):
    rows = iter([{'status': 'queued'}, {'status': 'running'},
                 {'id': 'abc12', 'status': 'error', 'error_reason': 'Evaluation timed out'}])
    monkeypatch.setattr(api, 'request', lambda payload: next(rows))
    sleeps = []
    monkeypatch.setattr(api.time, 'sleep', sleeps.append)
    api.main(['wait-for-score', 'abc12'])
    output = capsys.readouterr().out
    assert sleeps == [5, 5]
    assert output.startswith('Failure: Evaluation timed out\n')
    assert 'Pooled search score' not in output
    assert '--details' in output
    assert 'ROI ' not in output
    assert 'queued' not in output and 'running' not in output


@pytest.mark.parametrize(('status', 'trace', 'reason'), [
    ('Isolated candidate failed', 'torch.OutOfMemoryError: CUDA out of memory', 'CUDA out of memory'),
    ('Isolated candidate failed: CUDA out of memory', '', 'CUDA out of memory'),
    ('Isolated candidate failed', 'memory allocation failed with OOM on device', 'CUDA out of memory'),
    ('Scientific provenance mismatch', '', 'Evaluation failed'),
    ('Invalid candidate parameter count', '', 'Parameter contract failed'),
    ('Feature shape (2, 10000)', '', 'Feature shape or width contract failed'),
    ('Feature array contains NaN or infinite values', '', 'Nonfinite model features or score'),
    ('non-finite score', '', 'Nonfinite model features or score'),
    ('Representation shape depends on batch composition', '', 'Per-image independence check failed'),
    ('TimeoutExpired: timed out', '', 'Evaluation timed out'),
    ('bubblewrap namespace failed', '', 'Isolated candidate execution failed'),
])
def test_failure_classification_and_private_evaluator_text(status, trace, reason):
    row = {'id': 'abc12', 'status': 'error', 'dataset': 'laion', 'region': 'ventral',
           'metric': 'rsa', 'evaluation_metadata': {
               'status': status + ' /private/host/path secret=HOST_SECRET',
               'candidate_error': trace,
               'log': 'HOST_SECRET /private/host/path',
           }}
    compact = api.public_row(row)
    assert compact['error_reason'].startswith(reason)
    assert 'candidate_error' not in compact
    assert 'subregion_scores' not in compact
    for details in (False, True):
        output = json.dumps(api.public_row(row, details=details))
        assert 'HOST_SECRET' not in output
        assert '/private/host/path' not in output


def test_candidate_trace_only_on_request_and_bounded():
    trace = 'x' * 2000 + '\nFile "/some/path/model.py", line 7\nRuntimeError: invalid features'
    row = {'status': 'error', 'evaluation_metadata': {'candidate_error': trace}}
    assert 'candidate_error' not in api.public_row(row)
    detailed = api.public_row(row, details=True)['candidate_error']
    assert len(detailed) <= 1500
    assert '/some/path' not in detailed
    assert 'line 7' in detailed
    assert detailed.endswith('RuntimeError: invalid features')


def test_pending_feedback_shows_identity_without_unavailable_scores(monkeypatch, capsys):
    monkeypatch.setattr(api, 'request', lambda payload: {
        'id': 'abc12', 'status': 'queued', 'search_score': None,
        'subregion_scores': {'V1': {'score': None, 'status': 'unavailable'}},
    })
    api.main(['status', 'abc12'])
    assert capsys.readouterr().out == 'Model abc12 | status queued\n'
