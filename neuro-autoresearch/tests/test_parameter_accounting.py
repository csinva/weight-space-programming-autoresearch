"""Measured parameter counts survive isolated execution, queue storage and feedback."""
from pathlib import Path

import numpy as np
import pytest

from neuro_eval.model import IsolatedModel, _validate_model_size, check_isolation
from neuro_eval._evaluation import _error_row
from search import queue
from search.api import Broker


@pytest.fixture
def candidate(tmp_path):
    try:
        check_isolation()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    def make(body):
        path = tmp_path / 'candidate.py'
        path.write_text(body)
        return path, IsolatedModel(path)
    return make


@pytest.mark.parametrize(('body', 'expected'), [
    ('''import torch
torch.manual_seed(9)
class Model(torch.nn.Module):
 n_params=0
 def __init__(self):
  super().__init__()
  self.net=torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(12,6)),torch.nn.ReLU(),torch.nn.Linear(6,4))
  self.net.register_buffer('offset',torch.arange(3.))
  self.alias=self.net[0][0].weight
 def forward(self,X):return self.net(X)
MODEL=Model()
''', 109),
    ('''import torch
class Model:
 n_params=0
 def __call__(self,X):
  if not hasattr(self,'weights'):
   self.weights=torch.arange(48,dtype=torch.float32).reshape(12,4)
   self.alias=self.weights
  return X @ self.weights
MODEL=Model()
''', 48),
    ('''import numpy as np
import torch
W=np.arange(48,dtype=np.float32).reshape(12,4)
class Model:
 n_params=0
 weights=W
 def __call__(self,X):return X @ torch.from_numpy(self.weights)
MODEL=Model()
''', 48),
    ('''class Model:
 n_params=0
 def __call__(self,X):return X[:,:4]
MODEL=Model()
''', 0),
    ('''class Model:
 n_params=200
 def __call__(self,X):return X[:,:4]
MODEL=Model()
''', 200),
])
def test_measured_count_round_trip(candidate, run, search_result, tmp_path, body, expected):
    path, model = candidate(body)
    mid = queue.propose(str(path), proposer='p00', run=str(run))
    values = model(np.random.default_rng(4).random((8, 12), dtype=np.float32))
    assert _validate_model_size(model, values.shape[1]) == expected
    assert queue.record(mid, {**search_result, 'n_params': model.n_params}, run=str(run)) == 'ok'
    with queue.connect(str(run)) as connection:
        count, kind = connection.execute('SELECT n_params,typeof(n_params) FROM models WHERE id=?', (mid,)).fetchone()
    assert count == expected and kind == 'integer'
    broker = Broker(run, 'proposer', 'p00', tmp_path / 'unused.sock')
    assert broker.dispatch({'command': 'status', 'id': mid})['n_params'] == expected


def test_unknown_failed_count_is_not_stored_as_zero(candidate, run, tmp_path):
    from neuro_eval.datasets import get_dataset
    path, model = candidate('''class Model:
 n_params=0
 def __call__(self,X):raise RuntimeError('synthetic forward failure')
MODEL=Model()
''')
    mid = queue.propose(str(path), proposer='p00', run=str(run))
    with pytest.raises(RuntimeError) as error:
        model(np.ones((2, 12), dtype=np.float32))
    row = _error_row(model, None, get_dataset('laion', region='ventral'), error.value, 0)
    assert row['n_params'] is None
    assert queue.record(mid, row, 'error', run=str(run)) == 'error'
    with queue.connect(str(run)) as connection:
        assert connection.execute('SELECT n_params FROM models WHERE id=?', (mid,)).fetchone()[0] is None
    broker = Broker(run, 'proposer', 'p00', tmp_path / 'unused.sock')
    assert broker.dispatch({'command': 'status', 'id': mid})['n_params'] is None


def test_timeout_without_metadata_clears_provisional_count(run):
    path = run / 'proposers/p00/model.py'
    path.write_text('class Model:\n n_params=100\nMODEL=Model()\n')
    mid = queue.propose(str(path), run=str(run))
    queue.record(mid, {}, 'error', run=str(run))
    with queue.connect(str(run)) as connection:
        assert connection.execute('SELECT n_params FROM models WHERE id=?', (mid,)).fetchone()[0] is None


def test_nested_registered_weights_enforce_cap(candidate, monkeypatch):
    path, model = candidate('''import torch
torch.manual_seed(9)
class Model(torch.nn.Module):
 n_params=0
 def __init__(self):
  super().__init__()
  self.net=torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(12,6)),torch.nn.Linear(6,4))
 def forward(self,X):return self.net(X)
MODEL=Model()
''')
    values = model(np.random.default_rng(9).random((8,12), dtype=np.float32))
    monkeypatch.setenv('NEURO_MAX_PARAMS', '100')
    assert model.n_params == 106
    with pytest.raises(ValueError, match='exceeds'):
        _validate_model_size(model, values.shape[1])
