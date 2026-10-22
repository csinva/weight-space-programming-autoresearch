"""Synthetic OS boundary checks; never open a neural dataset."""
import shutil
import socket

import numpy as np
import pytest

from neuro_eval.model import IsolatedModel, check_isolation


@pytest.fixture
def isolated(tmp_path):
    if not shutil.which('bwrap'):
        pytest.skip('bubblewrap unavailable')
    try:
        check_isolation()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    def create(body):
        path = tmp_path / 'model.py'
        path.write_text(body)
        return IsolatedModel(path)
    return create


def test_candidate_cannot_read_host_or_connect(isolated, tmp_path):
    secret = tmp_path / 'private-neural.npy'
    secret.write_text('PRIVATE')
    # A closed port would pass even if network isolation were broken.
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        model = isolated(f'''import os, socket
class Model:
 def __call__(self, X):
  assert not os.path.exists({str(secret)!r})
  try:
   socket.create_connection({listener.getsockname()!r}, timeout=.1)
  except OSError:
   pass
  else:
   raise AssertionError('host network reachable')
  return X.flatten(1)
MODEL=Model()
''')
        images = np.ones((2, 3), dtype='float32')
        np.testing.assert_array_equal(model(images), images)


def test_batch_dependent_candidate_rejected(isolated):
    model = isolated('''class Model:
 def __call__(self, X): return X - X.mean(0)
MODEL=Model()
''')
    with pytest.raises(RuntimeError, match='batch composition'):
        model(np.arange(12, dtype='float32').reshape(4, 3))
    assert 'batch composition' in model.repair_log


def test_missing_isolation_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='refusing unsafe'):
        IsolatedModel(tmp_path / 'never-import.py')


def test_cuda_per_image_convolutions_allow_batch_roundoff(isolated):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = isolated('''import torch
import torch.nn.functional as F
class Model:
 def __init__(self):
  g=torch.Generator().manual_seed(0)
  self.w=[torch.randn(s,generator=g)*(2/(s[1]*s[2]*s[3]))**.5 for s in [(48,3,7,7),(64,48,3,3),(96,64,3,3)]]
  self.w[0]-=self.w[0].mean((1,2,3),keepdim=True)
 def __call__(self,X):
  x=X.reshape(-1,128,128,3).permute(0,3,1,2)-.5
  out=[]
  for w,grid in zip(self.w,(8,6,4)):
   x=F.max_pool2d(F.relu(F.conv2d(x,w.to(X.device),padding=w.shape[-1]//2)),2)
   f=F.adaptive_avg_pool2d(x,grid).flatten(1)
   out.append(f/(f.norm(dim=1,keepdim=True)+1e-8))
  return torch.cat(out,1)
MODEL=Model()
''')
    images = torch.rand((8, 128 * 128 * 3), device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(7))
    features = model(images)
    assert features.shape == (8, 6912)
    assert np.isfinite(features).all()


def test_large_offset_cannot_hide_batch_dependence(isolated):
    model = isolated('''class Model:
 def __call__(self, X): return 10000 + X[:, :8] + 5 * X[:, :8].mean(0, keepdim=True)
MODEL=Model()
''')
    images = np.random.default_rng(42).random((8, 12), dtype=np.float32)
    with pytest.raises(RuntimeError, match='batch composition'):
        model(images)


def test_candidate_cannot_override_host_independence_check(isolated):
    model = isolated('''import torch
# This bypassed the old checker running inside the candidate interpreter.
torch.linalg.vector_norm = lambda *args, **kwargs: torch.tensor(0.)
class Model:
 def __call__(self, X): return X - X.mean(0)
MODEL=Model()
''')
    with pytest.raises(RuntimeError, match='batch composition'):
        model(np.arange(12, dtype=np.float32).reshape(4, 3))


@pytest.mark.parametrize('storage', ['global_tensor', 'global_numpy', 'model_numpy'])
def test_weights_outside_model_tensors_count_towards_cap(isolated, storage, monkeypatch):
    from neuro_eval.model import _validate_model_size

    monkeypatch.setenv('NEURO_MAX_PARAMS', '100000')
    if storage == 'global_tensor':
        weights = 'W=torch.randn(12,8334,generator=torch.Generator().manual_seed(0))'
        init, forward = '', 'X @ W'
    elif storage == 'global_numpy':
        weights = 'W=np.random.default_rng(0).standard_normal((12,8334)).astype(np.float32)'
        init, forward = '', 'X @ torch.from_numpy(W)'
    else:
        weights = ''
        init = ' def __init__(self): self.W=np.random.default_rng(0).standard_normal((12,8334)).astype(np.float32)'
        forward = 'X @ torch.from_numpy(self.W)'
    model = isolated(f'''import torch
import numpy as np
{weights}
class Model:
 n_params=0
{init}
 def __call__(self, X): return {forward}
MODEL=Model()
''')
    features = model(np.random.default_rng(42).random((8, 12), dtype=np.float32))
    assert model.n_params == 100008
    with pytest.raises(ValueError, match='exceeds'):
        _validate_model_size(model, features.shape[1])
