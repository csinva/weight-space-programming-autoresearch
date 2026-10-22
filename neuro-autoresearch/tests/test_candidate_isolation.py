"""Synthetic OS boundary checks; never open a neural dataset."""
from pathlib import Path
import shutil

import numpy as np
import pytest

from harness.isolation import IsolatedModel, check_isolation


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
    model = isolated(f'''import os, socket
class Model:
 def __call__(self, X):
  assert not os.path.exists({str(secret)!r})
  assert not os.path.exists('/home/ymehta3/research/neuro-autoresearch/search.db')
  try:
   socket.create_connection(('127.0.0.1', 9), timeout=.1)
  except OSError:
   pass
  else:
   raise AssertionError('host network reachable')
  return X.flatten(1)
MODEL=Model()
''')
    np.testing.assert_array_equal(model(np.ones((2, 3), dtype='float32')), np.ones((2, 3)))


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
