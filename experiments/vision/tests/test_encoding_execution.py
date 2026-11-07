"""Persistent encoding execution preserves isolation and rejects candidates early."""
from contextlib import nullcontext
import subprocess
import numpy as np
import pytest
import torch

from neuro_eval import _evaluation as common
from neuro_eval.model import IsolatedModel, check_isolation


def proxy(execute):
    model=object.__new__(IsolatedModel)
    model.n_params=0
    model.repair_log=''
    model._execute=execute
    return model


def test_isolated_encoding_stages_cpu_without_parent_gpu_upload(monkeypatch):
    calls=[]
    def execute(array,device):
        assert isinstance(array,np.ndarray)
        calls.append((len(array),device))
        return array.copy()
    model=proxy(execute)
    model.encoding_session=lambda device:nullcontext(model)
    model.forward_encoding=lambda images,device:execute(images.numpy(),str(device))
    images=torch.arange(36,dtype=torch.float32).reshape(12,3)
    original_to=torch.Tensor.to
    def checked_to(self,*args,**kwargs):
        target=kwargs.get('device',args[0] if args else None)
        assert not str(target).startswith('cuda'), 'Parent attempted GPU upload'
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',checked_to)
    actual=common.encoding_forward(model,images,torch.device('cuda:2'),'features',batch_size=5)
    torch.testing.assert_close(actual,images)
    assert calls==[(5,'cuda:2'),(5,'cuda:2'),(2,'cuda:2')]
    assert actual.device.type=='cpu'


def test_explicit_device_keeps_fresh_context_checks():
    # A model using the batch mean must still be rejected by the singleton probe.
    model=proxy(lambda array,device:array-array.mean(0,keepdims=True))
    with pytest.raises(RuntimeError,match='batch composition'):
        model.forward_on_device(torch.arange(18.).reshape(6,3),'cpu')


def test_call_retains_inferred_device_and_numpy_behavior():
    devices=[]
    model=proxy(lambda array,device:(devices.append(device) or array.copy()))
    images=np.arange(12,dtype=np.float32).reshape(4,3)
    np.testing.assert_array_equal(model(images),images)
    assert devices==['cpu']*3


@pytest.mark.parametrize('kind',['width','nonfinite','parameters','rows','empty_width'])
def test_invalid_first_batch_fails_before_second_forward(kind):
    class Candidate:
        calls=0
        n_params=0
        def __call__(self,x):
            self.calls+=1
            if kind=='width':return torch.zeros(len(x),10000)
            if kind=='nonfinite':return x*float('nan')
            if kind=='parameters':self.n_params=100001
            if kind=='rows':return x[:-1]
            if kind=='empty_width':return x[:,:0]
            return x
    model=Candidate()
    with pytest.raises(ValueError):
        common.encoding_forward(model,torch.ones(12,3),torch.device('cpu'),'features',batch_size=4)
    assert model.calls==1


def test_later_width_change_rejected_before_third_batch():
    class Candidate:
        calls=0
        def __call__(self,x):
            self.calls+=1
            return x[:,:self.calls]
    model=Candidate()
    with pytest.raises(ValueError,match='width differs'):
        common.encoding_forward(model,torch.ones(12,3),torch.device('cpu'),'features',batch_size=4)
    assert model.calls==2


def test_numpy_and_tensor_models_keep_input_contract():
    for images in [np.arange(33,dtype=np.float32).reshape(11,3),torch.arange(33.).reshape(11,3)]:
        def model(batch):
            assert type(batch) is type(images)
            batch[:]=batch+2
            return batch
        expected=torch.as_tensor(images).clone()+2
        actual=common.encoding_forward(model,images,torch.device('cpu'),'features',batch_size=4)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.as_tensor(images),expected-2)


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_invalid_batch_size(size):
    with pytest.raises(ValueError,match='positive integer'):
        common.encoding_forward(lambda x:x,torch.ones(3,2),'cpu','features',batch_size=size)


def test_genuine_isolation_cpu_staging_matches_direct(tmp_path):
    try:
        check_isolation()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    path=tmp_path/'model.py'
    path.write_text('class Model:\n def __call__(self,X): return X[:,:3] * 2\nMODEL=Model()\n')
    model=IsolatedModel(path)
    images=torch.arange(20,dtype=torch.float32).reshape(5,4)
    actual=common.encoding_forward(model,images,'cpu','features',batch_size=3)
    np.testing.assert_array_equal(model(images),actual.numpy())
    torch.testing.assert_close(actual,images[:,:3]*2)


@pytest.mark.parametrize('kind',['parameters','nonfinite'])
def test_later_invalid_batch_is_checked_before_concatenation(kind):
    class Candidate:
        calls=0
        n_params=0
        def __call__(self,x):
            self.calls+=1
            if self.calls==2:
                if kind=='parameters': self.n_params=100001
                else: return x*float('inf')
            return x
    model=Candidate()
    with pytest.raises(ValueError):
        common.encoding_forward(model,torch.ones(12,3),'cpu','features',batch_size=4)
    assert model.calls==2


def test_encoding_default_is_dynamic_and_distinct_from_final(monkeypatch):
    from neuro_eval.model import FORWARD_BATCH
    assert FORWARD_BATCH==512
    assert common.ENCODING_FORWARD_BATCH==1024
    monkeypatch.setattr(common,'ENCODING_FORWARD_BATCH',3)
    calls=[]
    def model(x):
        calls.append(len(x))
        return x
    common.encoding_forward(model,torch.ones(8,2),'cpu','features')
    assert calls==[3,3,2]


@pytest.fixture
def isolated_workers(tmp_path, monkeypatch):
    try:
        check_isolation()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    workers = []
    original = subprocess.Popen

    def track(argv, *args, **kwargs):
        process = original(argv, *args, **kwargs)
        if '/worker.py' in argv:
            workers.append(process)
        return process

    monkeypatch.setattr(subprocess, 'Popen', track)

    def candidate(source):
        path = tmp_path / 'model.py'
        path.write_text(source)
        return IsolatedModel(path)

    return candidate, workers


def test_real_encoding_session_initializes_once_without_probes(isolated_workers):
    candidate, workers = isolated_workers
    model = candidate(
        'class Model:\n'
        ' def __init__(self): self.calls = 0\n'
        ' def __call__(self, X):\n'
        '  self.calls += 1\n'
        '  return X[:, :3] * 2 + self.calls\n'
        'MODEL = Model()\n')
    images = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    with model.encoding_session('cpu'):
        first = common.encoding_forward(model, images, 'cpu', 'features', batch_size=4)
        second = common.encoding_forward(model, images, 'cpu', 'features', batch_size=4)
        assert len(workers) == 1
        assert workers[0].poll() is None
    expected = images[:, :3] * 2
    torch.testing.assert_close(first, expected + torch.tensor([1, 1, 1, 1, 2, 2])[:, None])
    torch.testing.assert_close(second, expected + torch.tensor([3, 3, 3, 3, 4, 4])[:, None])
    assert workers[0].poll() is not None
    # A later candidate evaluation starts with a fresh instance.
    third = common.encoding_forward(model, images, 'cpu', 'features', batch_size=4)
    torch.testing.assert_close(third, first)
    assert len(workers) == 2
    assert workers[1].poll() is not None


@pytest.mark.parametrize('failure', ['candidate', 'validation', 'interrupt'])
def test_real_encoding_worker_cleanup_after_error(isolated_workers, failure):
    candidate, workers = isolated_workers
    body = {
        'candidate': "raise RuntimeError('deliberate candidate failure')",
        'validation': 'return X * float("nan")',
        'interrupt': 'return X * 2',
    }[failure]
    model = candidate('class Model:\n def __call__(self, X): ' + body + '\nMODEL = Model()\n')
    error = {'candidate': RuntimeError, 'validation': ValueError, 'interrupt': KeyboardInterrupt}[failure]
    with pytest.raises(error):
        with model.encoding_session('cpu'):
            common.encoding_forward(model, torch.ones(5, 3), 'cpu', 'features', batch_size=3)
            if failure == 'interrupt':
                raise KeyboardInterrupt
    assert len(workers) == 1
    assert workers[0].poll() is not None
