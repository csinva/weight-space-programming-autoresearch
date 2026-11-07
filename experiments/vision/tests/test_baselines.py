"""Every baseline meets the candidate output contract on random images."""
import os

import pytest
import torch

pytest.importorskip("kymatio")
pytest.importorskip("plenoptic")
from baselines import load
from baselines.alexnet import WEIGHTS

MODELS = {k: v for k, v in load().items()
          if k.startswith(("alexnet_random", "alexnet_fc8")) or not k.startswith("alexnet")}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_contract(name):
    if name.startswith("alexnet_fc") and not os.path.exists(WEIGHTS):
        pytest.skip("ImageNet weights unavailable")
    X = torch.rand(3, 128 * 128 * 3, generator=torch.Generator().manual_seed(0))
    X[2] = 0.5                                                   # flat image
    Y = MODELS[name](X)
    assert Y.ndim == 2 and len(Y) == 3 and Y.shape[1] < 10_000
    assert torch.isfinite(Y).all()
    torch.testing.assert_close(MODELS[name](X[:1]), Y[:1], atol=1e-4, rtol=1e-4)   # batch independence
