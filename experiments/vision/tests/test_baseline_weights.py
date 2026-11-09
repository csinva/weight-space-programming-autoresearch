"""Missing optional pretrained weights produce setup guidance before allocating AlexNet."""
import pytest

from baselines import alexnet


def test_missing_pretrained_weights_explains_override(tmp_path, monkeypatch):
    monkeypatch.setattr(alexnet, "WEIGHTS", tmp_path / "missing.pth")
    def unexpected_allocation():
        pytest.fail("Do not allocate the network before checking its weights")
    monkeypatch.setattr(alexnet, "alexnet", unexpected_allocation)
    alexnet.network.cache_clear()
    with pytest.raises(FileNotFoundError, match="NEURO_ALEXNET_WEIGHTS"):
        alexnet.network(True)
