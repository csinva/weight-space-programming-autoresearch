"""RSA's empty training split must still yield a useful equivalence input spec."""
import json

import pytest
import torch

from scripts.simplify import capture_spec


@pytest.mark.parametrize("train_rows", [0, 40])
def test_capture_uses_search_when_train_empty_and_limits_training_sample(tmp_path, monkeypatch, train_rows):
    train = torch.arange(train_rows * 2).reshape(train_rows, 2)
    search = torch.tensor([[100, 101], [102, 103]])
    monkeypatch.setattr(capture_spec, "load_split", lambda _: (train, search, None, None, {}, None))
    output = tmp_path / "spec.json"
    capture_spec.main([str(output)])
    assert json.loads(output.read_text()) == {
        "stim_shape": [2], "dtype": "int64", "seed": 0,
        "batch": 32 if train_rows else 2,
        "low": 0.0 if train_rows else 100.0,
        "high": 63.0 if train_rows else 103.0,
    }


def test_empty_splits_fail_without_writing_spec(tmp_path, monkeypatch):
    empty = torch.empty((0, 12))
    monkeypatch.setattr(capture_spec, "load_split", lambda _: (empty, empty, None, None, {}, None))
    output = tmp_path / "spec.json"
    with pytest.raises(ValueError, match="both training and search/test stimulus splits are empty"):
        capture_spec.main([str(output)])
    assert not output.exists()
