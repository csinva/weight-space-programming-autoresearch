from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "evaluation"))

import run as evaluation_run  # noqa: E402
import vision  # noqa: E402


def test_dinov3_is_the_canonical_dino_baseline():
    spec = next(spec for spec in evaluation_run.VISION_BASELINES if "dino" in spec["id"])

    assert spec == {
        "id": "dinov3_pretrained",
        "name": "dinov3",
        "weights": "pretrained",
        "label": "DINOv3 ViT-B/16 (pretrained)",
    }
    assert vision.MODEL_REGISTRY["dinov3"] is vision.extract_dinov3


def test_dinov3_extractor_uses_meta_vitb16_checkpoint(monkeypatch):
    loaded = []

    class FakeProcessor:
        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

    class FakeAutoImageProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            loaded.append(("processor", model_id))
            return FakeProcessor()

    class FakeModel:
        config = SimpleNamespace(num_hidden_layers=12)

        def eval(self):
            return self

        def to(self, device):
            return self

        def requires_grad_(self, value):
            return self

        def __call__(self, pixel_values, output_hidden_states):
            batch = len(pixel_values)
            states = tuple(torch.full((batch, 5, 8), float(i)) for i in range(13))
            return SimpleNamespace(hidden_states=states, last_hidden_state=states[-1])

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id):
            loaded.append(("model", model_id))
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoImageProcessor=FakeAutoImageProcessor,
        AutoModel=FakeAutoModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    images = np.zeros((2, 4 * 4 * 3), dtype=np.float32)

    features = vision.extract_dinov3(
        images, imsize=4, dim=16, seed=0, device=torch.device("cpu"), batch=1
    )

    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert loaded == [("processor", model_id), ("model", model_id)]
    assert set(features) == {"block3", "block6", "block9", "block12", "embedding"}
    assert all(feature.shape == (2, 8) for feature in features.values())


def test_dinov3_extractor_explains_gated_access(monkeypatch):
    class GatedProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            raise OSError("You are trying to access a gated repo")

    fake_transformers = SimpleNamespace(AutoImageProcessor=GatedProcessor, AutoModel=object())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="hf auth login"):
        vision.extract_dinov3(
            np.zeros((1, 4 * 4 * 3), dtype=np.float32),
            imsize=4,
            dim=16,
            seed=0,
            device=torch.device("cpu"),
        )
