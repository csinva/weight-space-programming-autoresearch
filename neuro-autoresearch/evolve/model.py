"""THE editable artifact: the computational model the autoresearch loop evolves.

A model is a plain callable  stimulus batch X (N, D_in) -> activations (N, D_out).
Its weights are written directly in code; it is NEVER trained (no optimizer, no
backprop). Only the ridge readout in src/eval.py is fit — that is the evaluation.

To evolve: edit / add a Model below. Do NOT edit the harness in harness/.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))

from config import DATASET
from eval import evaluate, save_results


class IdentityModel:
    """Trivial baseline: activations = the raw stimulus pixels."""

    name = "identity"
    description = "Activations are the raw input pixels (no transform)."
    n_params = 0

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X


class RandomProjectionModel:
    """Fixed (seeded, in-code) random linear projection + ReLU. Never trained."""

    name = "random_projection"
    description = "ReLU of a fixed seed=0 Gaussian projection to 64 dims."

    def __init__(self, out_dim: int = 64, seed: int = 0):
        self.out_dim, self.seed, self.W = out_dim, seed, None

    @property
    def n_params(self) -> int:
        return 0 if self.W is None else self.W.numel()

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            g = torch.Generator().manual_seed(self.seed)
            self.W = torch.randn(X.shape[1], self.out_dim, generator=g) / X.shape[1] ** 0.5
        return torch.relu(X @ self.W)


class SpatialPyramidModel:
    """Fixed multi-scale average-pooling of the stimulus image (an image pyramid).

    The raw 768-pixel input overfits (200 samples); a random 64-d projection
    discards visual structure. This instead mean-pools each image at several
    coarse grids and concatenates them -- a fixed, in-code linear operator (no
    training). The pooled features are smooth and low-variance, which suits the
    early-visual fMRI signal and the small sample count.
    """

    name = "spatial_pyramid"
    description = "Concatenated mean-pool of the image at 4x4 + 2x2 + 1x1 grids (per channel)."
    n_params = 0

    def __init__(self, grids: tuple[int, ...] = (4, 2, 1), channels: int = 3):
        self.grids, self.channels = grids, channels

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.shape
        side = int(round((d / self.channels) ** 0.5))
        img = X.reshape(n, side, side, self.channels)
        feats = []
        for g in self.grids:
            block = side // g
            pooled = img.reshape(n, g, block, g, block, self.channels).mean(dim=(2, 4))
            feats.append(pooled.reshape(n, g * g * self.channels))
        return torch.cat(feats, dim=1)


MODELS = {
    "identity": IdentityModel,
    "random_projection": RandomProjectionModel,
    "spatial_pyramid": SpatialPyramidModel,
}

# The active model the loop is currently proposing.
MODEL_NAME = "spatial_pyramid"
MODEL_DESCRIPTION = SpatialPyramidModel.description


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default=MODEL_NAME)
    model = MODELS[p.parse_args().model]()

    row = evaluate(model, DATASET)
    print(f"[{row['dataset']}] {row['model_name']}: test_corr={row['test_corr']:.4f} "
          f"train_corr={row['train_corr']:.4f} n_params={row['n_params']} "
          f"({row['seconds']}s)")
    save_results(row)
