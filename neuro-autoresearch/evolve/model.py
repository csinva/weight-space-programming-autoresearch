"""THE editable artifact: the computational model the autoresearch loop evolves.

A model is a plain callable  stimulus batch X (N, D_in) -> activations (N, D_out).
Its weights are written directly in code; it is NEVER trained (no optimizer, no
backprop). Only the ridge readout in src/eval.py is fit — that is the evaluation.

To evolve: edit / add a Model below. Do NOT edit the harness in harness/.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))

from config import DATASET
from data import load_split
from eval import evaluate, evaluate_on_split, save_results


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


class RandomFourierModel:
    """Fixed (seeded, in-code) random Fourier features: cos(X W + b). Never trained.

    A modality-agnostic nonlinear basis (a random approximation of an RBF kernel
    map). Works for any input width D_in; the projection is scaled by
    1/sqrt(D_in) so its output range stays comparable across modalities.
    """

    name = "random_fourier"
    description = "Cosine of a fixed seed=0 random projection (256 random Fourier features)."

    def __init__(self, out_dim: int = 256, gamma: float = 1.0, seed: int = 0):
        self.out_dim, self.gamma, self.seed = out_dim, gamma, seed
        self.W = self.b = None

    @property
    def n_params(self) -> int:
        return 0 if self.W is None else self.W.numel() + self.b.numel()

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            g = torch.Generator().manual_seed(self.seed)
            scale = (self.gamma / X.shape[1]) ** 0.5
            self.W = torch.randn(X.shape[1], self.out_dim, generator=g) * scale
            self.b = torch.rand(self.out_dim, generator=g) * (2 * math.pi)
        return torch.cos(X @ self.W + self.b)


MODELS = {
    "identity": IdentityModel,
    "random_projection": RandomProjectionModel,
    "random_fourier": RandomFourierModel,
    "spatial_pyramid": SpatialPyramidModel,
}

# Models that run on ANY stimulus modality (image/text/audio). spatial_pyramid
# is image-only (it reshapes X into a square RGB image), so it is excluded.
MODALITY_AGNOSTIC = ["identity", "random_projection", "random_fourier"]

# The two proposals the loop evaluates by default: an initial model + ONE
# modification (a richer fixed nonlinear basis).
ITERATIONS = ["random_projection", "random_fourier"]

# The active model the loop is currently proposing.
MODEL_NAME = "random_fourier"
MODEL_DESCRIPTION = RandomFourierModel.description


def _print_row(prefix: str, row: dict) -> None:
    print(f"{prefix}{row['model_name']:<18} {row['n_params']:>8}  "
          f"{row['train_corr']:>10.4f}  {row['test_corr']:>9.4f}  "
          f"{row['frac_targets_above_0.1']:>8.4f}  {row['seconds']:>5.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, default=None,
                   help="evaluate a single named model (default: run two iterations)")
    args = p.parse_args()

    if args.model is not None:                       # single-model run (back-compat)
        row = evaluate(MODELS[args.model](), DATASET)
        print(f"[{row['dataset']}] {row['model_name']}: test_corr={row['test_corr']:.4f} "
              f"train_corr={row['train_corr']:.4f} n_params={row['n_params']} "
              f"({row['seconds']}s)")
        save_results(row)
        raise SystemExit

    # Two autoresearch iterations: an initial model + one modification. The split
    # is loaded ONCE (the slow step) and reused across both proposals.
    print(f"Dataset: {DATASET.study}  stimulus={DATASET.stimulus}  neural={DATASET.neural}")
    print("Loading split once (the slow step) ...")
    t = time.time()
    split = load_split(DATASET)
    Xtr, Xte, Ytr, Yte = split
    print(f"  X_train{tuple(Xtr.shape)} X_test{tuple(Xte.shape)} "
          f"Y_train{tuple(Ytr.shape)} Y_test{tuple(Yte.shape)}  "
          f"(load {time.time() - t:.1f}s)\n")

    header = (f"{'iter':>4}  {'model':<18} {'n_params':>8}  {'train_corr':>10}  "
              f"{'test_corr':>9}  {'frac>0.1':>8}  {'sec':>5}")
    print(header)
    print("-" * len(header))
    for i, name in enumerate(ITERATIONS, start=1):
        row = evaluate_on_split(MODELS[name](), split, DATASET)
        save_results(row)
        _print_row(f"{i:>4}  ", row)
