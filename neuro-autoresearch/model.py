# idea: random-projection baseline -- a fixed random linear projection of the raw stimulus
"""The model: a single function from stimulus to representation.

This is the baseline -- the simplest useful never-trained model: a FIXED random linear
projection of the raw stimulus. The projection matrix is drawn once from a fixed seed (never
fit to data, never sees the neural target ``Y``), so it is untrained and deterministic. It maps
the stimulus batch ``X`` of shape ``(N, D_in)`` to a representation ``(N, D_out)`` with
``D_out`` under the readout's width cap, and the fixed ridge readout is fit on that. It is the
floor every proposed model should beat, and the seed a search proposer evolves upward from.

A small stimulus (``D_in <= D_out``) passes straight through unchanged; anything wider is
projected down to ``D_out`` random features.

Contract the harness relies on (see ``search/proposer.md``): define ``class Model`` with
``name`` / ``idea`` / ``summary`` / ``n_params`` and ``__call__(self, X) -> (N, D_out)``,
then end the file with ``MODEL = Model()``. Two hard rules keep the reward honest:
UNTRAINED (weights analytic or random-but-fixed, never fit to data, never read the target
``Y``) and PER-SAMPLE (a sample's activations depend only on that sample).
"""
from __future__ import annotations

import torch

D_OUT = 8192  # representation width; strictly under the harness ridge cap (D_out < 10000)
SEED = 0      # fixes the random projection so the model is deterministic / untrained


class Model:
    name = "random-projection"
    idea = "fixed random linear projection of the raw stimulus"
    summary = ("Baseline: project the raw flattened stimulus through a fixed (seeded) random "
               "Gaussian matrix to D_out features. Untrained, per-sample.")
    n_params = 0

    def __init__(self):
        self._W = None  # lazily built once we know D_in (and the input's device/dtype)

    @torch.no_grad()
    def __call__(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.reshape(X.shape[0], -1).float()
        d_in = X.shape[1]
        if d_in <= D_OUT:
            return X
        if self._W is None or self._W.shape != (d_in, D_OUT):
            g = torch.Generator(device="cpu").manual_seed(SEED)
            W = torch.randn(d_in, D_OUT, generator=g) / (d_in ** 0.5)
            self._W = W.to(X.device)
        return X @ self._W.to(X.device)


MODEL = Model()
