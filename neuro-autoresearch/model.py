# idea: random-projection baseline -- a fixed random linear projection of the raw stimulus
"""
Random-projection baseline: passes the stimulus through a fixed seeded random linear map to D_OUT,
unless input is small, in which case it passes through unchanged. Untrained, per-sample, deterministic.
"""
import torch

D_OUT, SEED = 8192, 0

class Model:
    name = "random-projection"
    idea = "fixed random linear projection of the raw stimulus"
    summary = "Baseline: fixed random linear map to D_OUT features. Untrained, per-sample."
    n_params = 0

    def __init__(self):
        self._W = None

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X).reshape(X.shape[0], -1).float()
        d_in = X.shape[1]
        if d_in <= D_OUT:
            return X
        if self._W is None or self._W.shape != (d_in, D_OUT):
            g = torch.Generator(device=X.device).manual_seed(SEED)
            self._W = torch.randn(d_in, D_OUT, generator=g, device=X.device) / d_in**0.5
        return X @ self._W

MODEL = Model()
