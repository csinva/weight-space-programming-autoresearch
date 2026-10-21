"""Fixed sparse random-projection baseline, within the 100,000-parameter contract."""
import torch

D_OUT, FAN_IN, SEED = 2048, 4, 0


class Model:
    """A per-image representation; no learning or access to neural responses."""

    name = "sparse-random-projection"
    idea = "fixed sparse signed projection of raw pixels"
    description = (
        "Each of 2,048 features combines four seeded, randomly sampled input pixels "
        "with fixed random signs. Small inputs pass through unchanged."
    )
    insight = "A cheap untrained reference for RSA search and held-out encoding."
    n_params = 2 * D_OUT * FAN_IN

    def __init__(self):
        self._indices = self._signs = None
        self._input_width = None

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X).reshape(len(X), -1).float()
        if X.shape[1] <= D_OUT:
            return X
        if self._input_width != X.shape[1] or self._indices.device != X.device:
            generator = torch.Generator(device=X.device).manual_seed(SEED)
            self._indices = torch.randint(
                X.shape[1], (D_OUT, FAN_IN), generator=generator, device=X.device)
            self._signs = torch.randint(
                2, (D_OUT, FAN_IN), generator=generator, device=X.device).float() * 2 - 1
            self._input_width = X.shape[1]
        return (X[:, self._indices] * self._signs).sum(-1) / FAN_IN ** 0.5


MODEL = Model()
