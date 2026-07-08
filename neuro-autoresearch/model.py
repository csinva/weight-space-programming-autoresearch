# idea: random-projection baseline -- a fixed random linear projection of the raw stimulus
"""
Random-projection baseline: passes the stimulus through a fixed seeded random linear map to D_OUT,
unless input is small, in which case it passes through unchanged. Untrained, per-sample, deterministic.
"""
import torch

D_OUT, SEED = 8192, 0

class Model:
    """The evolving artifact. A fixed, never-trained map ``X (N, D_in) -> activations (N, D_out)``;
    the harness fits a ridge readout from these activations to brain activity and reports held-out
    ``test_corr`` as the reward. Only this file changes across the search."""
    name = "random-projection"
    idea = "fixed random linear projection of the raw stimulus"
    # description + insight are captured at score time into experiment_notes (the JSON that
    # peek-island shares with other islands, no source). Keep each to <=3 dense, plain-language,
    # re-implementable sentences.
    description = ("Fixed seeded random linear map from the flattened raw stimulus to D_OUT=8192 "
                   "features, with an identity passthrough when the input is already smaller. "
                   "Untrained, per-sample, and deterministic -- no learned or hand-tuned weights.")
    insight = ("Establishes the floor: a structure-free random projection of the raw stimulus that "
               "every engineered model should beat. Whatever predictivity it reaches is attributable "
               "to the ridge readout alone, not to any visual inductive bias.")
    n_params = 0

    def __init__(self):
        self._W = None

    @torch.no_grad()
    def __call__(self, X):
        """Stimulus batch ``X (N, ...)`` -> per-sample activations ``(N, D_out)``. Each row is
        computed independently (no cross-batch statistics); ``_W`` is lazily built once and reused."""
        X = torch.as_tensor(X).reshape(X.shape[0], -1).float()
        d_in = X.shape[1]
        if d_in <= D_OUT:
            return X
        if self._W is None or self._W.shape != (d_in, D_OUT):
            g = torch.Generator(device=X.device).manual_seed(SEED)
            self._W = torch.randn(d_in, D_OUT, generator=g, device=X.device) / d_in**0.5
        return X @ self._W

MODEL = Model()
