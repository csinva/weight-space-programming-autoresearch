"""
model.py — Sparse random sign projection features (no learning).

Each image's flattened raw pixels are used directly if there are 2,048 or
fewer values. Otherwise, 2,048 features are produced. Each feature is the sum
of 4 pixels, chosen uniformly at random with replacement, each multiplied by a
random +/-1 sign, then divided by 2. The pixel choices and signs come from
seed 0 and are shared across all images.

Materialized parameters: 2,048 * 4 indices + 2,048 * 4 signs = 16,384.
"""

import torch

N_FEATURES = 2048
K = 4            # pixels per feature
SEED = 0
CHUNK = 256      # images per chunk, which bounds GPU memory


class SparseSignProjection:
    def __init__(self, n_features=N_FEATURES, k=K, seed=SEED, chunk=CHUNK):
        self.n_features = n_features
        self.k = k
        self.seed = seed
        self.chunk = chunk
        self._cache = {}  # input dim -> (idx, signs) on CPU

    def _params(self, d_in, device):
        if d_in not in self._cache:
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed)
            idx = torch.randint(0, d_in, (self.n_features, self.k),
                                generator=g, dtype=torch.long)
            signs = torch.randint(0, 2, (self.n_features, self.k),
                                  generator=g).to(torch.float32) * 2.0 - 1.0
            self._cache[d_in] = (idx, signs)
        idx, signs = self._cache[d_in]
        return idx.to(device), signs.to(device)

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.reshape(X.shape[0], -1).to(torch.float32)
        n, d_in = X.shape

        if d_in <= self.n_features:
            return torch.nan_to_num(X.clone(), nan=0.0, posinf=0.0, neginf=0.0)

        idx, signs = self._params(d_in, X.device)
        flat_idx = idx.reshape(-1)  # (F*K,)
        outs = []
        for s in range(0, n, self.chunk):
            xc = X[s:s + self.chunk]                                   # (c, d_in)
            g = xc.index_select(1, flat_idx).view(-1, self.n_features, self.k)
            feats = (g * signs).sum(dim=2) / 2.0                        # (c, F)
            outs.append(feats)
        out = torch.cat(outs, dim=0) if outs else X.new_zeros((0, self.n_features))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


MODEL = SparseSignProjection()
