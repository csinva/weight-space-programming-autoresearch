"""Portilla-Simoncelli texture statistics over the whole image and a 2x2 grid."""
import torch
from plenoptic.models import PortillaSimoncelli

from baselines.filters import chunked, gray


class Texture:
    name = "texture"
    description = (
        "Portilla & Simoncelli (2000) texture statistics on the whole image and on each quadrant, "
        "a fixed-grid simplification of the eccentricity-scaled pooling windows of Freeman & Simoncelli (2011)."
    )
    n_params = 4 * 128 * 128  # rough: steerable-pyramid filters at full-image plus quadrant resolution

    def __init__(self):
        self._models = {}

    @torch.no_grad()
    def __call__(self, X):
        s = int((X.shape[1] // 3) ** 0.5)
        key = (s, X.device)
        if key not in self._models:  # built lazily; 4 scales on the full image, 3 on 64px quadrants
            self._models[key] = [PortillaSimoncelli((s // k,) * 2, n_scales=4 - k // 2, n_orientations=4).to(X.device)
                                 for k in (1, 2)]
        whole, quad = self._models[key]
        h = s // 2

        def run(x):
            img = gray(x)
            feats = [whole(img)[:, 0]]
            for i in (0, h):
                for j in (0, h):
                    feats.append(quad(img[:, :, i:i + h, j:j + h])[:, 0])
            y = torch.nan_to_num(torch.cat(feats, 1), nan=0.0, posinf=0.0, neginf=0.0)  # only flat images (0/0 moments)
            return y.sign() * y.abs().log1p()        # stats span ~5 decades; compress so ridge sees them all

        return chunked(run, X, 64)


MODELS = {"texture": Texture()}
