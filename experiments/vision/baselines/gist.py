"""GIST descriptor: oriented Gabor energy at 4 scales x 8 orientations on a 4x4 grid."""
import torch

from baselines.filters import chunked, gabor_bank
from baselines.gabor import FREQ, THETAS, pyramid


class Gist:
    name = "gist"
    description = ("Grayscale Gabor energy at 4 scales x 8 orientations averaged on a 4x4 spatial grid, "
                   "giving 512 features (Oliva & Torralba 2001, Int. J. Comput. Vis.)")

    n_params = gabor_bank(FREQ, THETAS).numel() * 2

    @torch.no_grad()
    def __call__(self, X):
        return chunked(lambda x: torch.cat(pyramid(x, (4,) * 4), 1), X)


MODELS = {"gist": Gist()}
