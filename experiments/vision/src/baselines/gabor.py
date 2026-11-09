"""Gabor wavelet pyramid (Kay et al. 2008) with optional divisive normalization."""
import math

import torch
import torch.nn.functional as F

from baselines.filters import chunked, energy, gabor_bank, gray

THETAS = tuple(math.pi * i / 8 for i in range(8))
FREQ = 0.25      # cycles/pixel of the single kernel bank (sigma = 2 px)


def pyramid(X, grids, normalize=False, sigma=0.1):
    """Gabor energy at len(grids) octave-spaced scales, avg-pooled to grids[k] x grids[k].

    Scale k halves the image k times and reuses the same kernels, i.e. an effective
    frequency of FREQ / 2**k. Returns a list of (N, 8 * g * g) tensors, one per scale.
    """
    img, out, bank = gray(X), [], gabor_bank(FREQ, THETAS, device=X.device)
    for g in grids:
        e = energy(img, bank)                                             # (N, 8, H, W)
        if normalize:                                                     # Heeger 1992 / Carandini & Heeger 2012
            pool = F.avg_pool2d(e.mean(1, keepdim=True), 5, 1, 2, count_include_pad=False)
            e = e / (sigma + pool)
        out.append(F.adaptive_avg_pool2d(e, g).flatten(1))
        img = F.avg_pool2d(img, 2)
    return out


class Gabor:
    description = ("Grayscale 8-orientation Gabor quadrature energy over 5 octave scales, pooled to finer grids at "
                   "higher frequency, log-compressed (Kay, Naselaris, Prenger & Gallant 2008, Nature)")
    grids = (16, 8, 4, 2, 1)                                             # 8 * (256+64+16+4+1) = 2728 features

    def __init__(self, normalize=False):
        self.normalize = normalize
        self.name = "gabor_norm" if normalize else "gabor"
        if normalize:
            self.description += "; with divisive normalization (Heeger 1992; Carandini & Heeger 2012)"
        self.n_params = gabor_bank(FREQ, THETAS).numel() * 2                # real + imaginary kernels

    @torch.no_grad()
    def __call__(self, X):
        return chunked(lambda x: torch.cat(pyramid(x, self.grids, self.normalize), 1).log1p(), X)


MODELS = {"gabor": Gabor(), "gabor_norm": Gabor(normalize=True)}
