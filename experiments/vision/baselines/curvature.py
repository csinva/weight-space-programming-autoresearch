"""V4 curvature/contour model: banana (curved Gabor) filter energy with local max pooling."""
import math

import torch
import torch.nn.functional as F

from baselines.filters import chunked, energy, gabor_bank, gray


class Curvature:
    """Image-computable proxy for V4 angular-position x curvature tuning.

    Pasupathy & Connor's APC model (1999, 2001, 2002) describes tuning to the angular position
    and curvature of shape *outlines*, which cannot be computed from natural images. Following
    Kruger (1998) and Ponce et al. (2017) we instead use banana filters (Gabors with a curved
    carrier). Caveat: this captures local contour curvature and orientation, not boundary
    segmentation or object-centred position, so it is only a proxy for the APC model.
    """
    name = "curvature"
    description = ("Banana (curved Gabor) filter energy at 3 scales with 4x4 local max pooling, a proxy for "
                   "V4 curvature tuning (Pasupathy & Connor 1999/2001/2002; Kruger 1998).")
    FREQ, GRID, SCALES = 0.2, 4, (1, 2, 4)           # cycles/pixel at the finest scale; pooled grid; image downsampling
    # Orientations span 0..pi with signed curvatures: a banana bending one way at theta equals the
    # opposite bend at theta + pi, so this covers every orientation/bend combination exactly once.
    # Curvature c in 1/pixel shifts the carrier by c*v^2: 0.05 bends ~1.25 px (0.25 wavelength) at 1 sigma.
    CURVATURES = (-0.1, -0.05, 0.0, 0.05, 0.1)

    def __init__(self):
        self.thetas = tuple(math.pi * i / 8 for i in range(8))
        self.n_params = 2 * gabor_bank(self.FREQ, self.thetas, self.CURVATURES).numel()   # (40, 17, 17) complex, shared across scales

    @torch.no_grad()
    def __call__(self, X):
        return chunked(self._features, X)

    def _features(self, X):
        img, bank, feats = gray(X), gabor_bank(self.FREQ, self.thetas, self.CURVATURES, device=X.device), []
        for s in self.SCALES:
            small = F.avg_pool2d(img, s) if s > 1 else img                   # coarser scale = downsampled image
            feats.append(F.adaptive_max_pool2d(energy(small, bank), self.GRID).flatten(1))
        return torch.cat(feats, 1).log1p()


MODELS = {"curvature": Curvature()}
