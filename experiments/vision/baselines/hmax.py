"""HMAX (S1-C1-S2-C2) model of the ventral stream."""
import math

import torch
import torch.nn.functional as F

from baselines.filters import chunked, energy, gabor_bank, gray


class HMAX:
    """Riesenhuber & Poggio (1999); Serre, Oliva & Poggio (2007).

    S1: Gabor energy, 4 orientations, on 6 image scales (factor 2^(-j/2)). C1: max over positions
    (8x8, stride 4) and over scale pairs. S2: Gaussian RBF match of 4x4x4 C1 patches to K prototypes.
    C2: global max over positions and scale bands, giving K features.

    Serre samples prototypes from C1 responses to natural images; with no training images we use
    seeded random sparse nonnegative patches (cubed uniforms, like sparse C1 activity) with unit
    norm. C1 patches are also unit-normalized, so exp(-beta*||x-p||^2) = exp(-2*beta*(1-cos)).
    """
    name = "hmax"
    description = ("HMAX with Gabor S1, max-pooled C1, RBF S2 on random C1 prototypes and global-max C2 "
                   "(Riesenhuber & Poggio 1999; Serre, Oliva & Poggio 2007).")
    K, PATCH, BETA, EPS = 1000, 4, 3.0, 1e-4

    def __init__(self):
        self.thetas = tuple(math.pi * i / 4 for i in range(4))
        self.bank = gabor_bank(0.2, self.thetas)                             # (4, 17, 17) complex
        g = torch.Generator().manual_seed(0)
        p = torch.rand(self.K, 4, self.PATCH, self.PATCH, generator=g) ** 3
        self.protos = p / p.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        self.n_params = 2 * self.bank.numel() + self.protos.numel()

    @torch.no_grad()
    def __call__(self, X):
        return chunked(self._features, X, 128)

    def _c1(self, img):
        """Max over positions at each of 6 scales, then over adjacent scale pairs -> 3 bands."""
        s1 = []
        for j in range(6):
            size = round(img.shape[-1] * 2 ** (-j / 2))
            small = F.interpolate(img, size=(size, size), mode="bilinear", antialias=True) if j else img
            s1.append(F.max_pool2d(energy(small, gabor_bank(0.2, self.thetas, device=img.device)), 8, 4))
        # Pair maps differ slightly in size; resize the coarser one to the finer one before taking the max.
        return [torch.maximum(a, F.interpolate(b, size=a.shape[-2:], mode="bilinear"))
                for a, b in zip(s1[::2], s1[1::2])]

    def _features(self, X):
        protos, c2 = self.protos.to(X.device), []
        for c1 in self._c1(gray(X)):
            dot = F.conv2d(c1, protos)                                       # x . p
            norm = (F.conv2d(c1 ** 2, torch.ones_like(protos[:1])) + self.EPS).sqrt()
            cos = dot / norm                                                 # = x.p / ||x|| since ||p|| = 1
            c2.append(torch.exp(-self.BETA * (2 - 2 * cos)).amax((2, 3)))
        return torch.stack(c2).amax(0)


MODELS = {"hmax": HMAX()}
