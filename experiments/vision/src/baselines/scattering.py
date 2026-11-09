"""2D wavelet scattering transform on luminance."""
import torch
import torch.nn.functional as F
# kymatio.torch fails to import with this scipy (removed sph_harm, needed only by 3D), so import 2D directly
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D as Scattering2D

from baselines.filters import chunked, gray


class Scattering:
    name = "scattering"
    description = "Order-0/1/2 wavelet scattering coefficients (J=4, L=8) pooled to 4x4, log(1+x) (Mallat 2012; Bruna & Mallat 2013)."
    n_params = 33 * 128 * 128  # rough: 33 Fourier-domain wavelet/low-pass filters (J*L + 1) at 128x128

    def __init__(self):
        self._ops = {}

    @torch.no_grad()
    def __call__(self, X):
        key = (int((X.shape[1] // 3) ** 0.5), X.device)
        if key not in self._ops:  # built lazily, per image size and device
            self._ops[key] = Scattering2D(J=4, L=8, shape=(key[0],) * 2, max_order=2).to(X.device)

        def run(x):
            s = self._ops[key](gray(x)[:, 0].contiguous())                  # (n, 417, S/16, S/16)
            s = F.adaptive_avg_pool2d(s, 4).flatten(1)                      # 417 * 16 = 6672
            return torch.log1p(s.clamp_min(0))                              # log compression, usual for scattering

        return chunked(run, X, 64)


MODELS = {"scattering": Scattering()}
