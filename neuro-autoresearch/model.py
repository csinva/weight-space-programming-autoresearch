"""Editable model. NSD: X = 16x16x3 RGB in [0,1] flattened (HWC). V1 front end, never trained."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

IMSIZE = 16


def _gabor_bank(orientations, freqs, sizes, sigmas):
    kmax = max(sizes)
    kernels = []
    for theta in orientations:
        for f in freqs:
            for ks, sigma in zip(sizes, sigmas):
                half = (ks - 1) / 2.0
                ys, xs = torch.meshgrid(torch.arange(ks) - half, torch.arange(ks) - half, indexing="ij")
                xr = xs * math.cos(theta) + ys * math.sin(theta)
                yr = -xs * math.sin(theta) + ys * math.cos(theta)
                env = torch.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2))
                carrier = 2 * math.pi * f * xr
                even = env * torch.cos(carrier); odd = env * torch.sin(carrier)
                even = even - even.mean(); odd = odd - odd.mean()
                even = even / (even.norm() + 1e-8); odd = odd / (odd.norm() + 1e-8)
                pad = (kmax - ks) // 2
                if pad:
                    even = F.pad(even, (pad, pad, pad, pad)); odd = F.pad(odd, (pad, pad, pad, pad))
                kernels.append(torch.stack([even, odd], 0))
    return torch.stack(kernels, 0)


class Model:
    name = "gabor_8orient_v6"
    idea = "v2 with 8 orientations (no normalization): raw+opp + Gabor energy pooled 8x8."
    n_params = 0
    N_ORIENT = 8

    def __init__(self):
        orientations = [i * math.pi / self.N_ORIENT for i in range(self.N_ORIENT)]
        self._bank = _gabor_bank(orientations, [0.2, 0.35, 0.5], [7, 7, 5], [2.5, 1.8, 1.4])
        self._ksize = self._bank.shape[-1]; self._nfilt = self._bank.shape[0]
        self._cached_dev = None

    def _weights(self, device, dtype):
        if self._cached_dev != device:
            self._w = self._bank.to(device=device, dtype=dtype); self._cached_dev = device
        return self._w

    def _energy(self, ch, wcat, pad):
        resp = F.conv2d(ch, wcat, padding=pad).reshape(ch.shape[0], self._nfilt, 2, IMSIZE, IMSIZE)
        return torch.sqrt(resp[:, :, 0] ** 2 + resp[:, :, 1] ** 2 + 1e-8)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        N = X.shape[0]
        img = X.reshape(N, IMSIZE, IMSIZE, 3).permute(0, 3, 1, 2)
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        rg = r - g; by = b - 0.5 * (r + g)
        w = self._weights(X.device, lum.dtype)
        pad = self._ksize // 2
        wcat = w.reshape(self._nfilt * 2, 1, self._ksize, self._ksize)
        feats = [X, rg.reshape(N, -1), by.reshape(N, -1)]
        for ch in (lum, rg, by):
            feats.append(F.adaptive_avg_pool2d(self._energy(ch, wcat, pad), 8).reshape(N, -1))
        return torch.cat(feats, dim=1)


MODEL = Model()
