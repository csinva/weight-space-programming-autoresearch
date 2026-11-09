"""Shared image and filter helpers for the baselines."""
import functools
import math

import torch
import torch.nn.functional as F


def images(X):
    """Flattened HWC images -> (N, 3, S, S) float tensor."""
    X = torch.as_tensor(X).float()
    s = math.isqrt(X.shape[1] // 3)
    return X.reshape(-1, s, s, 3).permute(0, 3, 1, 2)


def gray(X):
    """Flattened HWC images -> (N, 1, S, S) luminance, zero-mean around mid-gray."""
    rgb = images(X)
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(1, keepdim=True) - 0.5


def chunked(fn, X, size=256):
    """Apply ``fn`` to row chunks of X to bound memory."""
    return torch.cat([fn(X[i:i + size]) for i in range(0, len(X), size)])


@functools.cache
def gabor_bank(freq, thetas, curvatures=(0.0,), sigma=None, device=None):
    """Complex Gabor (or curved 'banana') kernels, shape (len(thetas)*len(curvatures), k, k).

    ``freq`` is in cycles/pixel. Curvature c bends the carrier: x' -> x' + c * y'^2,
    with c in 1/pixel. Real part even, imaginary part odd; each kernel is zero-mean
    with unit L2 norm. Cached: pass tuples and do not modify the result.
    """
    sigma = sigma or 0.5 / freq                      # ~one-octave bandwidth
    half = math.ceil(3 * sigma)
    r = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(r, r, indexing="ij")
    kernels = []
    for theta in thetas:
        u = x * math.cos(theta) + y * math.sin(theta)
        v = -x * math.sin(theta) + y * math.cos(theta)
        for c in curvatures:
            ub = u + c * v ** 2
            k = torch.exp(-(ub ** 2 + v ** 2) / (2 * sigma ** 2)) * torch.exp(2j * math.pi * freq * ub)
            k = k - k.mean()
            kernels.append(k / k.abs().pow(2).sum().sqrt())
    return torch.stack(kernels)


def energy(img, bank):
    """Phase-invariant energy |img * bank|, shape (N, K, H, W); img is (N, 1, H, W)."""
    k = torch.cat([bank.real, bank.imag]).unsqueeze(1).to(img)
    out = F.conv2d(F.pad(img, [k.shape[-1] // 2] * 4, mode="reflect"), k)
    even, odd = out.chunk(2, dim=1)
    return (even ** 2 + odd ** 2).sqrt()
