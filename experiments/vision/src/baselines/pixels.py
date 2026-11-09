"""Floor: downsampled pixels and global low-level image statistics."""
import torch
import torch.nn.functional as F

from baselines.filters import images


class Pixels:
    name = "pixels"
    description = "16x16 RGB thumbnail plus luminance/contrast/colour means and a radial power spectrum."
    n_params = 0

    @torch.no_grad()
    def __call__(self, X):
        rgb = images(X)
        thumb = F.adaptive_avg_pool2d(rgb, 16).flatten(1)                       # 768
        lum = rgb.mean(1)
        stats = torch.stack([lum.mean((1, 2)), lum.std((1, 2))], 1)             # luminance, RMS contrast
        colour = torch.cat([rgb.mean((2, 3)), rgb.std((2, 3))], 1)              # per-channel mean/sd
        power = torch.fft.fftshift(torch.fft.fft2(lum - lum.mean((1, 2), keepdim=True)).abs() ** 2, dim=(-2, -1))
        s = lum.shape[-1]
        r = torch.arange(s, device=lum.device) - s // 2
        radius = (r[:, None] ** 2 + r[None] ** 2).sqrt()
        bins = torch.logspace(0, torch.log10(torch.tensor(s / 2.0)), 13, device=lum.device)
        spectrum = torch.stack([power[:, (radius >= lo) & (radius < hi)].mean(1)
                                for lo, hi in zip(bins[:-1], bins[1:])], 1).log1p()   # 12 log-spaced bands
        return torch.cat([thumb, stats, colour, spectrum], 1)


MODELS = {"pixels": Pixels()}
