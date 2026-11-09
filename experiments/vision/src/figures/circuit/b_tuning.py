"""Panel b data: responses of three texture operations of the searched model (erjpu) to synthetic stimuli.

Each series varies one image statistic and records the operation's own intermediate quantity, together with
the mean first-stage oriented energy (half resolution) as a reference.

  M  texture boundaries: band-pass noise whose contrast is modulated by a 16 px grating; the grating
     orientation varies. Response: modulation energy of the four second-order orientation channels (Stage 2A),
     averaged over the eight orientation inputs and space. 16 noise samples per orientation.
  V  co-occurrence: eight horizontal and eight vertical Gabor patches on a 4 x 4 grid; the fraction of patches
     that share a cell with a patch of the other orientation varies. Response: spatial correlation between the
     horizontal and vertical energy maps (Stage 2B, half resolution). 16 random layouts per fraction.
  O  cross-scale structure: 64 shared LAION images whose Fourier phases are shifted by a fraction of a random
     phase field (amplitude spectrum unchanged before clipping to [0, 1]). Response: spatial correlation
     between the four full-resolution orientation channels and half-resolution orientation strength (Stage 2F).

Writes artifacts/figures/circuit/data/b.npz. Usage: python src/figures/circuit/b_tuning.py
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, channels24, correlation, intermediates, load_model, octave_summaries, second_order

N = 128
YY, XX = torch.meshgrid(torch.arange(N).float() - N / 2, torch.arange(N).float() - N / 2, indexing="ij")


def gray(lum):
    return lum.clamp(0, 1)[:, None].repeat(1, 3, 1, 1)


def edge_energy(m):
    return m["stage1"][1][:, 0:16].mean(dim=(1, 2, 3))


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return v.mean(0), v.std(0, ddof=1) / math.sqrt(len(v))


def band_noise(n):
    f = torch.fft.fftfreq(N)
    r = torch.sqrt(f[:, None] ** 2 + f[None] ** 2)
    t = torch.fft.ifft2(torch.fft.fft2(torch.randn(n, N, N)) * ((r > 0.15) & (r < 0.35))).real
    return t / t.std(dim=(1, 2), keepdim=True)


def gabor(cx, cy, theta, sigma=5.0, wavelength=6.0):
    t = math.radians(theta)
    u = (XX - cx) * math.cos(t) + (YY - cy) * math.sin(t)
    return torch.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma ** 2)) * torch.cos(2 * math.pi * u / wavelength)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    _, g = load_model()
    out = {}

    thetas = np.arange(0, 181, 15)
    resp, energy, examples = [], [], []
    for th in thetas:
        t = math.radians(float(th))
        envelope = 1 + torch.cos(2 * math.pi * (XX * math.cos(t) + YY * math.sin(t)) / 16.0)
        x = gray(0.5 + 0.12 * band_noise(16) * envelope)
        m = intermediates(g, x)
        resp.append(second_order(m)[:, :8].mean(dim=(1, 3, 4)).numpy())
        energy.append(edge_energy(m).numpy())
        if th in (0, 90):
            examples.append(x[0].permute(1, 2, 0).numpy())
    resp, energy = np.array(resp), np.array(energy)                       # (theta, sample, channel), (theta, sample)
    out.update(M_x=thetas, M_y=resp.mean(1), M_se=resp.std(1, ddof=1) / 4, M_energy=energy.mean(1),
               M_examples=np.array(examples))

    grid = [(a, b) for a in range(-48, 49, 32) for b in range(-48, 49, 32)]
    fractions, corr, energy, examples = np.arange(9) / 8, [], [], []
    for k in range(9):
        images = []
        for _ in range(16):
            perm, img = rng.permutation(16), torch.zeros(N, N)
            for i in perm[:k]:
                img += 0.5 * (gabor(*grid[i], 0) + gabor(*grid[i], 90))
            for i in perm[k:8]:
                img += 0.5 * gabor(*grid[i], 0)
            for i in perm[8:16 - k]:
                img += 0.5 * gabor(*grid[i], 90)
            images.append(img)
        x = gray(0.5 + 0.35 * torch.stack(images))
        m = intermediates(g, x)
        corr.append(correlation(channels24(m["stage1"][1]))[:, 0, 4].numpy())
        energy.append(edge_energy(m).numpy())
        if k in (0, 8):
            examples.append(x[0].permute(1, 2, 0).numpy())
    mean, se = stats(np.array(corr).T)
    out.update(V_x=fractions, V_y=mean, V_se=se, V_energy=np.array(energy).mean(1), V_examples=np.array(examples))

    R = np.load(DATA / "readout.npz")
    idx = rng.choice(len(R["images"]), 64, replace=False)
    x0 = torch.from_numpy(R["images"][idx]).float().div(255).permute(0, 3, 1, 2)
    phi = torch.angle(torch.fft.fft2(torch.rand(len(x0), 1, N, N)))
    alphas, corr, energy, examples = np.array([0, .125, .25, .375, .5, .625, .75, .875, 1.0]), [], [], []
    show = int(np.argmax(R["pred"].mean(0)[2][idx]))                      # the sampled image PPA responds to most
    for a in alphas:
        x = torch.fft.ifft2(torch.fft.fft2(x0) * torch.exp(1j * float(a) * phi)).real.clamp(0, 1)
        m = intermediates(g, x)
        o = octave_summaries(g, m)
        env = torch.cat([o[1][:, 4:8], m["curv"][1][:, [2, 5, 10, 12]]], dim=1)
        corr.append(correlation(F.avg_pool2d(o[0], 2, 2), env)[:, :4, 4].mean(1).numpy())
        energy.append(edge_energy(m).numpy())
        if a in (0, 1):
            examples.append(x[show].permute(1, 2, 0).numpy())
    mean, se = stats(np.array(corr).T)
    out.update(O_x=alphas, O_y=mean, O_se=se, O_energy=np.array(energy).mean(1), O_examples=np.array(examples))
    np.savez_compressed(DATA / "b.npz", **out)
    for k in "MVO":
        print(k, "response", np.round(out[f"{k}_y"], 3).tolist(), "\n  edge energy", np.round(out[f"{k}_energy"], 3).tolist())


if __name__ == "__main__":
    main()
