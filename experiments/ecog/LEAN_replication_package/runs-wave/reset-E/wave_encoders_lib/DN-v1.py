"""DIVISIVE NORMALIZATION front-end for patient1 waveform -> ECoG high-gamma.

Paradigm (Carandini-Heeger): the response of every cochlear channel at every
time is DIVIDED by a pooled drive from its spectrotemporal neighborhood. A bank
of pools with different shapes (narrow vs broad in frequency, fast vs slow in
time, causal vs acausal) and different exponents produces multiple normalized
representations -- each captures contrast / gain invariance that a LINEAR map
over the bands cannot reconstruct.

All operations are closed-form (FFT, cumsum, separable conv). No fitted
parameters. No optimizer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data import SR
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR = "/path/to/scratch/feat_cache_reset_E"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cochlear front-end
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=64, f_min=50.0, f_max=8000.0, order=4):
    """|H(f)|^2 of a 4th-order gammatone bank on the rFFT frequency grid."""
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))


def power_spectrum(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2


# ---------------------------------------------------------------------------
# Pooling primitives (separable Gaussian in frequency, EMA in time)
# ---------------------------------------------------------------------------

def gauss_freq_kernel(sigma_chans, device, exclude_self=False):
    """1D Gaussian kernel along channel axis (used for surround pooling)."""
    k = max(1, int(np.ceil(3.0 * sigma_chans)))
    coords = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma_chans ** 2))
    if exclude_self:
        g[k] = 0.0
    g = g / (g.sum() + 1e-12)
    return g.view(1, 1, -1), k


def conv_freq(x: torch.Tensor, kernel: torch.Tensor, pad: int) -> torch.Tensor:
    """Conv along channel axis of (T, C) using a (1,1,K) kernel."""
    xp = x.unsqueeze(1)  # (T, 1, C)
    xp = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    return torch.nn.functional.conv1d(xp, kernel).squeeze(1)


def causal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Causal exponential moving average along time axis 0."""
    if tau_frames <= 0.0:
        return x
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def acausal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Time-reversed EMA -- exponential lookahead."""
    if tau_frames <= 0.0:
        return x
    return torch.flip(causal_ema(torch.flip(x, dims=[0]), tau_frames), dims=[0])


def symmetric_time_pool(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Symmetric pool: average of causal + acausal EMA."""
    if tau_frames <= 0.0:
        return x
    return 0.5 * (causal_ema(x, tau_frames) + acausal_ema(x, tau_frames))


# ---------------------------------------------------------------------------
# Divisive normalization
# ---------------------------------------------------------------------------

# A "pool spec" = (freq_sigma_chans, time_tau_frames, time_kind, exclude_self)
# time_kind in {"causal", "acausal", "sym", "none"}
POOL_SPECS = [
    # Pure frequency surrounds (instantaneous DN across freq channels)
    (1.0,  0.0,  "none",    True),
    (3.0,  0.0,  "none",    True),
    (8.0,  0.0,  "none",    True),
    # Pure temporal adaptation (per channel, no freq pooling)
    (0.0,  3.0,  "causal",  False),
    (0.0,  15.0, "causal",  False),
    (0.0,  60.0, "causal",  False),
    # Joint freq + causal time pools (classic DN)
    (3.0,  4.5,  "causal",  True),
    (3.0,  15.0, "causal",  True),
    (8.0,  4.5,  "causal",  True),
    (8.0,  15.0, "causal",  True),
    (8.0,  60.0, "causal",  True),
    # Symmetric (acausal) joint pools -- DN with lookahead
    (3.0,  15.0, "sym",     True),
    (8.0,  60.0, "sym",     True),
]

# Exponents for divisive normalization (energy-based DN is canonical)
EXPONENTS = (2.0,)
# Semi-saturation fraction of per-chunk median pool value
SIGMA_FRAC = 0.1


def time_pool(x: torch.Tensor, tau: float, kind: str) -> torch.Tensor:
    if kind == "none" or tau <= 0.0:
        return x
    if kind == "causal":
        return causal_ema(x, tau)
    if kind == "acausal":
        return acausal_ema(x, tau)
    if kind == "sym":
        return symmetric_time_pool(x, tau)
    raise ValueError(kind)


def divisive_normalize(e_p: torch.Tensor, freq_kernel, freq_pad,
                       tau: float, time_kind: str,
                       sigma_frac: float = SIGMA_FRAC):
    """Carandini-Heeger DN:
        norm = e_p / (sigma + pool(e_p))
    Returns (norm, log_dn, residual_above_pool) all of shape (T, C).
    sigma = sigma_frac * median(pool) per chunk (scalar).
    """
    if freq_kernel is not None:
        pool = conv_freq(e_p, freq_kernel, freq_pad)
    else:
        pool = e_p
    pool = time_pool(pool, tau, time_kind)
    # per-chunk semi-saturation scale (data-dependent, not a fitted param)
    sigma = sigma_frac * pool.median().detach()
    sigma = sigma.clamp(min=1e-8)
    denom = sigma + pool
    norm = e_p / denom
    log_dn = torch.log(e_p.clamp(min=1e-12)) - torch.log(denom.clamp(min=1e-12))
    residual = (e_p - pool).clamp(min=0.0)  # rectified above-context drive
    return norm, log_dn, residual


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Divisive-normalization bank on a 4096-sample gammatone cochleagram."""

    win = 4096
    n_params = 0
    N_FILTERS = 48

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.N_FILTERS).to(device)
        # Pre-build pool freq kernels
        self._freq_kernels = {}
        for spec in POOL_SPECS:
            fs, _, _, excl = spec
            key = (round(fs, 3), excl)
            if key not in self._freq_kernels and fs > 0.0:
                self._freq_kernels[key] = gauss_freq_kernel(fs, device, exclude_self=excl)

    def _get_freq_kernel(self, fs: float, excl: bool):
        if fs <= 0.0:
            return None, 0
        return self._freq_kernels[(round(fs, 3), excl)]

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        p = power_spectrum(x, self.win, self.device)
        e = (p @ self.fb).clamp(min=1e-12)  # (T, F) cochleagram energy

        feats = []
        # Anchor: raw log-cochleagram
        log_e = torch.log(e)
        feats.append(log_e)

        # Divisive normalization bank over exponents x pool specs
        for exponent in EXPONENTS:
            e_p = e ** exponent
            for (fs, tau, kind, excl) in POOL_SPECS:
                fk, fp = self._get_freq_kernel(fs, excl)
                norm, log_dn, resid = divisive_normalize(e_p, fk, fp, tau, kind)
                feats.append(log_dn)
                feats.append(norm)
                # log of rectified residual (avoid raw scale issues)
                feats.append(torch.log(resid.clamp(min=1e-12)))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "DN-v1"
model_description = (
    "Divisive normalization bank: 48-ch gammatone cochleagram, exponent p=2, "
    "13 pool specs (freq-only, time-only, joint causal, joint symmetric), "
    "outputs log_dn + norm + log(residual). Plus log_e anchor. "
    "Carandini-Heeger paradigm: response / (sigma + pool). Feat dim 48 + 13*3*48 = 1920."
)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------

def build_embedder(device: str = "cuda") -> WaveEmbedder:
    return WaveEmbedder(device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available -- falling back to cpu")
        args.device = "cpu"

    t0 = time.time()
    cfg = EncodingConfig(num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device)
    r = run_encoding(embedder, cfg)
    n_params = int(getattr(embedder, "n_params", 0))

    upsert_overall_results(
        [make_result_row(r, model_shorthand_name, n_params, model_description)], RESULTS_DIR)
    plot_corr_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"subject:        {r['subject']}")
    print(f"test_corr:      {r['test_corr']:.4f}  (train_corr={r['train_corr']:.4f}, "
          f"frac_elec>0.1={r['frac_test_elec_above_0.1']:.4f})")
    print(f"n_params:       {n_params:.2e}")
    print(f"encoding_secs:  {r['encoding_seconds']:.1f}s")
    print(f"total_seconds:  {time.time() - t0:.1f}s")
