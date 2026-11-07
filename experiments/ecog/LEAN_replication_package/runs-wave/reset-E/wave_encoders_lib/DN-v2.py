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
    (1.0,  0.0,  "none",    True),    # P0: narrow surround
    (3.0,  0.0,  "none",    True),    # P1
    (8.0,  0.0,  "none",    True),    # P2
    (24.0, 0.0,  "none",    True),    # P3: near-global freq context
    # Pure temporal adaptation (per channel, no freq pooling)
    (0.0,  1.5,  "causal",  False),   # P4: very fast adaptation
    (0.0,  4.5,  "causal",  False),   # P5
    (0.0,  15.0, "causal",  False),   # P6
    (0.0,  60.0, "causal",  False),   # P7
    # Joint freq + causal time pools (classic Carandini-Heeger DN)
    (3.0,  4.5,  "causal",  True),    # P8
    (3.0,  15.0, "causal",  True),    # P9
    (8.0,  4.5,  "causal",  True),    # P10
    (8.0,  15.0, "causal",  True),    # P11
    (8.0,  60.0, "causal",  True),    # P12
    (24.0, 15.0, "causal",  True),    # P13: wide freq + medium time
    # Symmetric joint pools -- DN with bidirectional context
    (3.0,  15.0, "sym",     True),    # P14
    (8.0,  15.0, "sym",     True),    # P15
    (8.0,  60.0, "sym",     True),    # P16
]

# Exponents for divisive normalization
# p=0.3 Stevens-loudness; p=1 raw; p=2 energy; p=3 superlinear contrast.
EXPONENTS = (0.3, 1.0, 2.0, 3.0)
# Semi-saturation fraction of per-chunk median pool value
SIGMA_FRAC = 0.1

# Center-surround opponency contrasts: pairs of pool indices into POOL_SPECS.
# Each pair (narrow_idx, wide_idx) yields log(pool_wide) - log(pool_narrow)
# = log-domain center-surround signal -- nonlinearly captures local prominence
# above background. Done per exponent.
OPPONENCY_PAIRS = (
    (0, 2),   # narrow freq vs wide freq, both instantaneous
    (0, 3),   # narrow freq vs near-global
    (8, 12),  # narrow joint pool vs wide joint pool
    (5, 7),   # fast vs slow temporal adaptation
    (8, 10),  # short-time joint vs long-time joint at same freq
)


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


def build_pool(e_p: torch.Tensor, freq_kernel, freq_pad,
               tau: float, time_kind: str) -> torch.Tensor:
    """Apply freq + time pooling to e_p. Returns (T, C) pool tensor."""
    if freq_kernel is not None:
        pool = conv_freq(e_p, freq_kernel, freq_pad)
    else:
        pool = e_p
    return time_pool(pool, tau, time_kind)


def dn_outputs(e_p: torch.Tensor, pool: torch.Tensor,
               sigma_frac: float = SIGMA_FRAC):
    """Carandini-Heeger DN outputs.
        log_dn   = log(e_p) - log(sigma + pool)        (response / context)
        log_pool = log(sigma + pool)                    (context magnitude)
        log_resid = log(1 + clip(e_p - pool, min=0))    (above-context drive)
    sigma = sigma_frac * median(pool) per chunk (scalar). All shape (T, C)."""
    sigma = sigma_frac * pool.median().detach()
    sigma = sigma.clamp(min=1e-8)
    denom = (sigma + pool).clamp(min=1e-12)
    log_dn = torch.log(e_p.clamp(min=1e-12)) - torch.log(denom)
    log_pool = torch.log(denom)
    resid = (e_p - pool).clamp(min=0.0)
    log_resid = torch.log1p(resid / sigma.clamp(min=1e-12))
    return log_dn, log_pool, log_resid


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
        # Anchor: raw log-cochleagram (lets ridge anchor the linear part)
        log_e = torch.log(e)
        feats.append(log_e)

        # Canonical DN at p=2 across the full pool bank
        e_p2 = e ** 2.0
        pools_p2 = []
        for (fs, tau, kind, excl) in POOL_SPECS:
            fk, fp = self._get_freq_kernel(fs, excl)
            pool = build_pool(e_p2, fk, fp, tau, kind)
            pools_p2.append(pool)
            log_dn, _, _ = dn_outputs(e_p2, pool)
            feats.append(log_dn)

        # log_resid for a few selected pool specs (transient / above-context drive)
        # P0 narrow surround, P4 fast adapt, P8 short joint, P11 wide joint, P14 sym
        for idx in (0, 4, 8, 11, 14):
            _, _, log_resid = dn_outputs(e_p2, pools_p2[idx])
            feats.append(log_resid)

        # Center-surround / contrast OPPONENCY across multiple exponents.
        # For each pair of pool specs, compute log(pool_wide) - log(pool_narrow)
        # at exponents p=1, 2, 3. Genuinely nonlinear because pool^(1/q)
        # is a non-linear function of the log-cochleagram bands.
        for exponent in (1.0, 2.0, 3.0):
            e_p = e ** exponent
            pool_cache = {}
            for (i_n, i_w) in OPPONENCY_PAIRS:
                for idx in (i_n, i_w):
                    if idx not in pool_cache:
                        fs, tau, kind, excl = POOL_SPECS[idx]
                        fk, fp = self._get_freq_kernel(fs, excl)
                        pool_cache[idx] = build_pool(e_p, fk, fp, tau, kind)
                pn = pool_cache[i_n]
                pw = pool_cache[i_w]
                # add tiny sigma derived from pool magnitudes
                sn = (0.01 * pn.median().detach()).clamp(min=1e-12)
                sw = (0.01 * pw.median().detach()).clamp(min=1e-12)
                opp = torch.log(pw + sw) - torch.log(pn + sn)
                feats.append(opp)

        feat = torch.cat(feats, dim=1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.float().cpu().numpy()


model_shorthand_name = "DN-v2"
model_description = (
    "DN-v2: 48-ch gammatone cochleagram. log_e anchor (48). log_dn at p=2 for "
    "17 pool specs (816). log_resid at 5 specs (240). Center-surround opponency "
    "log(pool_wide)-log(pool_narrow) for 5 pairs x 3 exponents (1,2,3) = 720. "
    "Total ~1824 dim. v1 test_corr=0.0545."
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
