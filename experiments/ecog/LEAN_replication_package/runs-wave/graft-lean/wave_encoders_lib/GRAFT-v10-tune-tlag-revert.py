"""
Hand-designed auditory front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

The agent edits this file. Adds high-level feature classes (pitch salience,
phonetic templates, spectral moments/flatness/rolloff, HNR, cross-band
comodulation) on top of the best 3-window x 6-tau gammatone base.
"""
from __future__ import annotations

import argparse
import hashlib
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
CACHE_DIR = "/path/to/scratch/feat_cache_graft_lean"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=50, f_min=50.0, f_max=8000.0, order=4):
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))


def power_spectrum(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2


def leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def acausal_leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Time-reversed EMA — exponential 'lookahead' smoothing.
    Provides future-context per channel without re-windowing the audio."""
    return torch.flip(leaky_lowpass(torch.flip(x, dims=[0]), tau_frames), dims=[0])


# ---------------------------------------------------------------------------
# GRAFT-v2: Fixed-random echo-state reservoir helpers
# Closed-form: weights drawn from a fixed RNG (random, not data-fit).
# ---------------------------------------------------------------------------

def make_reservoir_W(N: int, spectral_radius: float, density: float, seed: int):
    rng = np.random.RandomState(seed)
    W = rng.randn(N, N).astype(np.float32)
    mask = (rng.rand(N, N) < density).astype(np.float32)
    W *= mask
    v = rng.randn(N).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(50):
        v = W @ v
        nrm = np.linalg.norm(v) + 1e-12
        v /= nrm
    rho = float(np.linalg.norm(W @ v))
    if rho < 1e-8:
        rho = 1.0
    W *= (spectral_radius / rho)
    return W


def make_input_matrix(N: int, dim_in: int, input_scale: float, seed: int):
    rng = np.random.RandomState(seed + 9999)
    return (rng.randn(N, dim_in).astype(np.float32) * input_scale) / np.sqrt(dim_in)


def make_leak_vector(N: int, leak_low: float, leak_high: float, seed: int):
    rng = np.random.RandomState(seed + 7777)
    log_low, log_high = np.log(leak_low), np.log(leak_high)
    leaks = np.exp(rng.uniform(log_low, log_high, size=N)).astype(np.float32)
    return leaks


@torch.no_grad()
def run_reservoir_mt(u: torch.Tensor, W: torch.Tensor, W_in: torch.Tensor,
                     leaks: torch.Tensor, activation: str = "tanh") -> torch.Tensor:
    """Per-frame sequential reservoir update — the ONE permitted time-loop
    (recurrence is intrinsically sequential). Vectorized across units on GPU.
    """
    T = u.shape[0]
    N = W.shape[0]
    x = torch.zeros(N, device=u.device, dtype=u.dtype)
    drive_in = u @ W_in.T                                  # (T, N)
    states = torch.empty(T, N, device=u.device, dtype=u.dtype)
    one_minus = 1.0 - leaks
    if activation == "tanh":
        for t in range(T):
            x = one_minus * x + leaks * torch.tanh(W @ x + drive_in[t])
            states[t] = x
    elif activation == "relu":
        # Saturated ReLU: tanh(0.3*max(z,0))*3 — bounded, no blow-up.
        for t in range(T):
            z = W @ x + drive_in[t]
            phi = torch.tanh(0.3 * z.clamp(min=0.0)) * 3.0
            x = one_minus * x + leaks * phi
            states[t] = x
    else:
        raise ValueError(activation)
    return states


# ---------------------------------------------------------------------------
# High-level features (closed-form, all derived from power spectrum / cochleagram)
# ---------------------------------------------------------------------------

def harmonic_comb_salience(power: torch.Tensor, win: int, f0_candidates: torch.Tensor,
                            n_harmonics: int = 8) -> torch.Tensor:
    """Harmonic-comb pitch salience: for each F0 candidate, sum log-power at
    its first n harmonic bins (normalized by total log-power).
    power: (T, n_freqs). Returns (T, len(f0_candidates))."""
    n_freqs = power.shape[1]
    freq_res = SR / win  # Hz per bin
    log_p = torch.log(torch.clamp(power, min=1e-10))
    total = log_p.mean(dim=1, keepdim=True)
    out = []
    for f0 in f0_candidates:
        # Bins at k*f0, k=1..n_harmonics
        ks = torch.arange(1, n_harmonics + 1, device=power.device, dtype=torch.float32)
        bins = (f0 * ks / freq_res).long().clamp(max=n_freqs - 1)
        # Pick those bins and sum
        sal = log_p[:, bins].mean(dim=1, keepdim=True) - total
        out.append(sal)
    return torch.cat(out, dim=1)  # (T, F0)


def spectral_moments(power: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Spectral centroid, spread, skewness, kurtosis from raw power spectrum.
    Returns (T, 4)."""
    p = power / (power.sum(dim=1, keepdim=True) + 1e-12)
    mu = (p * freqs).sum(dim=1, keepdim=True)
    df = freqs - mu
    var = (p * df ** 2).sum(dim=1, keepdim=True)
    std = torch.sqrt(var + 1e-12)
    skew = (p * df ** 3).sum(dim=1, keepdim=True) / (std ** 3 + 1e-12)
    kurt = (p * df ** 4).sum(dim=1, keepdim=True) / (var ** 2 + 1e-12)
    return torch.cat([torch.log(mu.clamp(min=1.0)), torch.log(std.clamp(min=1.0)),
                       skew.clamp(-10, 10), kurt.clamp(0, 50)], dim=1)


def spectral_flatness(power: torch.Tensor) -> torch.Tensor:
    """log(geo_mean / arith_mean). Higher = more noise-like, lower = more tonal.
    Returns (T, 1)."""
    log_p = torch.log(torch.clamp(power, min=1e-10))
    geo = log_p.mean(dim=1, keepdim=True)
    arith = torch.log(power.mean(dim=1, keepdim=True) + 1e-10)
    return geo - arith  # log of ratio


def spectral_rolloff(power: torch.Tensor, freqs: torch.Tensor,
                      thresholds=(0.85, 0.95)) -> torch.Tensor:
    """Frequency (log) below which `threshold` fraction of energy lies.
    Returns (T, len(thresholds))."""
    cs = torch.cumsum(power, dim=1)
    total = cs[:, -1:] + 1e-12
    out = []
    for thr in thresholds:
        # First bin where cumulative >= thr * total
        mask = cs >= thr * total
        idx = mask.float().argmax(dim=1)  # (T,)
        out.append(torch.log(freqs[idx].clamp(min=1.0)).unsqueeze(1))
    return torch.cat(out, dim=1)


def hnr_from_acf(power: torch.Tensor, win: int,
                  f_min: float = 75.0, f_max: float = 400.0) -> torch.Tensor:
    """Harmonics-to-noise ratio from normalized autocorrelation peak.
    Returns (T, 2): [peak_value, hnr_db]."""
    full = torch.fft.irfft(power, n=win)
    acf0 = full[:, :1] + 1e-12
    acf_n = full / acf0
    lag_min = max(1, int(SR / f_max))
    lag_max = min(win - 1, int(SR / f_min))
    region = acf_n[:, lag_min:lag_max + 1]
    peak_val, _ = region.max(dim=1, keepdim=True)
    peak_val = peak_val.clamp(min=0.0, max=0.999)
    hnr_db = 10.0 * torch.log10(peak_val.clamp(min=1e-3) /
                                  (1.0 - peak_val).clamp(min=1e-3))
    return torch.cat([peak_val, hnr_db], dim=1)


def phonetic_templates(log_e_long: torch.Tensor, n_filters: int) -> torch.Tensor:
    """Closed-form phonetic matched filters on the 20-channel long-window
    log-cochleagram.

      - sibilant: high-frequency band energy minus low (channels 14..19 vs 0..7)
      - vowel:    low-mid voiced energy (channels 4..12, with harmonicity bonus)
      - nasal:    narrow low energy (channels 1..6 minus 8..14)
      - plosive:  broadband flat energy minus tonal (high std, low harmonicity)
      - fricative: voiceless high energy minus low (sub-template of sibilant, sharper)

    Returns (T, 5)."""
    hi = log_e_long[:, max(0, n_filters - 6):].mean(dim=1, keepdim=True)
    lo = log_e_long[:, :max(1, n_filters // 3)].mean(dim=1, keepdim=True)
    mid = log_e_long[:, n_filters // 4:3 * n_filters // 4].mean(dim=1, keepdim=True)
    very_hi = log_e_long[:, max(0, n_filters - 3):].mean(dim=1, keepdim=True)
    narrow_lo = log_e_long[:, 1:6].mean(dim=1, keepdim=True)
    notch = log_e_long[:, 8:14].mean(dim=1, keepdim=True) if n_filters >= 14 else mid
    flatness = log_e_long.std(dim=1, keepdim=True)
    sibilant = hi - lo
    vowel = mid - flatness
    nasal = narrow_lo - notch
    plosive = flatness - (log_e_long.mean(dim=1, keepdim=True))
    fricative = very_hi - mid
    return torch.cat([sibilant, vowel, nasal, plosive, fricative], dim=1)


def cross_band_comodulation(log_e: torch.Tensor) -> torch.Tensor:
    """Products of band-energy pairs (cross-band correlation features).
    Returns (T, 6)."""
    nF = log_e.shape[1]
    # 6 representative bands evenly spaced
    band_idx = np.linspace(0, nF - 1, 7).astype(int)
    bands = []
    for i in range(6):
        bands.append(log_e[:, band_idx[i]:band_idx[i + 1] + 1].mean(dim=1, keepdim=True))
    bands_t = torch.cat(bands, dim=1)  # (T, 6)
    # Products of adjacent pairs + low*high + mid*high (6 pairs)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (2, 5)]
    out = []
    for a, b in pairs:
        out.append((bands_t[:, a] * bands_t[:, b]).unsqueeze(1))
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Best 3-window x 6-tau base + high-level features:
    pitch salience (harmonic comb), spectral moments, flatness, rolloff,
    HNR, phonetic templates, cross-band comodulation.
    """

    win = 4096
    n_params = 0
    WIN_LIST = (1024, 2048, 4096)
    TAUS = (1.5, 4.5, 15.0, 60.0, 150.0)
    SUB_WIN = 1024
    F0_CANDIDATES = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0, 230.0, 270.0, 320.0, 380.0, 450.0)
    HL_TAUS = (15.0, 60.0, 150.0)  # multi-tau for high-level features
    # (temporal Hz, spectral cyc/channel) Chi-Shamma rate-scale tuning curves
    STMF_RATES = (
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (12.0, 0.25), (12.0, -0.25),
        (4.0, 0.0),  (8.0, 0.0), (12.0, 0.0),
        (0.0, 0.25), (0.0, 0.5),
    )
    STMF_RATES_ADAPT = (
        (4.0, 0.25), (4.0, -0.25),
        (8.0, 0.25), (8.0, -0.25),
    )
    STMF_RATES_FAST = (
        (12.0, 0.25), (12.0, -0.25),
        (16.0, 0.25), (16.0, -0.25),
    )
    STMF_RATES_MID = (
        (4.0, 0.25), (4.0, -0.25),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5), (8.0, -0.5),
    )

    # --- Reservoir (GRAFT-v2) parameters ---
    RES_N = 80                        # units per family (4 families -> 320 raw dim)
    RES_DENSITY = 0.08
    RES_INPUT_SCALE = 1.5
    RES_LEAK_LOW = 0.02
    RES_LEAK_HIGH = 1.00
    RES_RHO_TANH = 0.95
    RES_RHO_RELU = 0.85
    RES_INPUT_LAGS = (0, 2, 5, 10)
    RES_SEED_TANH_FW = 12345
    RES_SEED_TANH_BW = 67890
    RES_SEED_RELU_FW = 13579
    RES_SEED_RELU_BW = 24680

    # --- TFS (analytic-signal) parameters (GRAFT-v1) ---
    TFS_N_BANDS = 24
    TFS_F_MIN = 80.0
    TFS_F_MAX = 6000.0
    TFS_INNER = 2048
    TFS_PHASE_LOCK_HZ = 1500.0
    TFS_PAC_NLOW = 0  # disabled in v1b
    TFS_PAC_NHI = 0
    TFS_PAC_HI_OFFSET = 12
    TFS_BATCH = 128

    def __init__(self, device: str = "cuda", n_filters: int = 20):
        self.device = device
        self.n_filters = n_filters
        self.fbs = [gammatone_powerbank(w, n_filters=n_filters).to(device)
                    for w in self.WIN_LIST]
        self.fb_sub = gammatone_powerbank(self.SUB_WIN, n_filters=n_filters).to(device)
        # Finer cochleagram (40 channels) on the LONG window for STMF only
        self.fb_fine = gammatone_powerbank(4096, n_filters=40).to(device)
        self.channel_idx = torch.arange(n_filters, dtype=torch.float32, device=device).view(1, -1)
        self.f0_cands = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32, device=device)
        self.long_freqs = torch.from_numpy(
            np.fft.rfftfreq(4096, d=1.0 / SR).astype(np.float32)).to(device)
        self._gabor_cache = {}
        self._stmf_cache = {}
        # === GRAFT-v1: analytic-signal filterbank for TFS block ===
        # 32-band gammatone on the long (4096) window; analytic-signal scaling
        # (2x for positive freqs, 1x at DC and Nyquist) so ifft gives the
        # analytic (complex) per-band signal.
        fcs_np = erb_space(self.TFS_F_MIN, self.TFS_F_MAX, self.TFS_N_BANDS)
        bw_np = (1.019 * erb(fcs_np)).astype(np.float32)
        freqs4096 = np.fft.rfftfreq(4096, d=1.0 / SR).astype(np.float32)
        H = (1.0 + ((freqs4096[:, None] - fcs_np[None, :]) / bw_np[None, :]) ** 2) ** (-4)
        H = (H / np.clip(H.max(axis=0, keepdims=True), 1e-12, None)).astype(np.float32)
        factor = np.full(H.shape[0], 2.0, dtype=np.float32)
        factor[0] = 1.0
        factor[-1] = 1.0
        H_analytic = H * factor[:, None]  # (n_freqs, n_bands)
        self.tfs_H = torch.from_numpy(H_analytic).to(device)              # (n_freqs, B)
        self.tfs_fcs = torch.from_numpy(fcs_np.astype(np.float32)).to(device)
        self.tfs_bw = torch.from_numpy(bw_np).to(device)
        self.tfs_n_freqs = freqs4096.size
        self.tfs_hann = torch.hann_window(4096, periodic=True, device=device)
        self.tfs_pl_mask = (self.tfs_fcs <= self.TFS_PHASE_LOCK_HZ).float()  # (B,)
        ti = torch.arange(self.TFS_INNER - 1, device=device, dtype=torch.float32)
        ti = ti - ti.mean()
        self.tfs_ti = ti

        # === GRAFT-v2: 4 fixed-random reservoirs (bidirectional tanh + relu) ===
        # Input dim = (n_filters) bands * 4 driving signals * len(LAGS).
        # We feed [log_e_long, cube_e_long, flux_pos, flux_neg], z-scored.
        self.res_dim_in = self.n_filters * 4 * len(self.RES_INPUT_LAGS)
        def _mk_res(N, rho, seed):
            W = make_reservoir_W(N, rho, self.RES_DENSITY, seed)
            Win = make_input_matrix(N, self.res_dim_in, self.RES_INPUT_SCALE, seed)
            leaks = make_leak_vector(N, self.RES_LEAK_LOW, self.RES_LEAK_HIGH, seed)
            return (torch.from_numpy(W).to(device),
                    torch.from_numpy(Win).to(device),
                    torch.from_numpy(leaks).to(device))
        self.res_tanh_fw = _mk_res(self.RES_N, self.RES_RHO_TANH, self.RES_SEED_TANH_FW)
        self.res_tanh_bw = _mk_res(self.RES_N, self.RES_RHO_TANH, self.RES_SEED_TANH_BW)
        self.res_relu_fw = _mk_res(self.RES_N, self.RES_RHO_RELU, self.RES_SEED_RELU_FW)
        self.res_relu_bw = _mk_res(self.RES_N, self.RES_RHO_RELU, self.RES_SEED_RELU_BW)
        # DCT-II basis (n_filters -> 13 cepstral coefs), closed-form
        k = np.arange(13)
        n = np.arange(n_filters)
        dct = np.cos(np.pi / n_filters * (n[:, None] + 0.5) * k[None, :])  # (F, 13)
        dct[:, 0] *= 1.0 / np.sqrt(2)
        dct = dct * np.sqrt(2.0 / n_filters)
        self.dct_basis = torch.from_numpy(dct.astype(np.float32)).to(device)

    def _stmf_envelope(self, log_e: torch.Tensor, omega_t: float, omega_f: float,
                        fs: float = 30.0) -> torch.Tensor:
        """Small Chi-Shamma 2-D Gabor envelope on (T, F) log-cochleagram.
        Cached kernel for speed; conv2d (small kernel)."""
        key = (round(omega_t, 3), round(omega_f, 3))
        if key not in self._stmf_cache:
            period_t = fs / max(abs(omega_t), 0.5)
            sigma_t = 1.0 * period_t / (2 * np.pi)
            Lt = max(1, int(np.ceil(3 * sigma_t)))
            sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi) * 1.0
            Lf = max(1, int(np.ceil(3 * sigma_f)))
            t = np.arange(-Lt, Lt + 1)
            f = np.arange(-Lf, Lf + 1)
            g = np.exp(-((t[:, None] ** 2) / (2 * sigma_t ** 2) +
                          (f[None, :] ** 2) / (2 * sigma_f ** 2)))
            phase = 2 * np.pi * omega_t * t[:, None] / fs + 2 * np.pi * omega_f * f[None, :]
            cos_k = (g * np.cos(phase)).astype(np.float32)
            sin_k = (g * np.sin(phase)).astype(np.float32)
            cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
            self._stmf_cache[key] = (
                torch.from_numpy(cos_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1),
                torch.from_numpy(sin_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1),
                Lt, Lf,
            )
        cos_k, sin_k, Lt, Lf = self._stmf_cache[key]
        xp = log_e.unsqueeze(0).unsqueeze(0)
        c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
        s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
        T, F = log_e.shape
        env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
        return env[:T, :F]

    def _dog_channels(self, x: torch.Tensor, sigma_c: float, sigma_s: float) -> torch.Tensor:
        """DoG conv across channel axis of (T, C)."""
        k = int(np.ceil(3 * sigma_s))
        coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
        g_c = torch.exp(-(coords ** 2) / (2 * sigma_c ** 2)); g_c = g_c / g_c.sum()
        g_s = torch.exp(-(coords ** 2) / (2 * sigma_s ** 2)); g_s = g_s / g_s.sum()
        dog = (g_c - g_s).view(1, 1, -1)
        return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)

    def _gabor_envelope(self, x: torch.Tensor, rate_hz: float,
                         fs: float = 30.0) -> torch.Tensor:
        """Gabor band-pass envelope on a (T, C) signal along time."""
        if rate_hz not in self._gabor_cache:
            period = fs / rate_hz
            sigma = 1.0 * period
            L = int(np.ceil(4 * sigma))
            t = np.arange(-L, L + 1)
            g = np.exp(-(t ** 2) / (2 * sigma ** 2))
            cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
            sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
            cos_k = cos_k - cos_k.mean()
            sin_k = sin_k - sin_k.mean()
            cos_k = cos_k / (np.linalg.norm(cos_k) + 1e-12)
            sin_k = sin_k / (np.linalg.norm(sin_k) + 1e-12)
            self._gabor_cache[rate_hz] = (
                torch.from_numpy(cos_k.astype(np.float32)).to(self.device),
                torch.from_numpy(sin_k.astype(np.float32)).to(self.device),
            )
        cos_k, sin_k = self._gabor_cache[rate_hz]
        L = cos_k.numel()
        pad = L // 2
        C = x.shape[1]
        xp = x.transpose(0, 1).unsqueeze(0)
        ck = cos_k.view(1, 1, -1).expand(C, 1, L).contiguous()
        sk = sin_k.view(1, 1, -1).expand(C, 1, L).contiguous()
        xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
        c = torch.nn.functional.conv1d(xp_p, ck, groups=C)
        s = torch.nn.functional.conv1d(xp_p, sk, groups=C)
        c = c[..., :x.shape[0]]
        s = s[..., :x.shape[0]]
        env = torch.sqrt(c * c + s * s + 1e-12)
        return env.squeeze(0).transpose(0, 1).contiguous()

    @torch.no_grad()
    def _tfs_block(self, x: torch.Tensor) -> torch.Tensor:
        """GRAFT-v1: TFS block (phase / temporal-fine-structure).

        Analytic signal of 32 bands on the 4096 window via Hilbert (FFT zero
        the negative half, ifft). Per inner window of INNER=2048 samples we
        compute: log instantaneous frequency, normalized IF deviation from
        band center, IF std (instantaneous bandwidth), FM up/down slope, PLV
        across band pairs (gap 1 + gap 2), and phase-amplitude coupling (PAC)
        between 8 low-band phases and 8 mid/high-band envelopes.

        Outputs (T, 288) = 7*32 + 8*8.  All linear-in-band reductions are
        masked to bands <= PHASE_LOCK_HZ where TFS is meaningful.
        """
        T, win = x.shape
        assert win == 4096, f"TFS block expects 4096-sample windows, got {win}"
        out_chunks = []
        i0 = (4096 - self.TFS_INNER) // 2
        pl = self.tfs_pl_mask.unsqueeze(0)  # (1, B)
        low_idx = torch.arange(0, self.TFS_PAC_NLOW, device=self.device)
        hi_idx = torch.arange(self.TFS_PAC_HI_OFFSET,
                              self.TFS_PAC_HI_OFFSET + self.TFS_PAC_NHI,
                              device=self.device)
        for s in range(0, T, self.TFS_BATCH):
            e = min(s + self.TFS_BATCH, T)
            xb = x[s:e]  # (Tb, 4096)
            Xf = torch.fft.rfft(xb * self.tfs_hann, n=4096, dim=1)  # (Tb, n_freqs)
            # per-band spectrum * analytic-scaled filter
            Yp = Xf.unsqueeze(1) * self.tfs_H.transpose(0, 1).unsqueeze(0)  # (Tb, B, n_freqs)
            Zfull = torch.zeros((Yp.shape[0], self.TFS_N_BANDS, 4096),
                                dtype=torch.complex64, device=self.device)
            Zfull[..., :self.tfs_n_freqs] = Yp
            z = torch.fft.ifft(Zfull, n=4096, dim=2)              # (Tb, B, 4096) complex
            zc = z[..., i0:i0 + self.TFS_INNER]                   # (Tb, B, INNER)
            amp = zc.abs()                                         # (Tb, B, INNER)

            # IF via phase-difference (in Hz)
            zr = zc[..., 1:] * zc[..., :-1].conj()
            ifreq = torch.angle(zr) * (SR / (2.0 * np.pi))         # (Tb, B, INNER-1)
            w = amp[..., 1:]
            wsum = w.sum(dim=2).clamp(min=1e-10)
            if_mean = (w * ifreq).sum(dim=2) / wsum
            df = ifreq - if_mean.unsqueeze(2)
            if_var = (w * df ** 2).sum(dim=2) / wsum
            if_std = torch.sqrt(if_var.clamp(min=0.0) + 1e-6)

            # IF deviation from filter center, normalized to band bandwidth
            if_dev = if_mean - self.tfs_fcs.unsqueeze(0)
            if_dev_norm = if_dev / self.tfs_bw.unsqueeze(0)
            log_if = torch.log(if_mean.clamp(min=20.0))

            # FM slope (weighted regression of ifreq on time idx ti)
            num = (w * df * self.tfs_ti).sum(dim=2)
            denom = (w * self.tfs_ti ** 2).sum(dim=2).clamp(min=1e-6)
            fm_slope = num / denom                                  # Hz/sample
            fm_hz_s = fm_slope * SR
            fm_feat = torch.sign(fm_hz_s) * torch.log1p(fm_hz_s.abs())
            fm_up = fm_feat.clamp(min=0.0)
            fm_dn = (-fm_feat).clamp(min=0.0)

            # Inter-band PLV (cross-band phase coherence, gap 1 and gap 2)
            unit = zc / zc.abs().clamp(min=1e-8)                   # (Tb, B, INNER) complex
            plv1 = (unit[:, :-1, :] * unit[:, 1:, :].conj()).mean(dim=2).abs()
            plv2 = (unit[:, :-2, :] * unit[:, 2:, :].conj()).mean(dim=2).abs()
            plv1 = torch.cat([plv1, plv1[:, -1:]], dim=1)          # pad to B
            plv2 = torch.cat([plv2, plv2[:, -2:]], dim=1)

            # Apply phase-locking mask (zero out non-phase-locked bands)
            log_if_pl    = log_if * pl
            if_dev_pl    = if_dev_norm * pl
            if_std_pl    = if_std * pl
            fm_up_pl     = fm_up * pl
            fm_dn_pl     = fm_dn * pl
            plv1_pl      = plv1 * pl
            plv2_pl      = plv2 * pl

            # PAC disabled in v1b (was 8x8=64 dim)
            pac_parts = []
            if self.TFS_PAC_NLOW > 0:
                phi_low = zc[:, low_idx, :] / zc[:, low_idx, :].abs().clamp(min=1e-8)
                env_hi = amp[:, hi_idx, :]
                env_hi_norm = env_hi.sum(dim=2).clamp(min=1e-8)
                num_complex = torch.einsum('tli,thi->tlh', phi_low, env_hi.to(torch.complex64))
                pac = num_complex.abs() / env_hi_norm.unsqueeze(1)
                pac_parts.append(pac.reshape(pac.shape[0], -1))

            # v1c: keep only the most TFS-unique pieces (fm slopes + cross-band PLV).
            # log_if / if_dev / if_std were largely redundant with envelope info; drop.
            block_parts = [fm_up_pl, fm_dn_pl, plv1_pl, plv2_pl] + pac_parts
            block = torch.cat(block_parts, dim=1)
            out_chunks.append(block)
            del z, zc, amp, zr, ifreq, w, df, unit, plv1, plv2, Xf, Yp, Zfull
        tfs = torch.cat(out_chunks, dim=0)                          # (T, 288)
        # Replace any inf/nan from edge cases with 0 (per-chunk z-score will
        # handle scale; we just must not pass non-finite into ridge).
        tfs = torch.nan_to_num(tfs, nan=0.0, posinf=0.0, neginf=0.0)
        return tfs

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []
        log_es = []
        powers = []
        for w, fb in zip(self.WIN_LIST, self.fbs):
            if w == 4096:
                xw = x
            else:
                start = (4096 - w) // 2
                xw = x[:, start:start + w]
            p = power_spectrum(xw, w, self.device)
            powers.append(p)
            log_e = torch.log(torch.clamp(p @ fb, min=1e-10))
            log_es.append(log_e)
            feats.append(log_e)
            for tau in self.TAUS:
                feats.append(log_e - leaky_lowpass(log_e, tau_frames=tau))

        # [LEAN-v10] removed: sub-window AM stats (am_depth, am_range, am_onset, 60 dim)

        # [LEAN-v11] removed: Multi-tau loudness + centroid per window (36 dim)

        # ============== HIGH-LEVEL FEATURES from long-window power spectrum ==============
        power_long = powers[-1]  # (T, n_freqs)
        log_e_long = log_es[-1]   # (T, F)

        # 1. Harmonic-comb pitch salience: (T, 7)
        ps = harmonic_comb_salience(power_long, 4096, self.f0_cands, n_harmonics=8)
        feats.append(ps)
        # Best pitch and its strength (max over candidates) + multi-tau dynamic
        best_strength, best_idx = ps.max(dim=1, keepdim=True)
        best_f0 = self.f0_cands[best_idx.squeeze(1)].unsqueeze(1)
        log_f0 = torch.log(best_f0.clamp(min=20.0))
        feats.append(best_strength)
        feats.append(log_f0)
        for tau in (4.5, 15.0, 60.0):
            feats.append(log_f0 - leaky_lowpass(log_f0, tau_frames=tau))
            feats.append(best_strength - leaky_lowpass(best_strength, tau_frames=tau))

        # 2. Spectral moments (4) on long-window power spectrum
        mom = spectral_moments(power_long, self.long_freqs)
        feats.append(mom)

        # 3. Spectral flatness (1)
        flat = spectral_flatness(power_long)
        feats.append(flat)

        # 4. Spectral rolloff (2): 85%, 95%
        roll = spectral_rolloff(power_long, self.long_freqs)
        feats.append(roll)

        # 5. HNR (2): autocorr peak + HNR_db
        hnr = hnr_from_acf(power_long, 4096)
        feats.append(hnr)
        # [LEAN-v14] removed: HNR multi-tau dynamics (6 dim)

        # 6. Phonetic templates (5)
        phon = phonetic_templates(log_e_long, self.n_filters)
        feats.append(phon)
        # [LEAN-v13] removed: phon multi-tau dynamics (15 dim)

        # 7. Cross-band comodulation (6) on long log_e — needed by mod_signals
        comod = cross_band_comodulation(log_e_long)
        # [LEAN-v15] removed comod from feats (still used in mod_signals)

        # 8. Spectral flux (broadband pos/neg/abs per window). PROVEN +0.002.
        flux_feats = []
        for log_e_w in log_es:
            d = torch.zeros_like(log_e_w)
            d[1:] = log_e_w[1:] - log_e_w[:-1]
            flux_pos = d.clamp(min=0.0).sum(dim=1, keepdim=True)
            flux_neg = (-d).clamp(min=0.0).sum(dim=1, keepdim=True)
            flux_abs = d.abs().sum(dim=1, keepdim=True)
            flux_feats.append(torch.cat([flux_pos, flux_neg, flux_abs], dim=1))
        flux_all = torch.cat(flux_feats, dim=1)
        feats.append(flux_all)
        for tau in self.HL_TAUS:
            feats.append(flux_all - leaky_lowpass(flux_all, tau_frames=tau))

        # 9. MODULATION FILTERBANK on key 1-D signals at prosodic/syllabic rates.
        long_loud = log_es[-1].mean(dim=1, keepdim=True)
        e_w = torch.softmax(log_es[-1], dim=1)
        long_centroid = (e_w * self.channel_idx).sum(dim=1, keepdim=True)
        mod_signals = [long_loud, long_centroid, hnr, best_strength, log_f0,
                        flux_all, phon, mom, flat, roll, comod]
        for rate in (2.0, 4.0, 6.0, 10.0):
            for sig in mod_signals:
                feats.append(self._gabor_envelope(sig, rate))

        # 10. Chi-Shamma 2-D STMF on the long log-cochleagram (T, F=20).
        stmf_means = []
        for (omega_t, omega_f) in self.STMF_RATES:
            stmf_e = self._stmf_envelope(log_e_long, omega_t, omega_f)
            feats.append(stmf_e)
            stmf_means.append(stmf_e.mean(dim=1, keepdim=True))
        stmf_mean_all = torch.cat(stmf_means, dim=1)
        feats.append(self._gabor_envelope(stmf_mean_all, 2.0))
        # [LEAN-v1] removed: STMF on lateral-inhibited cochleagram (160 dim)

        # 10b. FORMANT-LIKE peak tracking (2 formants — the best config).
        nF = self.n_filters
        f1_band = log_e_long[:, :nF // 2]
        f2_band = log_e_long[:, nF // 2:]
        w1 = torch.softmax(f1_band * 2.0, dim=1)
        w2 = torch.softmax(f2_band * 2.0, dim=1)
        ch1 = torch.arange(nF // 2, device=log_e_long.device,
                            dtype=log_e_long.dtype).view(1, -1)
        ch2 = torch.arange(nF // 2, device=log_e_long.device,
                            dtype=log_e_long.dtype).view(1, -1) + nF // 2
        f1_loc = (w1 * ch1).sum(dim=1, keepdim=True)
        f2_loc = (w2 * ch2).sum(dim=1, keepdim=True)
        f1_amp = (w1 * f1_band).sum(dim=1, keepdim=True)
        f2_amp = (w2 * f2_band).sum(dim=1, keepdim=True)
        f12_diff = f2_loc - f1_loc
        formants = torch.cat([f1_loc, f2_loc, f12_diff, f1_amp, f2_amp,
                                f2_amp - f1_amp], dim=1)
        feats.append(formants)
        for tau in (4.5, 15.0, 60.0):
            feats.append(formants - leaky_lowpass(formants, tau_frames=tau))
        fmt_loc = torch.cat([f1_loc, f2_loc, f12_diff], dim=1)
        fmt_d = torch.zeros_like(fmt_loc)
        fmt_d[1:] = fmt_loc[1:] - fmt_loc[:-1]
        feats.append(torch.cat([fmt_d, fmt_d.clamp(min=0),
                                 (-fmt_d).clamp(min=0)], dim=1))

        # [LEAN-v4] removed: CUBE-ROOT cochleagrams (180 dim)

        # 10b3. PER-CHANNEL TEMPORAL VARIANCE (smoothed) — captures stable vs
        # modulated bands. On LONG window log + cube cochleagrams.
        # [LEAN-v7] removed: per-channel log_e variance (40 dim)
        # [LEAN-v8] removed: per-channel cube variance (40 dim)
        cube_long_var = torch.pow(power_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)

        # 10b4. PER-CHANNEL "modulation depth": HWR of high-pass cochleagram,
        # low-passed to get the slow envelope of fast modulation per band.
        for hp_tau, lp_tau in ((4.5, 15.0), (1.5, 15.0)):
            hp = log_e_long - leaky_lowpass(log_e_long, tau_frames=hp_tau)
            feats.append(leaky_lowpass(hp.abs(), tau_frames=lp_tau))

        # 10b5. SLOW TONIC smoothed cochleagrams (state/context at prosodic scale).
        # Causal + acausal slow tonic — past AND future per-band loudness state.
        for i, (w, fb) in enumerate(zip(self.WIN_LIST, self.fbs)):
            le_w = log_es[i]
            cube_w = torch.pow(powers[i] @ fb + 1e-10, 1.0 / 3.0)
            for tau in (60.0, 150.0):
                feats.append(leaky_lowpass(le_w, tau_frames=tau))
                feats.append(leaky_lowpass(cube_w, tau_frames=tau))
            feats.append(acausal_leaky_lowpass(le_w, tau_frames=60.0))

        # 10b6. FUTURE + PAST DELTA per band: log_e[t±k] - log_e[t].
        # Provides ridge explicit per-band lookahead and lookback change signals.
        for k in (3, 6, 12):
            fut_k = torch.cat([log_e_long[k:], log_e_long[-1:].expand(k, -1)], dim=0)
            future_delta = fut_k - log_e_long
            feats.append(future_delta)
            feats.append(future_delta.clamp(min=0.0))
            feats.append((-future_delta).clamp(min=0.0))
            past_k = torch.cat([log_e_long[:1].expand(k, -1), log_e_long[:-k]], dim=0)
            past_delta = log_e_long - past_k
            feats.append(past_delta)
        # Future delta on CUBE cochleagram, lag 6 — different gain regime.
        fut_c = torch.cat([cube_long_var[6:], cube_long_var[-1:].expand(6, -1)], dim=0)
        feats.append(fut_c - cube_long_var)


        # [LEAN-v16] removed: per-formant BANDWIDTH block (16 dim)


        # 11. F0 dynamics (slope, curvature, on/off) + HNR onset
        f0_d1 = torch.zeros_like(log_f0)
        f0_d1[1:] = log_f0[1:] - log_f0[:-1]
        f0_d2 = torch.zeros_like(log_f0)
        f0_d2[1:] = f0_d1[1:] - f0_d1[:-1]
        feats.append(torch.cat([f0_d1, f0_d2,
                                  f0_d1.clamp(min=0), (-f0_d1).clamp(min=0)], dim=1))
        # HNR derivative (voicing onset)
        hnr_db = hnr[:, 1:2]
        hnr_d = torch.zeros_like(hnr_db)
        hnr_d[1:] = hnr_db[1:] - hnr_db[:-1]
        feats.append(torch.cat([hnr_d, hnr_d.clamp(min=0), (-hnr_d).clamp(min=0)], dim=1))

        # [LEAN-v2] removed: PER-BAND ENVELOPE AUTOCORRELOGRAM (127 dim)

        # GRAFT-v1 (TFS) REVERTED: v1a=288 dim 0.0745, v1b=168 dim 0.0753,
        # v1c=96 dim 0.0756; all below baseline 0.0760 (best -0.0004 to -0.0015).
        # _tfs_block / TFS_* config remain in code as reference but are unused.

        # GRAFT-v2 (reservoir) REVERTED: v2/v2a OOM'd at N=256/200; v2b (N=128,
        # 512 dim) = 0.0744 (train 0.1001, overfit); v2c (N=80, 320 dim) = 0.0758
        # (Δ=-0.0002, within noise but block too large for small-bonus). The
        # reservoir reliably raises train_corr but does not transfer to test.

        # === GRAFT-v3: DIVISIVE-NORMALIZATION block ===
        # Genuine nonlinearity LEAN lacks: ratios e / (sigma + pool) and
        # log(sigma + pool). Built from long-window cochleagram e_long (T, 20):
        #   - 2x saturating rate-level e / (e + half_sat)          ->  40 dim
        #   - 4x DN cochleagrams  log(e) - log(sigma + pool_k)     ->  80 dim
        #   - 4x context magnitude log(sigma + pool_k)             ->  80 dim
        #   - 2x DRNL gain (theta / (theta + pool))^(1-c)          ->  40 dim
        # 4 pool flavours: narrow-freq, wide-freq, fast-causal-time, slow-time.
        e_long = (power_long @ self.fbs[-1]).clamp(min=1e-10)  # (T, 20)
        # Frequency pools via DoG kernels: width 3 and 8 channels (Gaussian on
        # channel axis, normalized to sum=1). Use existing _dog_channels by
        # supplying matched (sigma_c, sigma_s) -- but we want PURE Gaussian
        # smoothing, not DoG. Implement inline.
        def _gauss_freq_pool(x, sigma):
            k = int(np.ceil(3 * sigma))
            coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
            ker = g.view(1, 1, -1)
            xp = x.unsqueeze(1)
            out = torch.nn.functional.conv1d(xp, ker, padding=k).squeeze(1)
            return out
        pool_narrow = _gauss_freq_pool(e_long, sigma=3.0)
        pool_wide   = _gauss_freq_pool(e_long, sigma=8.0)
        pool_fast   = leaky_lowpass(e_long, tau_frames=4.5)
        pool_slow   = leaky_lowpass(e_long, tau_frames=60.0)
        dn_pools = [pool_narrow, pool_wide, pool_fast, pool_slow]
        # Per-chunk sigma = 10% of pool median (Carandini-Heeger flavour).
        log_e_long_clip = torch.log(e_long)
        dn_maps = []
        ctx_maps = []
        for pool in dn_pools:
            sigma_d = (0.1 * pool.median()).clamp(min=1e-8)
            denom = (sigma_d + pool).clamp(min=1e-12)
            dn_maps.append(log_e_long_clip - torch.log(denom))   # (T, 20)
            ctx_maps.append(torch.log(denom))                     # (T, 20)
        feats.append(torch.cat(dn_maps, dim=1))                    # 80 dim
        # v3b: drop ctx_maps (linearly recoverable from logE+log_pool) + DRNL
        # gain (redundant with DN). Keep just DN cochleagrams + saturation.
        # Saturating rate-level: e / (e + half_sat). Two half-sat thresholds.
        e_med = e_long.median().clamp(min=1e-8)
        sat1 = e_long / (e_long + 0.3 * e_med)                     # (T, 20)
        sat2 = e_long / (e_long + 3.0 * e_med)                     # (T, 20)
        feats.append(torch.cat([sat1, sat2], dim=1))               # 40 dim

        # === GRAFT-v5c (products) REVERTED in v6-stacked-clean ===
        # v5a (54 dim) = 0.0762, v5b (39 dim) = 0.0763, v5c (15 dim) = 0.0763.
        # All <= best v4b (0.0764). Within-noise but not load-bearing; dropping
        # for a cleaner stacked baseline before lightly tuning v4b.

        # === GRAFT-v4b: cross-band running correlations only ===
        # The most uniquely-texture piece; running products LEAN lacks.
        tau_tex = 15.0
        mu_tex = leaky_lowpass(log_e_long, tau_tex)
        d_tex = log_e_long - mu_tex
        var_tex = leaky_lowpass(d_tex * d_tex, tau_tex).clamp(min=1e-10)
        nF = log_e_long.shape[1]
        for offset in (1, 2, 4):
            mu_a = mu_tex[:, :nF - offset]; mu_b = mu_tex[:, offset:]
            da = log_e_long[:, :nF - offset] - mu_a
            db = log_e_long[:, offset:] - mu_b
            cov_ab = leaky_lowpass(da * db, tau_tex)
            va = var_tex[:, :nF - offset]; vb = var_tex[:, offset:]
            rho = cov_ab / torch.sqrt(va * vb + 1e-10)
            feats.append(rho.clamp(-1.0, 1.0))                       # (nF - offset) dim

        # === GRAFT-v10 tune: TIME-LAGGED cross-band running correlations ===
        # v4b above does SAME-TIME, different-band correlations. This block
        # adds DIFFERENT-TIME, different-band correlations: rho(band_i(t),
        # band_j(t-lag)) — asymmetric temporal-spectral coupling (forward-
        # masking flavour). LEAN has per-band deltas + same-time products but
        # no cross-band TIME-LAGGED running products. Small block under the
        # 150-dim small-block threshold; one freq-offset (2) x two time-lags
        # (3, 9 frames) on the 20-band cochleagram = 2*(20-2) = 36 dim.
        for time_lag in (3, 9):
            freq_off = 2
            a = log_e_long[time_lag:, :nF - freq_off]                 # band i at t
            b = log_e_long[:-time_lag, freq_off:]                     # band j at t-lag
            mu_a_t = mu_tex[time_lag:, :nF - freq_off]
            mu_b_t = mu_tex[:-time_lag, freq_off:]
            va_t = var_tex[time_lag:, :nF - freq_off]
            vb_t = var_tex[:-time_lag, freq_off:]
            da_t = a - mu_a_t
            db_t = b - mu_b_t
            cov_t = leaky_lowpass(da_t * db_t, tau_tex)
            rho_t = cov_t / torch.sqrt(va_t * vb_t + 1e-10)
            rho_t = rho_t.clamp(-1.0, 1.0)
            # Pad front with zeros to restore original T length.
            pad = torch.zeros(time_lag, rho_t.shape[1], device=rho_t.device, dtype=rho_t.dtype)
            feats.append(torch.cat([pad, rho_t], dim=0))               # (T, nF-freq_off) = 18 dim

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "GRAFT-v10-tune-tlag"
model_description = (
    "GRAFT v10 light-tune on v6-stacked-clean: ADD time-lagged cross-band "
    "running correlations rho(band_i(t), band_j(t-lag)), freq offset=2, "
    "time lags (3, 9 frames), tau=15 EMA. 36 raw dim (2*(20-2)). Captures "
    "asymmetric temporal-spectral coupling (forward-masking flavour) — "
    "distinct from v4b same-time freq-offset correlations and from LEAN "
    "per-band deltas. v6 stacked-clean baseline = 0.0764; v7 offset=8 = "
    "0.0764 tied, v8 sat3 = 0.0763, v9 multi-tau = 0.0759 all reverted."
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
        print("cuda not available — falling back to cpu")
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
