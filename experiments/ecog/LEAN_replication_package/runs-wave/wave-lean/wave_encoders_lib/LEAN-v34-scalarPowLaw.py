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

# ---------------------------------------------------------------------------
# GPU-FAULT WORKAROUND: this node's A100 is in a "Pending: Unknown Error" state
# (nvidia-smi shows ERR! for power/temp; cuInit returns CUDA_ERROR_NO_DEVICE).
# The CPU fallback in src/encoding.py uses np.linalg.svd on the full (N, p)
# design matrix, which for our N~350k p~6300 settings:
#   - allocates a (N, p) U matrix at ~17 GB float64,
#   - peaks well beyond the 192 GB SLURM cap (we got OOM-killed at 200 GB),
#   - takes ~50 min on CPU for one bootstrap split (5+1 needed).
# Monkey-patch encoding.bootstrap_ridge with a torch-CPU XtX-eigh path that:
#   - never materializes (N, p) in float64 (XtX is just p x p ~ 320 MB),
#   - is mathematically equivalent to the SVD path (same closed-form ridge),
#   - reuses the GPU code's exact algorithm with device='cpu'.
# This is transparent to the encoder and reproduces ridge results within fp32.
# ---------------------------------------------------------------------------
from src import encoding as _encoding


def _cpu_accum_XtX_XtY(stim, resp, dev, block=None):
    p = stim.shape[1]
    if block is None:
        block = max(10_000, min(200_000, int(3.0e9 / (p * 8))))
    XtX = torch.zeros(p, p, dtype=torch.float64, device=dev)
    XtY = torch.zeros(p, resp.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, stim.shape[0], block):
        Xb = torch.from_numpy(np.ascontiguousarray(stim[i:i + block])).to(dev, torch.float64)
        Yb = torch.from_numpy(np.ascontiguousarray(resp[i:i + block])).to(dev, torch.float64)
        XtX += Xb.T @ Xb
        XtY += Xb.T @ Yb
        del Xb, Yb
    return XtX, XtY


def _cpu_eig_good(XtX, singcutoff):
    Lam, V = torch.linalg.eigh(XtX)
    S = torch.sqrt(torch.clamp(Lam, min=0.0))
    keep = S > singcutoff
    return V[:, keep], Lam[keep]


def _cpu_zs_t(v):
    s = v.std(0)
    s[s == 0] = 1.0
    return (v - v.mean(0)) / s


def _bootstrap_ridge_cpu_eigh(stim_train, resp_train, stim_test, resp_test, alphas,
                               nboots=5, chunklen=40, nchunks=20, singcutoff=1e-10):
    dev = "cpu"
    nresp, nvox = resp_train.shape
    nalphas = len(alphas)
    a2 = torch.tensor(np.asarray(alphas) ** 2, dtype=torch.float64, device=dev)

    XtX_full, XtY_full = _cpu_accum_XtX_XtY(stim_train, resp_train, dev)
    splits = _encoding._temporal_chunk_splits(nboots, nresp, chunklen, nchunks)
    all_corrs = torch.zeros(nalphas, nvox, dtype=torch.float64, device=dev)
    for train_idx, tune_idx in splits:
        Xtu = torch.from_numpy(np.ascontiguousarray(stim_train[tune_idx])).to(dev, torch.float64)
        Ytu = torch.from_numpy(np.ascontiguousarray(resp_train[tune_idx])).to(dev, torch.float64)
        XtX_sub = XtX_full - Xtu.T @ Xtu
        XtY_sub = XtY_full - Xtu.T @ Ytu
        V, Lam = _cpu_eig_good(XtX_sub, singcutoff)
        A = Xtu @ V
        B = V.T @ XtY_sub
        zYtu = _cpu_zs_t(Ytu)
        for ai in range(nalphas):
            pred = A @ (B / (Lam + a2[ai]).unsqueeze(1))
            all_corrs[ai] += torch.nan_to_num((zYtu * _cpu_zs_t(pred)).mean(0))
        del Xtu, Ytu, XtX_sub, XtY_sub, V, Lam, A, B, zYtu

    best_alpha_idx = torch.argmax(all_corrs, dim=0).cpu().numpy()
    alphas_best = np.asarray(alphas)[best_alpha_idx]

    V, Lam = _cpu_eig_good(XtX_full, singcutoff)
    B = V.T @ XtY_full
    wt = torch.zeros(stim_train.shape[1], nvox, dtype=torch.float64, device=dev)
    for ua in np.unique(alphas_best):
        sel = np.nonzero(alphas_best == ua)[0]
        scaled = B[:, sel] / (Lam + float(ua) ** 2).unsqueeze(1)
        wt[:, sel] = V @ scaled

    Xte = torch.from_numpy(np.ascontiguousarray(stim_test)).to(dev, torch.float64)
    Yte = torch.from_numpy(np.ascontiguousarray(resp_test)).to(dev, torch.float64)
    pred = torch.nan_to_num(Xte @ wt)
    corrs = torch.nan_to_num((_cpu_zs_t(Yte) * _cpu_zs_t(pred)).mean(0))
    return wt.cpu().numpy(), corrs.cpu().numpy(), alphas_best


if not torch.cuda.is_available():
    _encoding.bootstrap_ridge = _bootstrap_ridge_cpu_eigh
    print("[wave_encoder] GPU unavailable -- monkey-patched encoding.bootstrap_ridge "
          "with CPU XtX-eigh path (memory-safe).")


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR = "/path/to/scratch/feat_cache_wave"
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


def power_law_smooth(x: torch.Tensor, beta: float = 0.6, t0: float = 1.0,
                     L: int = 180) -> torch.Tensor:
    """Causal scale-free (power-law) smoothing: kernel k[t] = (t+t0)^-beta,
    t=0..L-1, normalized to sum=1. No single characteristic tau."""
    t = torch.arange(L, device=x.device, dtype=x.dtype)
    k = (t + t0).pow(-beta)
    k = k / k.sum()
    K = L
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = k.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


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

        # ============== [LEAN-v28] LITERATURE-BACKED SAFE SMALL BLOCK ==============
        # All NON-DIVIDING (additive) versions of v27's blocks. v27 used z-score
        # division by a sigma clamped at 1e-3, which blew up to ~thousands in
        # transient voiced frames and overfit (train 0.30 / test 0.041).
        # Voicing gate from harmonic-comb strength (sigmoid-ish).
        v_gate = torch.sigmoid((best_strength - 1.0) * 4.0)  # (T,1) in [0,1]

        # (a) RELATIVE pitch (SAFE, additive): rel = (log_f0 - voiced_running_mean) * v_gate.
        #     No division by sigma. Captures relative-pitch contour without blow-up.
        #     Chang 2017: STG tracks RELATIVE pitch. Taus 90, 200 frames.
        rel_pitch_blocks = []
        for tau in (90.0, 200.0):
            mu_num = leaky_lowpass(log_f0 * v_gate, tau_frames=tau)
            mu_den = leaky_lowpass(v_gate, tau_frames=tau).clamp(min=0.05)
            mu = mu_num / mu_den  # safe: clamp 0.05 keeps mu in log-Hz range
            rel = (log_f0 - mu) * v_gate
            drel = torch.zeros_like(rel); drel[1:] = rel[1:] - rel[:-1]
            rel_pitch_blocks.append(torch.cat([
                rel, drel.clamp(min=0), (-drel).clamp(min=0)], dim=1))
        feats.append(torch.cat(rel_pitch_blocks, dim=1))  # 6 dim

        # (b) CONTRASTIVE FORMANTS: f1_loc / f2_loc / f12_diff minus long mean
        #     (Sjerps/Chang 2019). 3 features x 2 taus = 6 dim.
        fmt_base = torch.cat([f1_loc, f2_loc, f12_diff], dim=1)
        for tau in (90.0, 200.0):
            feats.append(fmt_base - leaky_lowpass(fmt_base, tau_frames=tau))  # 6 dim total

        # (c) peakRate LANDMARK: discrete sparse impulse train at local maxima
        #     of d(long_loud)/dt; Oganian/Chang 2019. magnitude-weighted.
        loud_d = torch.zeros_like(long_loud)
        loud_d[1:] = long_loud[1:] - long_loud[:-1]
        d_prev = torch.cat([loud_d[:1], loud_d[:-1]], dim=0)
        d_next = torch.cat([loud_d[1:], loud_d[-1:]], dim=0)
        is_peak = (loud_d > d_prev) & (loud_d > d_next) & (loud_d > 0)
        peakRate = torch.where(is_peak, loud_d, torch.zeros_like(loud_d))
        is_trough = (loud_d < d_prev) & (loud_d < d_next) & (loud_d < 0)
        peakTrough = torch.where(is_trough, -loud_d, torch.zeros_like(loud_d))
        loud_baseline = leaky_lowpass(long_loud, tau_frames=30.0)
        is_silent = (long_loud < loud_baseline - 1.0).float()
        recent_silence = leaky_lowpass(is_silent, tau_frames=6.0)
        silence_onset = peakRate * (recent_silence > 0.3).float()
        peakRate_sm = leaky_lowpass(peakRate, tau_frames=4.5)
        feats.append(torch.cat([peakRate, peakTrough, silence_onset,
                                  peakRate_sm], dim=1))  # 4 dim

        # ============== [LEAN-v29] SYLLABLE-RATE CONTEXT ==============
        # Encode the LOCAL RATE of peakRate / silence-onset events at multiple
        # taus. Captures speech rhythm / syllable rate, distinct from the sparse
        # peakRate events themselves and from continuous loudness. Cheap (+5 dim).
        peakRate_evt = (peakRate > 0).to(torch.float32)
        sil_onset_evt = (silence_onset > 0).to(torch.float32)
        rate_blocks = []
        for tau in (15.0, 30.0, 60.0):
            rate_blocks.append(leaky_lowpass(peakRate_evt, tau_frames=tau))
        for tau in (30.0, 60.0):
            rate_blocks.append(leaky_lowpass(sil_onset_evt, tau_frames=tau))
        feats.append(torch.cat(rate_blocks, dim=1))  # 5 dim

        # ============== [LEAN-v30] VOICING-GATED NONLINEAR INTERACTIONS ==============
        # Small focused set of voiced-gated products that ridge (a linear model)
        # cannot construct from existing features. Targets known STG mechanisms:
        # voiced pitch-change neurons, periodicity-coupled onsets, voiced formant
        # tracking. v21 tried 34-dim variant +0.0000; this is a tighter 8-dim
        # subset with peakRate*hnr (a novel new interaction).
        d_log_f0 = torch.zeros_like(log_f0)
        d_log_f0[1:] = log_f0[1:] - log_f0[:-1]
        hnr_db_v = hnr[:, 1:2]
        feats.append(torch.cat([
            d_log_f0 * v_gate,                                  # voiced pitch change
            (d_log_f0.clamp(min=0)) * v_gate,                   # voiced rising pitch
            ((-d_log_f0).clamp(min=0)) * v_gate,                # voiced falling pitch
            peakRate * hnr_db_v,                                 # onset × periodicity
            peakRate * v_gate,                                   # voiced onset (syllable nuclei)
            peakRate * (1.0 - v_gate),                           # unvoiced onset (consonant)
            best_strength * long_loud,                           # voicing × loudness
            f12_diff * v_gate,                                   # voiced F2-F1 contour
        ], dim=1))  # 8 dim

        # ============== [LEAN-v31] BAND-SEGMENTED SPECTRAL FLUX ==============
        # The existing flux_feats sum ReLU(d_log_e) ACROSS ALL BANDS (broadband).
        # Per-band log_e is in features but ReLU is nonlinear so band-restricted
        # ReLU sums add genuinely new info. Targets phonetic onset/offset spectra:
        # high-band positive flux = fricative/burst onset, low-band = voiced
        # onset/release. Adds 9 dim (3 bands x {pos,neg,abs}) + 3 lag-3 = 12 dim.
        nF31 = log_e_long.shape[1]
        lo_sl = slice(0, nF31 // 3)               # lo: 0..6
        mid_sl = slice(nF31 // 3, 2 * nF31 // 3)  # mid: 7..13
        hi_sl = slice(2 * nF31 // 3, nF31)        # hi: 14..19
        d1 = torch.zeros_like(log_e_long)
        d1[1:] = log_e_long[1:] - log_e_long[:-1]
        d3 = torch.zeros_like(log_e_long)
        d3[3:] = log_e_long[3:] - log_e_long[:-3]
        band_flux_blocks = []
        for sl in (lo_sl, mid_sl, hi_sl):
            band = d1[:, sl]
            band_flux_blocks.append(band.clamp(min=0).sum(dim=1, keepdim=True))   # pos
            band_flux_blocks.append((-band).clamp(min=0).sum(dim=1, keepdim=True))  # neg
            band_flux_blocks.append(band.abs().sum(dim=1, keepdim=True))           # abs
            band3 = d3[:, sl]
            band_flux_blocks.append(band3.clamp(min=0).sum(dim=1, keepdim=True))  # pos lag3
        feats.append(torch.cat(band_flux_blocks, dim=1))  # 12 dim

        # ============== [LEAN-v32] MULTI-DURATION SILENCE-GATED ONSETS ==============
        # The single silence_onset uses tau~6 frames (~200ms). Hamilton/Chang
        # posterior-STG onset zone responds to PHRASE boundaries, which are
        # longer silences (>300ms, ~600ms). Add 3 silence-duration scales:
        # 130/300/600 ms with separate onset detectors. Adds 6 dim.
        for sil_tau in (4.0, 9.0, 18.0):
            recent_sil_d = leaky_lowpass(is_silent, tau_frames=sil_tau)
            sil_onset_d = peakRate * (recent_sil_d > 0.3).float()
            sil_onset_d_sm = leaky_lowpass(sil_onset_d, tau_frames=4.5)
            feats.append(torch.cat([sil_onset_d, sil_onset_d_sm], dim=1))  # 2 dim each

        # ============== [LEAN-v33] PER-BAND-GROUP PEAK-RATE ==============
        # Broadband peakRate (v28) is computed on long_loud = mean over all 20
        # bands. STG neurons are spectrotopically organized — band-specific
        # onsets carry phoneme info (high-band onsets = fricatives/sibilants,
        # low-band = voiced stops). Compute peakRate-like landmarks per band
        # group (lo/mid/hi) with their smoothed copies. Adds 7 dim.
        for sl in (lo_sl, mid_sl, hi_sl):
            band_loud = log_e_long[:, sl].mean(dim=1, keepdim=True)
            band_d = torch.zeros_like(band_loud)
            band_d[1:] = band_loud[1:] - band_loud[:-1]
            bd_prev = torch.cat([band_d[:1], band_d[:-1]], dim=0)
            bd_next = torch.cat([band_d[1:], band_d[-1:]], dim=0)
            is_peak_b = (band_d > bd_prev) & (band_d > bd_next) & (band_d > 0)
            band_peakRate = torch.where(is_peak_b, band_d, torch.zeros_like(band_d))
            band_pr_sm = leaky_lowpass(band_peakRate, tau_frames=4.5)
            feats.append(torch.cat([band_peakRate, band_pr_sm], dim=1))  # 2 dim per band x 3 = 6
        # broadband 2nd-order peakRate (acceleration onset)
        loud_d2 = torch.zeros_like(loud_d)
        loud_d2[1:] = loud_d[1:] - loud_d[:-1]
        d2_prev = torch.cat([loud_d2[:1], loud_d2[:-1]], dim=0)
        d2_next = torch.cat([loud_d2[1:], loud_d2[-1:]], dim=0)
        is_peak_a = (loud_d2 > d2_prev) & (loud_d2 > d2_next) & (loud_d2 > 0)
        peakAcc = torch.where(is_peak_a, loud_d2, torch.zeros_like(loud_d2))
        feats.append(peakAcc)  # 1 dim

        # ============== [LEAN-v34] SCALAR POWER-LAW DEVIATION ==============
        # Multi-second scale-free contextual normalization on 4 key 1-D signals
        # using a power-law kernel (beta=0.6, L=180 frames = 6s). Distinct from
        # existing 60/150-frame leaky EMAs (single characteristic tau) by
        # integrating ALL scales up to L. v19 tried it per-band (+40 dim, +0.0002);
        # this is the SCALAR version (cheap, 8 dim).
        pl_signals = torch.cat([long_loud, long_centroid, hnr[:, 1:2], best_strength], dim=1)
        pl_smoothed = power_law_smooth(pl_signals, beta=0.6, t0=1.0, L=180)
        feats.append(pl_smoothed)              # 4 dim: power-law smoothed
        feats.append(pl_signals - pl_smoothed)  # 4 dim: deviation from scale-free baseline

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "LEAN-v34-scalarPowLaw"
model_description = (
    "v28-v33 cumulative: 0.0761/0.0759/0.0758/0.0758/0.0755/0.0755. v34 adds "
    "SCALAR POWER-LAW DEVIATION block (+8 dim): power-law (beta=0.6, L=180=6s) "
    "smoothed long_loud/long_centroid/hnr_db/best_strength + their deviation. "
    "Multi-second scale-free normalization (NO single tau), distinct from "
    "60/150-frame leaky EMAs. v19 tried per-band +40 dim +0.0002; this is the "
    "scalar version. Total +62 dim. Keep if >=0.0747."
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
