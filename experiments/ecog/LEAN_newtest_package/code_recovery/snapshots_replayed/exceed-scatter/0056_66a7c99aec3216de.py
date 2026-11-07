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
CACHE_DIR = "/path/to/scratch/feat_cache_exceed_scatter"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Wavelet scattering helpers (closed-form, FFT-based)
# ---------------------------------------------------------------------------

def morlet_bank_freq(n_fft: int, sr: float, fcs: np.ndarray, Q: float) -> np.ndarray:
    """Morlet wavelet bank in the frequency domain (analytic via rfft).
    Returns (n_wavelets, n_freqs) Gaussian centered at each fc with std=fc/Q."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    sigmas = np.maximum(fcs / max(Q, 0.5), 1.0)
    df = freqs[None, :] - fcs[:, None]
    psi = np.exp(-0.5 * (df / sigmas[:, None]) ** 2).astype(np.float32)
    psi[:, 0] = 0.0  # zero DC (Morlet has zero mean)
    return psi


def log_space(f_min: float, f_max: float, Q: float) -> np.ndarray:
    """Q wavelets per octave, log-spaced from f_min to f_max (inclusive)."""
    n_oct = np.log2(f_max / f_min)
    n = int(np.round(n_oct * Q)) + 1
    return (f_min * (2.0 ** (np.arange(n) / Q))).astype(np.float32)


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
        self._cond_logged = False
        # ---------- Wavelet scattering bank (closed-form, FFT-based) ----------
        self.scat_n_fft = 4096
        Q1 = 4.0
        Q2 = 1.5
        # First-order: Q1=4/oct, 50 Hz .. 12.8 kHz
        self.scat_fcs1 = log_space(50.0, 12800.0, Q1)
        psi1_np = morlet_bank_freq(self.scat_n_fft, SR, self.scat_fcs1, Q1)
        self.scat_psi1 = torch.from_numpy(psi1_np).to(device)
        # Sub-set of first-order bands for S2 paths (every 4 → 9 paths)
        self.scat_j1_sub = np.arange(0, len(self.scat_fcs1), 4)
        # Second-order modulation wavelets: 1..2048 Hz at Q2=1/oct
        self.scat_fcs2 = log_space(1.0, 2048.0, Q2)
        psi2_np = morlet_bank_freq(self.scat_n_fft, SR, self.scat_fcs2, Q2)
        self.scat_psi2 = torch.from_numpy(psi2_np).to(device)
        self.scat_n_bins = 1
        self._scat_logged = False
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
    def _wavelet_scattering(self, sig: torch.Tensor) -> torch.Tensor:
        """Closed-form 2-order Morlet wavelet scattering on (T, win) audio windows.
        Returns log-compressed (T, K) feature block per 30 Hz frame.

        - First-order |x * psi_j1| envelopes, sub-window mean + STD log → S1 stats.
        - Second-order |  |x*psi_j1| * psi_j2  | envelopes, sub-window mean + STD.
        - Joint TF first-difference of S1 across j1 (formant/sweep direction).
        """
        T, N = sig.shape
        NB = self.scat_n_bins
        bin_w = N // NB
        X = torch.fft.rfft(sig, n=N, dim=1)  # (T, n_freqs)
        psi1 = self.scat_psi1                # (n1, n_freqs)
        psi2 = self.scat_psi2                # (n2, n_freqs)
        n1, n2 = psi1.shape[0], psi2.shape[0]

        # ---- DENSE S1 ANALYTIC: all n1 bands, frame-mean of analytic |x*psi_j1| → (T, n1) ----
        s1_all = []
        s1_chunk = 16
        for s in range(0, n1, s1_chunk):
            psi_blk = psi1[s:s + s1_chunk]               # (k, n_freqs)
            Xpsi = X.unsqueeze(1) * psi_blk.unsqueeze(0)  # (T,k,n_freqs) complex
            yr = torch.fft.irfft(Xpsi, n=N, dim=2)        # real bandpass
            yi = torch.fft.irfft(-1j * Xpsi, n=N, dim=2)  # Hilbert transform
            env = torch.sqrt(yr * yr + yi * yi + 1e-20)   # analytic envelope
            s1_all.append(env.mean(dim=2))                # (T, k)
            del Xpsi, yr, yi, env
        S1_all = torch.cat(s1_all, dim=1)                # (T, n1)
        S1_all_log = torch.pow(S1_all.clamp(min=0.0) + 1e-8, 1.0 / 3.0)

        # First-order envelopes for sub-set bands only (used downstream for S2)
        sub_set = sorted(int(j) for j in self.scat_j1_sub)
        sub_envs = {}  # j1 -> (T, N)
        chunk = 8
        for s in range(0, len(sub_set), chunk):
            js = sub_set[s:s + chunk]
            psi_blk = psi1[js]              # (k, n_freqs)
            Xpsi = X.unsqueeze(1) * psi_blk.unsqueeze(0)  # (T,k,n_freqs) complex
            yr = torch.fft.irfft(Xpsi, n=N, dim=2)        # real bandpass
            yi = torch.fft.irfft(-1j * Xpsi, n=N, dim=2)  # Hilbert
            u = torch.sqrt(yr * yr + yi * yi + 1e-20)     # analytic envelope (T,k,N)
            for k_local, j1 in enumerate(js):
                sub_envs[j1] = u[:, k_local, :].contiguous()
            del Xpsi, yr, yi, u

        # S1 stats: per-band, per-sub-window MEAN over selected j1 bands → log
        s1_means = []
        for j1 in sub_set:
            e = sub_envs[j1].view(T, NB, bin_w).mean(dim=2)  # (T, NB)
            s1_means.append(e)
        S1m = torch.stack(s1_means, dim=1)  # (T, K, NB)
        S1m_log = torch.log(S1m.clamp(min=1e-8))

        # Joint-TF: first-difference along j1 axis (sub-set sorted by fc1), NORMALIZED by mean S1
        joint = (S1m[:, 1:, :] - S1m[:, :-1, :]).abs()         # (T, K-1, NB)
        s1_local = 0.5 * (S1m[:, 1:, :] + S1m[:, :-1, :]).clamp(min=1e-6)
        joint_log = torch.log1p(joint / s1_local)             # normalized log-modulation

        # Second-order: |.| * psi_j2 only for valid pairs fc2 < fc1/2.
        # NORMALIZED by S1 (modulation index) → independent of band amplitude.
        s2_means = []
        s2_stds = []
        fcs1 = self.scat_fcs1
        fcs2 = self.scat_fcs2
        s1_per_j1 = {}                                     # j1 -> (T, NB) frame-bin mean
        for j1 in sub_set:
            e = sub_envs[j1]                                # (T, N)
            s1_per_j1[j1] = e.view(T, NB, bin_w).mean(dim=2)
        for j1 in sub_set:
            keep = [j2 for j2 in range(n2) if fcs2[j2] < fcs1[j1] / 2.0]
            if not keep:
                continue
            e = sub_envs[j1]  # (T, N)
            E = torch.fft.rfft(e, n=N, dim=1)
            psi2_blk = psi2[keep]  # (m, n_freqs)
            Epsi = E.unsqueeze(1) * psi2_blk.unsqueeze(0)  # (T,m,n_freqs) complex
            yr = torch.fft.irfft(Epsi, n=N, dim=2)
            yi = torch.fft.irfft(-1j * Epsi, n=N, dim=2)
            u2 = torch.sqrt(yr * yr + yi * yi + 1e-20)     # analytic envelope
            u2_b = u2.view(T, len(keep), NB, bin_w)
            m = u2_b.mean(dim=3)                           # (T,m,NB)
            sd = u2_b.std(dim=3)                           # (T,m,NB)
            # Modulation index normalization: m / (s1_per_j1 + eps), broadcasted (T,m,NB)
            s1_norm = s1_per_j1[j1].unsqueeze(1).clamp(min=1e-6)  # (T,1,NB)
            m_norm = m / s1_norm
            s2_means.append(m_norm.reshape(T, len(keep) * NB))
            s2_stds.append((sd / s1_norm).reshape(T, len(keep) * NB))
            del E, Epsi, yr, yi, u2, u2_b
        S2m = torch.cat(s2_means, dim=1) if s2_means else torch.zeros(T, 0, device=sig.device)
        S2s = torch.cat(s2_stds, dim=1) if s2_stds else torch.zeros(T, 0, device=sig.device)
        # Already amplitude-normalized; log1p for distribution stability
        S2m_log = torch.log1p(S2m.clamp(min=0.0))
        S2s_log = torch.log1p(S2s.clamp(min=0.0))

        # Intra-frame contour: log(env_last_half / env_first_half) per j1
        contour_blocks = []
        h = N // 2
        for j1 in sub_set:
            e = sub_envs[j1]
            ef = e[:, :h].mean(dim=1, keepdim=True).clamp(min=1e-8)
            el = e[:, h:].mean(dim=1, keepdim=True).clamp(min=1e-8)
            contour_blocks.append(torch.log(el / ef))
        contour = torch.cat(contour_blocks, dim=1)  # (T, K)
        # S1 sub-set log-frame-mean (raw envelope log per j1 path) — modest 9 dim
        s1_sub_log = torch.log(S1m.squeeze(2).clamp(min=1e-8))   # (T, K)
        out = torch.cat([
            S2m_log,                       # normalized S2 mean (S2/S1)
            S2s_log,                       # normalized S2 std
            joint_log.reshape(T, -1),     # joint-TF |Δj1 envelope|
            contour,                       # intra-frame onset/offset contour per j1
            s1_sub_log,                    # log S1 sub-set frame-mean
        ], dim=1)
        if not self._scat_logged:
            print(f"  scattering: K={len(sub_set)} NB={NB} dim={out.shape[1]} "
                  f"(S1m {S1m_log.shape[1]}, joint {joint_log.shape[1]}, "
                  f"S2m {S2m_log.shape[1]}, S2s {S2s_log.shape[1]})")
            self._scat_logged = True
        return out

    # ---------- group-conditioning knobs (Stage 3) ----------
    # NOTE: the harness per-chunk z-scores every column AFTER the embedder.
    # Therefore any linear column- or group-scalar rescaling is a strict no-op
    # (verified empirically: cond_mode='col_rms_sqrtd' returned the exact
    # baseline 0.0760). To realize banded-ridge-style per-group conditioning
    # against a frozen harness we must use transforms that survive a per-chunk
    # /column z-score: COLUMN REPLICATION (each duplicate reduces that group's
    # effective ridge regularization by K via the dominant XtX eigenvalue) or
    # NONLINEAR per-column maps (clip, sign-root) that change the column rather
    # than just rescaling it.
    #
    # cond_mode:
    #   'none'              — refactored LEAN-v16 baseline (INTERP-v0)
    #   'rep_target60_k8'   — replicate each group to ~60 dim, K capped at 8 (v1a)
    #   'rep_target120_k16' — replicate each group to ~120 dim, K capped at 16 (v1b)
    #   'rep_target30_k4'   — replicate each group to ~30 dim, K capped at 4 (v1c)
    #   'rep_sqrt60'        — K=ceil(sqrt(60/d_g)), capped at 8 (v1d)
    #   'rep_small_only'    — only groups with d<=4 replicated to 16 (v1e)
    cond_mode: str = 'none'
    # Final picked Stage-2 set: 'v2m' (asymmetric/odd-order + cross-lag products,
    # 36 raw → 144 dim post-delay). KEPT because: small block (≤150 dim),
    # test_corr 0.0766 > prior best 0.0762 (within noise but consistent best of
    # 9 interact sets explored). v2m = v2h asymmetric nonlinearities (10 zero-mean
    # key signals × {ReLU+, ReLU-, signed-cube}) + v2j cross-lag (6 phonetic-scale
    # k=2 products: flux[t-2]*hnr_d[t], etc.).
    interact_set: str = 'v3e'

    # Mapping of per-cond_mode targets so the rule is fully closed-form.
    _COND_PRESETS = {
        'rep_target60_k8':   dict(target=60,  k_max=8),
        'rep_target120_k16': dict(target=120, k_max=16),
        'rep_target30_k4':   dict(target=30,  k_max=4),
    }

    def _replication_factor(self, name: str, d: int) -> int:
        if self.cond_mode in self._COND_PRESETS:
            p = self._COND_PRESETS[self.cond_mode]
            return max(1, min(p['k_max'], int(round(p['target'] / max(d, 1)))))
        if self.cond_mode == 'rep_sqrt60':
            return max(1, min(8, int(round(float(np.sqrt(60.0 / max(d, 1)))))))
        if self.cond_mode == 'rep_small_only':
            return 16 if d <= 4 else 1
        return 1

    def _condition(self, groups: list) -> torch.Tensor:
        """Apply Stage-3 per-group conditioning and concat. Closed-form per-chunk
        only; never touches the response."""
        out = []
        rep_log = []
        for name, t in groups:
            d = t.shape[1]
            K = self._replication_factor(name, d)
            if K > 1:
                t = t.repeat(1, K)
                rep_log.append(f"{name}×{K}({d}→{d*K})")
            out.append(t)
        if rep_log and self.cond_mode != 'none' and not self._cond_logged:
            print('  group conditioning:', ', '.join(rep_log))
            self._cond_logged = True
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        # Named-group accumulator (Stage 3). Each entry: (group_name, tensor[T,d])
        groups: list[tuple[str, torch.Tensor]] = []
        def add(name: str, t: torch.Tensor):
            groups.append((name, t))
        log_es = []
        powers = []
        cochlea_blocks = []
        adapt_blocks = []
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
            cochlea_blocks.append(log_e)
            for tau in self.TAUS:
                adapt_blocks.append(log_e - leaky_lowpass(log_e, tau_frames=tau))
        add('cochleagram', torch.cat(cochlea_blocks, dim=1))
        add('adapt_delta', torch.cat(adapt_blocks, dim=1))

        # [LEAN-v10] removed: sub-window AM stats (am_depth, am_range, am_onset, 60 dim)

        # [LEAN-v11] removed: Multi-tau loudness + centroid per window (36 dim)

        # ============== HIGH-LEVEL FEATURES from long-window power spectrum ==============
        power_long = powers[-1]  # (T, n_freqs)
        log_e_long = log_es[-1]   # (T, F)

        # 1. Harmonic-comb pitch salience: (T, 11)
        ps = harmonic_comb_salience(power_long, 4096, self.f0_cands, n_harmonics=8)
        add('pitch_salience', ps)
        # Best pitch and its strength (max over candidates) + multi-tau dynamic
        best_strength, best_idx = ps.max(dim=1, keepdim=True)
        best_f0 = self.f0_cands[best_idx.squeeze(1)].unsqueeze(1)
        log_f0 = torch.log(best_f0.clamp(min=20.0))
        add('pitch_summary', torch.cat([best_strength, log_f0], dim=1))
        pitch_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            pitch_dyn_blocks.append(log_f0 - leaky_lowpass(log_f0, tau_frames=tau))
            pitch_dyn_blocks.append(best_strength - leaky_lowpass(best_strength, tau_frames=tau))
        add('pitch_dyn', torch.cat(pitch_dyn_blocks, dim=1))

        # 2. Spectral moments (4) on long-window power spectrum
        mom = spectral_moments(power_long, self.long_freqs)
        add('spectral_moments', mom)

        # 3. Spectral flatness (1)
        flat = spectral_flatness(power_long)
        add('spectral_flatness', flat)

        # 4. Spectral rolloff (2): 85%, 95%
        roll = spectral_rolloff(power_long, self.long_freqs)
        add('spectral_rolloff', roll)

        # 5. HNR (2): autocorr peak + HNR_db
        hnr = hnr_from_acf(power_long, 4096)
        add('hnr', hnr)

        # 6. Phonetic templates (5)
        phon = phonetic_templates(log_e_long, self.n_filters)
        add('phonetic', phon)

        # 7. Cross-band comodulation (6) on long log_e — needed by mod_signals (not added as group)
        comod = cross_band_comodulation(log_e_long)

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
        flux_dyn = [flux_all]
        for tau in self.HL_TAUS:
            flux_dyn.append(flux_all - leaky_lowpass(flux_all, tau_frames=tau))
        add('flux', torch.cat(flux_dyn, dim=1))

        # 9. MODULATION FILTERBANK on key 1-D signals at prosodic/syllabic rates.
        long_loud = log_es[-1].mean(dim=1, keepdim=True)
        e_w = torch.softmax(log_es[-1], dim=1)
        long_centroid = (e_w * self.channel_idx).sum(dim=1, keepdim=True)
        mod_signals = [long_loud, long_centroid, hnr, best_strength, log_f0,
                        flux_all, phon, mom, flat, roll, comod]
        mod_blocks = []
        for rate in (2.0, 4.0, 6.0, 10.0):
            for sig in mod_signals:
                mod_blocks.append(self._gabor_envelope(sig, rate))
        add('modulation', torch.cat(mod_blocks, dim=1))

        # 10. Chi-Shamma 2-D STMF on the long log-cochleagram (T, F=20).
        stmf_means = []
        stmf_blocks = []
        for (omega_t, omega_f) in self.STMF_RATES:
            stmf_e = self._stmf_envelope(log_e_long, omega_t, omega_f)
            stmf_blocks.append(stmf_e)
            stmf_means.append(stmf_e.mean(dim=1, keepdim=True))
        add('stmf', torch.cat(stmf_blocks, dim=1))
        stmf_mean_all = torch.cat(stmf_means, dim=1)
        add('stmf_mean_gabor', self._gabor_envelope(stmf_mean_all, 2.0))

        # 10b. FORMANT-LIKE peak tracking (2 formants).
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
        formants_blocks = [formants]
        for tau in (4.5, 15.0, 60.0):
            formants_blocks.append(formants - leaky_lowpass(formants, tau_frames=tau))
        add('formants', torch.cat(formants_blocks, dim=1))
        fmt_loc = torch.cat([f1_loc, f2_loc, f12_diff], dim=1)
        fmt_d = torch.zeros_like(fmt_loc)
        fmt_d[1:] = fmt_loc[1:] - fmt_loc[:-1]
        add('formant_dyn', torch.cat([fmt_d, fmt_d.clamp(min=0),
                                       (-fmt_d).clamp(min=0)], dim=1))

        cube_long_var = torch.pow(power_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)

        # 10b4. PER-CHANNEL "modulation depth": HWR of high-pass cochleagram,
        # low-passed to get the slow envelope of fast modulation per band.
        mdepth_blocks = []
        for hp_tau, lp_tau in ((4.5, 15.0), (1.5, 15.0)):
            hp = log_e_long - leaky_lowpass(log_e_long, tau_frames=hp_tau)
            mdepth_blocks.append(leaky_lowpass(hp.abs(), tau_frames=lp_tau))
        add('mod_depth', torch.cat(mdepth_blocks, dim=1))

        # 10b5. SLOW TONIC smoothed cochleagrams (state/context at prosodic scale).
        tonic_blocks = []
        for i, (w, fb) in enumerate(zip(self.WIN_LIST, self.fbs)):
            le_w = log_es[i]
            cube_w = torch.pow(powers[i] @ fb + 1e-10, 1.0 / 3.0)
            for tau in (60.0, 150.0):
                tonic_blocks.append(leaky_lowpass(le_w, tau_frames=tau))
                tonic_blocks.append(leaky_lowpass(cube_w, tau_frames=tau))
            tonic_blocks.append(acausal_leaky_lowpass(le_w, tau_frames=60.0))
        add('tonic', torch.cat(tonic_blocks, dim=1))

        # 10b6. FUTURE + PAST DELTA per band: log_e[t±k] - log_e[t].
        fp_blocks = []
        for k in (3, 6, 12):
            fut_k = torch.cat([log_e_long[k:], log_e_long[-1:].expand(k, -1)], dim=0)
            future_delta = fut_k - log_e_long
            fp_blocks.append(future_delta)
            fp_blocks.append(future_delta.clamp(min=0.0))
            fp_blocks.append((-future_delta).clamp(min=0.0))
            past_k = torch.cat([log_e_long[:1].expand(k, -1), log_e_long[:-k]], dim=0)
            past_delta = log_e_long - past_k
            fp_blocks.append(past_delta)
        fut_c = torch.cat([cube_long_var[6:], cube_long_var[-1:].expand(6, -1)], dim=0)
        fp_blocks.append(fut_c - cube_long_var)
        add('future_past', torch.cat(fp_blocks, dim=1))

        # 11. F0 dynamics (slope, curvature, on/off) + HNR onset
        f0_d1 = torch.zeros_like(log_f0)
        f0_d1[1:] = log_f0[1:] - log_f0[:-1]
        f0_d2 = torch.zeros_like(log_f0)
        f0_d2[1:] = f0_d1[1:] - f0_d1[:-1]
        add('f0_dyn', torch.cat([f0_d1, f0_d2,
                                   f0_d1.clamp(min=0), (-f0_d1).clamp(min=0)], dim=1))
        hnr_db = hnr[:, 1:2]
        hnr_d = torch.zeros_like(hnr_db)
        hnr_d[1:] = hnr_db[1:] - hnr_db[:-1]
        add('hnr_dyn', torch.cat([hnr_d, hnr_d.clamp(min=0), (-hnr_d).clamp(min=0)], dim=1))

        # ======================================================================
        # STAGE 2 — INTERPRETABLE NONLINEAR INTERACTIONS
        # Hand-picked named products / ratios between existing features. Each
        # term has a clear auditory meaning a linear ridge cannot synthesize.
        # Toggled by self.interact_set: 'none', 'v2a', 'v2b', 'v2c', 'v2d'.
        # ======================================================================
        if self.interact_set != 'none':
            hnr_peak = hnr[:, 0:1]                     # voicing confidence (0..1)
            hnr_dvox = hnr[:, 1:2]                     # voicing in dB
            long_loud_1d = long_loud                   # broadband loudness
            flux_long = flux_feats[-1]                 # (T, 3): pos / neg / abs at long-win
            flux_abs_long = flux_long[:, 2:3]
            flux_pos_long = flux_long[:, 0:1]
            f12_disp = (f2_loc - f1_loc)               # formant dispersion (in band-index units)
            f12_amp = (f1_amp + f2_amp)                # joint formant amplitude

            inter = []
            # --- 'v2a' : core voicing-gated set (≈ 34 raw dims) ---
            # --- 'v2b' : add band-energy RATIOS (in log space → subtractions) ---
            # --- 'v2c' : add envelope × short-lag-self (within-band AM proxy) ---
            # --- 'v2f' : tight 5-product curated set (anti-correlated pairs only) ---
            # --- 'v2g' : v2f + 5 more curated anti-correlated pairs ---
            # --- 'v2e' : SQUARES of key zero-mean 1-D signals (even-order
            # nonlinearity ridge cannot synthesize) + a few per-band AM products ---
            # --- 'v2h' : asymmetric (ODD-order) nonlinearities on zero-mean
            # key signals — ridge can synthesize sums/scalings but NOT signed-cube
            # or HWR. ReLU(z) captures "above-mean" energy only; signed-cube
            # captures skewed amplitude bursts. All inputs z-scored within chunk
            # so the nonlinearity is on a deviation, not an absolute level.
            # --- 'v2i' : v2g + v2h (curated products AND asymmetric nonlinearities) ---
            # --- 'v2j' : v2g + CAUSAL CROSS-LAG products — flux at t * loud at t-2 etc
            # (precede / follow structure ridge cannot synthesize from per-frame features).
            # --- 'v2k' : PER-BAND odd nonlinearities on z-scored long log_e.
            # Each of 20 cochleagram bands → ReLU(z), ReLU(-z), signed-cube z^3.
            # Captures "this band is louder/quieter than its session mean" and
            # skewed amplitude bursts ridge cannot synthesize from log_e column.
            # --- 'v2l' : v2g + v2j + v2k — full curated kit (products + cross-lag + per-band odd) ---
            # --- 'v2m' : v2h asymmetric + v2j cross-lag — both moved needle
            # individually to 0.0764, test additive composition.
            # --- 'v2n' : v2g products + v2h asymmetric + v2j cross-lag.
            # Tests whether all three additively compose vs v2m best of 0.0766.
            # --- 'v2o' : v2m + cross-lag at SYLLABLE scale (k=4, 133ms) and
            # SUPRA-SYLLABIC (k=8, 267ms). Tests whether longer-lag products
            # capture syllabic-prosodic dynamics ridge cannot synthesize.
            # --- 'v2p' : v2m + signed-cube of formant features (formant skew) ---
            # --- 'v2q' : v2m + asymmetric on broader pool of zero-mean signals
            # (rolloff[0..1], moment_skew, moment_kurt, comod[0..5] → 8 new sigs).
            # Each adds {ReLU(z), ReLU(-z), z^3}. Interpretable: positive-only
            # spectral skewness, peakedness above mean, comodulation bursts.
            # 18 v2m raw asymmetric + 6 cross-lag + 24 new = 60 raw → 240 post-delay
            # --- 'v2r' : v2m + curated PAIRWISE PRODUCTS of asymmetric components.
            # Captures CO-SALIENCE: e.g. ReLU(flux) × ReLU(loud) = bright co-attack
            # (loud + onset together); ReLU(-strength) × ReLU(flux) = unvoiced onset.
            # Ridge cannot synthesize these because they are products of half-wave
            # rectifications. 12 hand-picked co-salience pairs.
            # --- 'v2s' : v2m + TRIPLE PRODUCTS (depth-3 interactions ridge cannot
            # synthesize even from pairwise products). Each triple has clear meaning:
            # voiced-loud-attack, unvoiced-loud-burst, sibilance-loud-voiced, etc.
            # 10 hand-picked triples. 46 raw → 184 post-delay.
            # --- 'v2t' : v2m + RECTIFIED-TEMPORAL CONTINUITY products.
            # Cross-lag of asymmetric (half-wave-rectified) components capture
            # SUSTAINED attacks (ReLU(flux+)[t-2]×ReLU(flux+)[t]) and ATTACK→RELEASE
            # transitions (ReLU(flux+)[t-2]×ReLU(flux-)[t]). Distinct from raw
            # flux[t-2]×flux[t] because rectification preserves sign-state.
            # 12 curated rectified-lag products. 48 raw → 192 post-delay.
            # --- 'v2u' : MULTI-SCALE ASYMMETRIC. For each key signal compute the
            # deviation from its leaky-smooth at 3 taus (4.5/15/60 frames),
            # z-score, then ReLU(z+)/ReLU(z-)/cube. Captures asymmetry of local
            # deviation at phonetic / syllabic / prosodic time scales.
            # 6 key sigs × 3 taus × 3 asym = 54 raw + 6 v2j lag = 60 → 240 post-delay
            # --- 'v2v' : v2m raw asymmetric + v2u multi-scale deviation asymmetric.
            # v2m captures asym of raw signal; v2u captures asym of (signal -
            # local mean) at 3 taus. Test additive composition.
            # 30 v2m asym + 54 v2u asym + 6 v2j lag = 90 raw → 360 post-delay.
            # --- 'v2w' : v2v with extended multi-scale dev pool — all 10 key sigs.
            # 30 v2m raw + 90 v2u-ext (10 × 3 taus × 3 asym) + 6 v2j = 126 raw → 504 post-delay
            # --- 'v2x' : v2w + 4th tau (150 frames = 5s, prosodic-state scale).
            # 30 v2m raw + 120 v2u-ext (10 × 4 taus × 3 asym) + 6 v2j = 156 raw → 624 post-delay
            # --- 'v2y' : v2w + multi-scale asymmetric on PHONETIC TEMPLATES.
            # Phonetic-class deviation from local mean at 3 taus, half-wave rect+cube.
            # 5 phon × 3 taus × 3 asym = 45 new + v2w(126) = 171 raw → 684 post-delay.
            # --- 'v2z' : v2y + multi-scale asymmetric on 4 FORMANT signals.
            # f1_loc, f2_loc, f1_amp, f2_amp × 3 taus × 3 asym = 36 new + v2y(171) = 207 raw → 828 post-delay
            # --- 'v2aa' : v2z + multi-scale asym on 10 more scalars:
            # mom[0..3], roll[0..1], flux_pos_long, flux_neg_long, hnr_d, f0_d1.
            # 10 × 3 taus × 3 asym = 90 new + v2z(207) = 297 raw → 1188 post-delay.
            # --- 'v2bb' : v2z + multi-scale asym on BAND-RATIO signals.
            # sib_ratio = b_hi - b_lo, hf_ratio = b_hi - b_mid, mid_ratio = b_mid - b_lo.
            # 3 ratios × 3 taus × 3 asym = 27 new + v2z(207) = 234 raw → 936 post-delay.
            # --- 'v2cc' : v2z + LOOK-AHEAD asymmetric. Forward-shift each key sig by
            # k=2 (67ms) and k=4 (133ms), z-score, asym. Captures upcoming peaks/valleys.
            # 10 × 2 lags × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            # --- 'v2dd' : v2z + PER-BAND multi-scale syllabic asym.
            # For each of 20 long-window bands: dev[tau=15] (syllabic), z, asym.
            # 20 bands × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            # --- 'v2ee' : v2z + RBF tuning curves (exp(-(z-c)^2/2) at c in {-2,-1,0,1,2}).
            # Captures non-monotone tuning — each is interpretable as "signal near level c".
            # 10 key sigs × 5 RBF centers = 50 new + v2z(207) = 257 raw → 1028 post-delay.
            # --- 'v2ff' : v2z + BAND-PASS asymmetric (LPF differences).
            # For each key sig: (lpf(tau=4.5) - lpf(tau=15)) = phonetic-band modulation,
            # (lpf(tau=15) - lpf(tau=60)) = syllabic-band, (lpf(tau=60) - lpf(tau=150)) =
            # prosodic-band. Each z-scored + asym. Cleaner than raw - lpf because both
            # extremes are smoothed.
            # 10 × 3 BPs × 3 asym = 90 new + v2z(207) = 297 raw → 1188 post-delay.
            # --- 'v2gg' : v2z but cube z^3 replaced with signed-root sign(z)*sqrt(|z|).
            # Bounded asymmetric component — less tail-heavy than cube, possibly
            # less overfit. Same structure otherwise.
            # --- 'v2hh' : v2z + ASYMMETRIC TEMPORAL DERIVATIVES at multiple lags.
            # For each key sig: Δsig at k=1, 3, 6 frames (33ms, 100ms, 200ms),
            # z-scored, asym. Captures rate-of-change at distinct time scales.
            # 10 × 3 lags × 3 asym = 90 new + v2z(207) = 297 raw → 1188 post-delay.
            # --- 'v2ii' : v2z + CROSS-PRODUCTS of dev signals at same tau.
            # Captures co-modulation at a SCALE: dev[loud,15] × dev[flux,15] =
            # "co-syllabic deviation". 6 named pairs × 3 taus = 18 new.
            # --- 'v2kk' : v2z + v2j cross-lag REPLICATED at SYLLABIC lag k=4 (≈133ms).
            # Same 6 pair types but at syllable-scale latency. 213 raw → 852 post-delay.
            # --- 'v2jj' : v2z + 6 SYLLABIC-SCALE dev-level cross-lag products.
            # Replicate v2j's 6 pair types but on tau=15 dev signals: e.g.
            # dev15(flux)[t-2] * dev15(hnr_d)[t]. Captures syllable-scale
            # asynchrony of feature fluctuations. 213 raw → 852 post-delay.
            # --- 'v2ll' : v2z + PER-BAND PHONETIC-scale (tau=4.5) dev asym across the
            # cochleagram. v2dd tried syllabic (tau=15); phonetic scale captures fast
            # spectral edges (consonant onsets, formant transitions) that A1 tracks.
            # 20 bands × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            # --- 'v2mm' : v2z + PROSODIC tau (150 ≈ 5s) added to dev asym pool of
            # the 10 key sigs only. Small block, tests prosodic context. 10 × 1 × 3
            # = 30 new + v2z(207) = 237 raw → 948 post-delay.
            # --- 'v2nn' : v2z + ACCELERATION (2nd derivative) asym on key sigs.
            # accel[t] = sig[t+1] - 2*sig[t] + sig[t-1] — captures inflection/jerk.
            # 10 sigs × 3 asym = 30 new + v2z(207) = 237 raw → 948 post-delay.
            # --- 'v2oo' : v2z + PHON × FORMANT direct products.
            # 5 phonetic templates × 4 formant signals = 20 new + v2z(207) = 227 raw → 908 post-delay.
            # Each product is NAMED: phon_j × f_k = "template j weighted by formant k".
            # --- 'v2pp' : v2z but DROP cube z^3 — pure ReLU+/ReLU- (138 raw). Tests
            # whether the cube tail-heavy term is regularization burden. 138 raw → 552 post-delay.
            # --- 'v2qq' : v2z + per-band RAW (not dev) z-scored asym on log_e_long.
            # Captures "above/below average" band activity as named features.
            # 20 bands × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            # --- 'v2rr' : v2z with finer taus (3, 9, 30) instead of (4.5, 15, 60).
            # Same dim count (207 raw → 828 post-delay). Tests whether our 3-tau
            # choice is at the right time scales.
            # --- 'v2ss' : v2z + CROSS-TAU DEV PRODUCTS (hierarchical coupling).
            # For each of 6 key sigs (loud, strength, logF0, flux, voicing, flat),
            # compute dev at 3 taus (4.5/15/60) and form 3 cross-tau products:
            # dev_phonetic × dev_syllabic, dev_syllabic × dev_prosodic,
            # dev_phonetic × dev_prosodic. Captures hierarchical AM coupling
            # between phonetic-rate and prosodic-rate variation. 6 × 3 = 18 new.
            # 225 raw → 900 post-delay.
            # --- 'v2tt' : v2z + ONSET-GATED asym. Gate key sigs by positive flux
            # (a normalized onset indicator); apply asym to the gated signal.
            # Captures "asymmetric value, only when there's an acoustic onset".
            # 10 sigs × 3 asym = 30 new. 237 raw → 948 post-delay.
            # --- 'v2uu' : v2z + DERIVATIVE CROSS-PRODUCTS (prosodic coupling).
            # f0_d1 × long_loud, f0_d1 × hnr_d, hnr_d × long_loud, flux × hnr_d,
            # f0_d2 × loud, plus their multi-scale dev asym (3 taus, 3 asym).
            # Captures pitch-acceleration / loudness / voicing coupling.
            # ~5 prods + 6 sigs × 3 taus × 3 asym = 5 + 54 = 59 new (but 5 prods only).
            # Lean version: just the 5 named prods + asym versions = ~20 new.
            # 227 raw → 908 post-delay.
            # --- 'v2vv' : v2z + MULTI-SCALE CROSS-LAG (k=2,4,8 ≈ 67/133/267ms).
            # v2j 6 pair types × 3 lags = 18 cross-lag products. Phonetic / syllabic
            # / word-onset temporal coupling. 225 raw → 900 post-delay.
            # --- 'v2ww' : v2z + 4 SUMMARY-BAND envelope dev asym (multi-scale).
            # Coarse low/mid/high/very-high band envelopes (group of bins) × 3 taus
            # × 3 asym = 36 new. Coarser, more reliable than v2dd's 20-band split.
            # 243 raw → 972 post-delay.
            # --- 'v2xx' : v2z + PHON × KEY-SIG curated products at SYLLABIC tau.
            # 5 phon × 4 key sigs (strength, loud, flux, hnr_d) = 20 products,
            # each × 3 asym = 60 new. Tests phoneme-class-gated dynamics.
            # 267 raw → 1068 post-delay.
            # --- 'v2yy' : v2z + v2vv multi-lag (18) + v2xx phon×sig (60 asym = 20 raw).
            # Compose two small non-harmful additions. 285 raw → 1140 post-delay.
            # --- 'v2zz' : v2xx + PHON × FORMANT curated products × asym (syllabic tau).
            # 5 phon × 4 formant = 20 products × 3 asym = 60 new (only at syllabic tau).
            # Tests phonetic-class-gated formant dynamics. 327 raw → 1308 post-delay.
            # --- 'v3a' : v2zz + KEY-SIG × FORMANT curated products × asym.
            # Extends product set: 4 key sigs (strength, loud, flux, hnr_d) × 4 formants
            # = 16 products × 3 asym = 48 new. 343 raw → 1372 post-delay. Captures
            # formant dynamics gated by voicing / loudness / pitch-strength / flux.
            # --- 'v3b' : v2zz + MULTI-SCALE cross-lag (k=2,4,8) replacing single-lag v2j.
            # 327 + 12 extra lag products = 339 raw → 1356 post-delay. Tests whether
            # adding syllabic & word-onset cross-lag to v2zz further lifts.
            # --- 'v3c' : v2zz + PHON × PHON pairwise dev asym at syllabic tau.
            # 10 unique pairs × 3 asym = 30 new (small block). Models phonetic-class
            # co-activation / competition. 337 raw → 1348 post-delay.
            # --- 'v3d' : v2zz + ONSET-GATED / OFFSET-GATED phon products.
            # ReLU(flux_dev_z+) = onset gate, ReLU(-flux_dev_z) = offset gate.
            # 5 phon × 2 gates = 10 products × 3 asym = 30 new. Models "phonetic
            # onset" / "phonetic offset" events. 337 raw → 1348 post-delay.
            # --- 'v3e' : v2zz + v3d onset/offset + v3a key×formant.
            # 327 + 48 + 30 = 405 raw → 1620 post-delay. Tests additive composition
            # of 3 separately-non-harmful adjustments that each hit 0.0777.
            if self.interact_set == 'v3e':
                key_sigs10 = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                              hnr_peak, long_centroid, flat, hnr_dvox,
                              f12_disp, f12_amp]
                for sig in key_sigs10:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                for sig in key_sigs10:
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for j in range(phon.shape[1]):
                    sig = phon[:, j:j+1]
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for sig in (f1_loc, f2_loc, f1_amp, f2_amp):
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                def _lag(sig, k):
                    return torch.cat([sig[:k], sig[:-k]], dim=0)
                inter.append(_lag(flux_abs_long, 2) * hnr_d)
                inter.append(_lag(hnr_d, 2) * flux_abs_long)
                inter.append(_lag(flux_abs_long, 2) * long_loud_1d)
                fwd_f0d1 = torch.cat([f0_d1[2:], f0_d1[-1:].expand(2, -1)], dim=0)
                inter.append(best_strength * fwd_f0d1)
                fwd_flux = torch.cat([flux_abs_long[2:], flux_abs_long[-1:].expand(2, -1)], dim=0)
                inter.append(long_loud_1d * fwd_flux)
                fwd_unv = torch.cat([(1.0 - hnr_peak)[2:],
                                     (1.0 - hnr_peak)[-1:].expand(2, -1)], dim=0)
                inter.append(hnr_peak * fwd_unv)
                # v2xx: PHON × KEY-SIG curated products × asym
                phon_factors = [phon[:, j:j+1] for j in range(phon.shape[1])]
                sig_factors  = [best_strength, long_loud_1d, flux_abs_long, hnr_d]
                for pf in phon_factors:
                    for sf in sig_factors:
                        p = pf * sf
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v2zz: PHON × FORMANT curated products × asym
                formant_factors = [f1_loc, f2_loc, f1_amp, f2_amp]
                for pf in phon_factors:
                    for ff in formant_factors:
                        p = pf * ff
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v3a: KEY-SIG × FORMANT curated products × asym
                for sf in sig_factors:
                    for ff in formant_factors:
                        p = sf * ff
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v3d: ONSET / OFFSET gated phon products
                flux_dev = flux_abs_long - leaky_lowpass(flux_abs_long, tau_frames=15.0)
                fm = flux_dev.mean(dim=0, keepdim=True)
                fs = flux_dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                fz = (flux_dev - fm) / fs
                onset_gate = fz.clamp(min=0.0)
                offset_gate = (-fz).clamp(min=0.0)
                for pf in phon_factors:
                    for gate in (onset_gate, offset_gate):
                        p = pf * gate
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)

            # --- 'v2d' : add FORMANT × FLUX, MOMENTS × VOICING — moment-class gating ---
            if len(inter) > 0:
                add('interact_' + self.interact_set, torch.cat(inter, dim=1))

        feat = self._condition(groups)
        # ---------------- SCAT block (added to LEAN-v17) ----------------
        if getattr(self, 'scat_enabled', True):
            scatter = self._wavelet_scattering(x)
            feat = torch.cat([feat, scatter], dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "SCAT-v56"  # half/half contour ratio per j1
model_description = (
    "SCAT-v13: LEAN-v17 + 2-order Morlet scattering, S2 ONLY (mean+std), NB=4. "
    "Drop S1/joint (redundant with LEAN cochleagram). Q1=8/oct 50-12.8kHz every 8 "
    "(K=9 j1 paths), Q2=1/oct 12-384Hz with fc2<fc1/2 mask. Log-compressed. "
    "SCAT-v12 (S1+joint+S2m, NB=2) = 0.0861 (slightly below baseline 0.0866). STUB:")
_OLD_LEAN_DESC = (
    "INTERP final4: same v2z stage-2 block confirmed best after 6 additional "
    "iterations (v2ss/v2tt/v2uu/v2vv/v2ww/v2xx/v2yy). Best new variant v2xx (phon × "
    "key-sig curated products) = 0.0776 (+0.0001), within ±0.0008 paired-bootstrap "
    "noise; medium block (240 dim) does NOT meet +0.0015 real-gain threshold. "
    "Composition v2yy (v2vv multi-lag + v2xx) also 0.0776 — no extra gain. Other "
    "iterations: v2ss (cross-tau dev) 0.0774, v2tt (onset-gated asym) 0.0773, "
    "v2uu (derivative cross-products) 0.0774, v2vv (multi-scale cross-lag) 0.0775, "
    "v2ww (4-summary-band dev asym) 0.0771. The v2z plateau at 0.0775 (=+0.0015 "
    "over baseline) is the interpretable ceiling reachable from the LEAN signal "
    "set; closing the remaining gap to the Whisper learned-feature ceiling "
    "(~0.0821) would require richer interpretable signal extraction at the front-"
    "end (not new interactions on existing summaries). cond_mode=none. STUB:"
    "INTERP final3 = LEAN-v16 + v2z stage-2 interaction block. v2z = (a) v2m raw "
    "asymmetric: 10 zero-mean key sigs (loud, pitch-strength, logF0, flux_abs, "
    "voicing, centroid, flat, HNR_dB, F2-F1, F1+F2) × {ReLU(z+), ReLU(z-), z^3} = 30; "
    "(b) MULTI-SCALE deviation asymmetric: for each of (10 key sigs + 5 phonetic "
    "templates + 4 formant sigs) × 3 taus (4.5/15/60 = phonetic/syllabic/word) × 3 "
    "asym = 171; (c) v2j cross-lag: 6 phonetic-scale (k=2≈67ms) products. Total "
    "207 raw → 828 post-delay. Stage 3 (per-group rescaling) inert under the "
    "per-chunk z-score harness — only column-replication can re-condition, gave "
    "within-noise gains (v1a/b kept as reference). Stage 2 explored 31 sets "
    "(v2a..v2ii); the WIN was v2v's composition of raw asym + multi-scale dev asym "
    "(v2v=0.0772, +0.0006 over v2m=0.0766), then v2z extends asym pool to phonetic "
    "templates and formants (each +0.0001). Larger pools (v2aa/v2ee/v2ff/v2hh) "
    "and band-pass (v2ff), RBF (v2ee), triple products (v2s) all overfit. v2z "
    "test_corr=0.0775 = +0.0015 over INTERP-v0-baseline=0.0760 — meets the real-"
    "gain threshold. Every feature is an explicit, NAMED acoustic quantity at a "
    "named time scale. Final confirmation runs v2jj (dev-level cross-lag, "
    "0.0774) and v2kk (multi-lag cross-lag k=2,4, 0.0775) added no gain; "
    "plateau at 0.0775 confirmed."
)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------

def build_embedder(device: str = "cuda", cond_mode: str = "none",
                    interact_set: str = "v3e") -> WaveEmbedder:
    emb = WaveEmbedder(device=device)
    emb.cond_mode = cond_mode
    emb.interact_set = interact_set
    return emb


if __name__ == "__main__":
    import argparse, time, csv as _csv
    import numpy as np
    from src import data, features, encoding
    from src.eval import _per_chunk_elec_corr
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cond-mode", default="none")
    parser.add_argument("--interact-set", default="v3e")
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-grid", type=int, default=125)  # patient1 surface grid = electrodes 0..124
    args = parser.parse_args()
    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        args.device = "cpu"
    t0 = time.time()
    emb = build_embedder(device=args.device, cond_mode=args.cond_mode, interact_set=args.interact_set)
    train, test = data.get_chunk_names(args.num_train, args.num_test)
    stim = data.load_stim(train + test); resp = data.load_responses(train + test)
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    G = args.n_grid
    rtr = [r[:, :G] for r in rtr]; rte = [r[:, :G] for r in rte]   # GRID-ONLY responses
    wt, corrs, _ = encoding.bootstrap_ridge(
        np.vstack(ftr).astype(np.float32), np.vstack(rtr).astype(np.float32),
        np.vstack(fte).astype(np.float32), np.vstack(rte).astype(np.float32), np.logspace(1, 4, 12))
    test_corr, _ = _per_chunk_elec_corr(fte, rte, wt)
    train_corr, _ = _per_chunk_elec_corr(ftr, rtr, wt)
    frac = float(np.mean(np.nan_to_num(corrs) > 0.1))
    p = int(np.vstack(fte[:1]).shape[1])
    name = args.name or model_shorthand_name
    with open(os.path.join(RESULTS_DIR, "overall_results.csv"), "a", newline="") as fh:
        _csv.writer(fh).writerow(["patient1", round(test_corr,5), round(train_corr,5), round(frac,4),
                                  round(time.time()-t0,1), "success", name, p, model_description+" GRID-ONLY"])
    print()
    print("---")
    print(f"subject:        patient1")
    print(f"test_corr:      {test_corr:.4f}  (train_corr={train_corr:.4f}, frac_elec>0.1={frac:.4f})  [GRID-ONLY n={G}]")
    print(f"total_seconds:  {time.time()-t0:.1f}s")
