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

        # Sub-window AM stats
        sub_logs = []
        for i in range(4):
            xs = x[:, i * self.SUB_WIN:(i + 1) * self.SUB_WIN]
            ps = power_spectrum(xs, self.SUB_WIN, self.device)
            le = torch.log(torch.clamp(ps @ self.fb_sub, min=1e-10))
            sub_logs.append(le)
        sub_stack = torch.stack(sub_logs, dim=2)
        am_depth = sub_stack.std(dim=2)
        am_range = sub_stack.amax(dim=2) - sub_stack.amin(dim=2)
        diffs = sub_stack[:, :, 1:] - sub_stack[:, :, :-1]
        am_onset = diffs.amax(dim=2).clamp(min=0.0)
        feats.append(am_depth)
        feats.append(am_range)
        feats.append(am_onset)

        # Multi-tau loudness + centroid per window
        extra = []
        idx = 0
        for w in self.WIN_LIST:
            log_e_w = feats[idx]
            loud = log_e_w.mean(dim=1, keepdim=True)
            e_w = torch.softmax(log_e_w, dim=1)
            centroid = (e_w * self.channel_idx).sum(dim=1, keepdim=True)
            for sig in (loud, centroid):
                extra.append(sig)
                for tau in self.TAUS:
                    extra.append(sig - leaky_lowpass(sig, tau_frames=tau))
            idx += 1 + len(self.TAUS)
        feats.extend(extra)

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

        # 5. HNR (2): autocorr peak + HNR_db + multi-tau (voicing dynamics)
        hnr = hnr_from_acf(power_long, 4096)
        feats.append(hnr)
        for tau in self.HL_TAUS:
            feats.append(hnr - leaky_lowpass(hnr, tau_frames=tau))

        # 6. Phonetic templates (5) + multi-tau (phoneme/syllable transitions)
        phon = phonetic_templates(log_e_long, self.n_filters)
        feats.append(phon)
        for tau in self.HL_TAUS:
            feats.append(phon - leaky_lowpass(phon, tau_frames=tau))

        # 7. Cross-band comodulation (6) on long log_e
        comod = cross_band_comodulation(log_e_long)
        feats.append(comod)

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
        # Small additional STMF on lateral-inhibited (sharpened) cochleagram.
        log_e_li = self._dog_channels(log_e_long, 0.7, 2.5)
        for (omega_t, omega_f) in ((4.0, 0.25), (4.0, -0.25),
                                     (8.0, 0.25), (8.0, -0.25),
                                     (4.0, 0.5), (8.0, 0.5),
                                     (4.0, 0.0), (8.0, 0.0)):
            feats.append(self._stmf_envelope(log_e_li, omega_t, omega_f))

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

        # 10b2. CUBE-ROOT compression cochleagrams for all 3 windows.
        for i, (w, fb) in enumerate(zip(self.WIN_LIST, self.fbs)):
            cube_w = torch.pow(powers[i] @ fb + 1e-10, 1.0 / 3.0)
            feats.append(cube_w)
            for tau in (4.5, 15.0):
                feats.append(cube_w - leaky_lowpass(cube_w, tau_frames=tau))

        # 10b3. PER-CHANNEL TEMPORAL VARIANCE (smoothed) — captures stable vs
        # modulated bands. On LONG window log + cube cochleagrams.
        log_e_sq = log_e_long ** 2
        for tau in (4.5, 15.0):
            ex2 = leaky_lowpass(log_e_sq, tau)
            ex_sq = leaky_lowpass(log_e_long, tau) ** 2
            feats.append(torch.sqrt((ex2 - ex_sq).clamp(min=0.0) + 1e-12))
        cube_long_var = torch.pow(power_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)
        cube_sq = cube_long_var ** 2
        for tau in (4.5, 15.0):
            ex2 = leaky_lowpass(cube_sq, tau)
            ex_sq = leaky_lowpass(cube_long_var, tau) ** 2
            feats.append(torch.sqrt((ex2 - ex_sq).clamp(min=0.0) + 1e-12))

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

        # 10b7. FUTURE/PAST PEAK-HOLD per band — max pooling over short windows.
        # Different from futdelta: captures upcoming/recent envelope PEAK regardless
        # of monotonic change. Approximates auditory peripheral peak detectors.
        for K in (3, 6, 12):
            # future peak: max(log_e[t..t+K])
            pad_f = torch.cat([log_e_long, log_e_long[-1:].expand(K, -1)], dim=0)
            pad_f_u = pad_f.unsqueeze(0).transpose(1, 2)  # (1, F, T+K)
            fut_peak = torch.nn.functional.max_pool1d(pad_f_u, kernel_size=K + 1, stride=1).squeeze(0).transpose(0, 1)
            fut_peak = fut_peak[:log_e_long.shape[0]]
            feats.append(fut_peak - log_e_long)  # peak-relative-to-now per band
            # past peak: max(log_e[t-K..t])
            pad_p = torch.cat([log_e_long[:1].expand(K, -1), log_e_long], dim=0)
            pad_p_u = pad_p.unsqueeze(0).transpose(1, 2)
            past_peak = torch.nn.functional.max_pool1d(pad_p_u, kernel_size=K + 1, stride=1).squeeze(0).transpose(0, 1)
            past_peak = past_peak[:log_e_long.shape[0]]
            feats.append(past_peak - log_e_long)


        # 10c. Per-formant BANDWIDTH (weighted std around soft-argmax) + sharpness.
        d1 = ch1 - f1_loc; d2 = ch2 - f2_loc
        f1_bw = torch.sqrt((w1 * d1 * d1).sum(dim=1, keepdim=True) + 1e-12)
        f2_bw = torch.sqrt((w2 * d2 * d2).sum(dim=1, keepdim=True) + 1e-12)
        f1_peak = f1_band.amax(dim=1, keepdim=True) - f1_band.mean(dim=1, keepdim=True)
        f2_peak = f2_band.amax(dim=1, keepdim=True) - f2_band.mean(dim=1, keepdim=True)
        fmt_bw = torch.cat([f1_bw, f2_bw, f1_peak, f2_peak], dim=1)
        feats.append(fmt_bw)
        for tau in (4.5, 15.0, 60.0):
            feats.append(fmt_bw - leaky_lowpass(fmt_bw, tau_frames=tau))


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


        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "Gammatone-BEST+peakhold-futpast"
model_description = (
    "FINAL closed-form auditory front-end. 3-window (23/46/93 ms) gammatone x "
    "6-tau base + within-frame AM (depth/range/onset) + per-window loudness & "
    "centroid multi-tau + HIGH-LEVEL features: harmonic-comb pitch salience (11 "
    "F0 candidates, 8 harmonics) + best-F0 log-pitch & strength multi-tau + "
    "spectral moments/flatness/rolloff + HNR multi-tau + phonetic matched "
    "filters (sib/vowel/nasal/plosive/fric) multi-tau + cross-band "
    "comodulation + SPECTRAL FLUX (broadband pos/neg/abs per window) multi-tau "
    "+ MODULATION FILTERBANK (Gabor envelopes at 2/4/6/10 Hz on 11 HL signals) "
    "+ Chi-Shamma 2-D SPECTRO-TEMPORAL MODULATION (17 rate-scale pairs on long "
    "log-cochleagram) + STMF broadband modulation envelope at 2 Hz. All "
    "closed-form, 0 trained parameters."
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
