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
        self._cond_logged = False
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
    # Final picked Stage-2 set: 'v2g' (10 anti-correlated curated pairs, 40 dim
    # post-delay). KEPT because: small block (≤150 dim), non-harmful (test_corr
    # 0.0762 ≥ baseline 0.0760 - 0.0003), most-interpretable variety.
    interact_set: str = 'v2g'

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
            if self.interact_set in ('v2a', 'v2b', 'v2c', 'v2d'):
                # 1. pitch-strength × voicing — confident-pitch gate
                inter.append(best_strength * hnr_peak)                          # 1
                # 2. pitch-strength × log F0 — pitch contour weighted by confidence
                inter.append(best_strength * log_f0)                            # 1
                # 3. pitch-strength × Δlog F0 — confident pitch change
                inter.append(best_strength * f0_d1)                             # 1
                # 4. HNR onset × loudness — voicing-onset salience
                inter.append(hnr_d * long_loud_1d)                              # 1
                # 5. formant dispersion × voicing — F2-F1 only when voiced
                inter.append(f12_disp * hnr_peak)                               # 1
                # 6. joint formant amplitude × voicing
                inter.append(f12_amp * hnr_peak)                                # 1
                # 7. flux × voicing — voiced-onset rate
                inter.append(flux_abs_long * hnr_peak)                          # 1
                # 8. flux × (1−voicing) — unvoiced-consonant onset rate
                inter.append(flux_abs_long * (1.0 - hnr_peak))                  # 1
                # 9. flux_pos × loudness — loud onset salience
                inter.append(flux_pos_long * long_loud_1d)                      # 1
                # 10. phon templates × flux_abs — phoneme-class-specific onsets
                inter.append(phon * flux_abs_long)                              # 5
                # 11. phon templates × loudness — phoneme-class energy
                inter.append(phon * long_loud_1d)                               # 5
                # 12. phon templates × voicing — voicing-gated phoneme classes
                inter.append(phon * hnr_peak)                                   # 5
                # 13. harmonic-comb salience × voicing — pitch comb gated by HNR
                inter.append(ps * hnr_peak)                                     # 11

            # --- 'v2b' : add band-energy RATIOS (in log space → subtractions) ---
            if self.interact_set in ('v2b', 'v2c', 'v2d'):
                nFh = self.n_filters
                # 4 equal-width band averages on long log-cochleagram (named "low", "mid-lo", "mid-hi", "high")
                b1 = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b2 = log_e_long[:, nFh//4:nFh//2].mean(dim=1, keepdim=True)
                b3 = log_e_long[:, nFh//2:3*nFh//4].mean(dim=1, keepdim=True)
                b4 = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                # log-ratios (subtractions): high/low, high/mid, mid/low, very-high/very-low
                inter.append(b4 - b1)                                           # 1  (high/low)
                inter.append(b4 - b2)                                           # 1
                inter.append(b3 - b1)                                           # 1
                inter.append(b4 - b3)                                           # 1
                # band ratios × voicing — voiced sibilance vs vowel energy
                inter.append((b4 - b1) * hnr_peak)                              # 1
                inter.append((b4 - b1) * (1.0 - hnr_peak))                      # 1  (unvoiced sibilance)
                # spectral-tilt (centroid) × loudness
                inter.append(long_centroid * long_loud_1d)                      # 1

            # --- 'v2c' : add envelope × short-lag-self (within-band AM proxy) ---
            if self.interact_set in ('v2c', 'v2d'):
                # x(t) × x(t-1) for long_loud, hnr_peak, flux_abs, best_strength — captures local AM
                for sig in (long_loud_1d, hnr_peak, flux_abs_long, best_strength):
                    sig_lag = torch.cat([sig[:1], sig[:-1]], dim=0)
                    inter.append(sig * sig_lag)                                 # 1 each → 4 total
                # spectral-flatness × loudness — noise-like vs tonal × loudness
                inter.append(flat * long_loud_1d)                               # 1
                # HNR onset × pitch change — voicing-onset with pitch movement
                inter.append(hnr_d * f0_d1)                                     # 1

            # --- 'v2f' : tight 5-product curated set (anti-correlated pairs only) ---
            if self.interact_set in ('v2f', 'v2g'):
                inter.append(hnr_d * long_loud_1d)                              # 1 voiced-onset salience
                inter.append((1.0 - hnr_peak) * flux_abs_long)                  # 1 unvoiced-onset rate
                inter.append(flat * long_loud_1d)                               # 1 broadband-noise energy
                inter.append(flat * flux_abs_long)                              # 1 noisy-onset (fricative burst)
                nFh = self.n_filters
                b_lo = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b_hi = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                inter.append((b_hi - b_lo) * flux_abs_long)                     # 1 high/low ratio × flux (frics)

            # --- 'v2g' : v2f + 5 more curated anti-correlated pairs ---
            if self.interact_set == 'v2g':
                inter.append(best_strength * flat)                              # 1 pitch confidence × noise-like
                inter.append(best_strength * (b_hi - b_lo))                     # 1 voicing × sibilance ratio
                inter.append(f12_disp * long_loud_1d)                           # 1 formant dispersion × loudness
                inter.append(log_f0 * flat)                                     # 1 pitch in noise
                inter.append(hnr_d * flux_abs_long)                             # 1 voicing onset coincident with flux

            # --- 'v2e' : SQUARES of key zero-mean 1-D signals (even-order
            # nonlinearity ridge cannot synthesize) + a few per-band AM products ---
            if self.interact_set == 'v2e':
                # per-frame zero-mean / unit-RMS within chunk so the square
                # captures DEVIATION energy not absolute level
                for sig in (long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat,
                            hnr_dvox, f12_disp):
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z * z)                                         # 1 each → 9 total
                # cross-band temporal AM proxy: log_e_long[t] × log_e_long[t-3]
                le_lag = torch.cat([log_e_long[:3], log_e_long[:-3]], dim=0)
                am_band = log_e_long * le_lag                                   # (T, 20)
                inter.append(am_band)                                           # 20
                # cochleagram_band × per-band onset (flux per band)
                d_le = torch.zeros_like(log_e_long)
                d_le[1:] = log_e_long[1:] - log_e_long[:-1]
                inter.append(log_e_long * d_le)                                 # 20  energy × onset per band

            # --- 'v2d' : add FORMANT × FLUX, MOMENTS × VOICING — moment-class gating ---
            if self.interact_set == 'v2d':
                inter.append(f12_disp * flux_abs_long)                          # 1
                inter.append(f12_amp * flux_abs_long)                           # 1
                inter.append(mom[:, 0:1] * hnr_peak)                            # 1  log-centroid × voicing
                inter.append(mom[:, 1:2] * hnr_peak)                            # 1  log-spread  × voicing
                inter.append(mom[:, 0:1] * flux_abs_long)                       # 1  centroid × flux
                # F1_loc × voicing, F2_loc × voicing — voiced formant locations
                inter.append(f1_loc * hnr_peak)                                 # 1
                inter.append(f2_loc * hnr_peak)                                 # 1

            if len(inter) > 0:
                add('interact_' + self.interact_set, torch.cat(inter, dim=1))

        feat = self._condition(groups)
        return feat.float().cpu().numpy()


model_shorthand_name = "INTERP-final"
model_description = (
    "INTERP final = LEAN-v16 refactored into 21 named groups + v2g curated "
    "anti-correlated product set (10 raw → 40 post-delay). Stage 3 (per-group "
    "rescaling) is mathematically inert under this harness because src/features.py "
    "per-chunk z-scores every column post-embedder; replication-based banded-ridge "
    "emulation gave within-noise gains. Stage 2 small curated products kept under "
    "small-block-non-harmful rule (no +0.0015 real gain). test_corr=0.0762."
)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------

def build_embedder(device: str = "cuda", cond_mode: str = "none",
                    interact_set: str = "none") -> WaveEmbedder:
    emb = WaveEmbedder(device=device)
    emb.cond_mode = cond_mode
    emb.interact_set = interact_set
    return emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cond-mode", default="none")
    parser.add_argument("--interact-set", default="none")
    parser.add_argument("--name", default=None)
    parser.add_argument("--description", default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available — falling back to cpu")
        args.device = "cpu"

    t0 = time.time()
    cfg = EncodingConfig(num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device, cond_mode=args.cond_mode,
                                interact_set=args.interact_set)
    r = run_encoding(embedder, cfg)
    n_params = int(getattr(embedder, "n_params", 0))

    sname = args.name or model_shorthand_name
    sdesc = args.description or f"{model_description} cond_mode={args.cond_mode}"
    upsert_overall_results(
        [make_result_row(r, sname, n_params, sdesc)], RESULTS_DIR)
    plot_corr_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"subject:        {r['subject']}")
    print(f"test_corr:      {r['test_corr']:.4f}  (train_corr={r['train_corr']:.4f}, "
          f"frac_elec>0.1={r['frac_test_elec_above_0.1']:.4f})")
    print(f"n_params:       {n_params:.2e}")
    print(f"encoding_secs:  {r['encoding_seconds']:.1f}s")
    print(f"total_seconds:  {time.time() - t0:.1f}s")
