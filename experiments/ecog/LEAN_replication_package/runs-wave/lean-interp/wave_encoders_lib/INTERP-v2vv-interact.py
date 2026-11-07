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
    # Final picked Stage-2 set: 'v2m' (asymmetric/odd-order + cross-lag products,
    # 36 raw → 144 dim post-delay). KEPT because: small block (≤150 dim),
    # test_corr 0.0766 > prior best 0.0762 (within noise but consistent best of
    # 9 interact sets explored). v2m = v2h asymmetric nonlinearities (10 zero-mean
    # key signals × {ReLU+, ReLU-, signed-cube}) + v2j cross-lag (6 phonetic-scale
    # k=2 products: flux[t-2]*hnr_d[t], etc.).
    interact_set: str = 'v2vv'

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

            # --- 'v2h' : asymmetric (ODD-order) nonlinearities on zero-mean
            # key signals — ridge can synthesize sums/scalings but NOT signed-cube
            # or HWR. ReLU(z) captures "above-mean" energy only; signed-cube
            # captures skewed amplitude bursts. All inputs z-scored within chunk
            # so the nonlinearity is on a deviation, not an absolute level.
            if self.interact_set == 'v2h':
                key_sigs = [('loud',  long_loud_1d), ('pitch_strength', best_strength),
                            ('logF0', log_f0),       ('flux_abs',       flux_abs_long),
                            ('hnr_peak', hnr_peak),  ('centroid',       long_centroid),
                            ('flatness', flat),      ('hnr_db',         hnr_dvox),
                            ('f12_disp', f12_disp),  ('f12_amp',        f12_amp)]
                for _name, sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))                              # 1 ReLU above-mean
                    inter.append((-z).clamp(min=0.0))                           # 1 ReLU below-mean
                    inter.append(z * z * z)                                     # 1 signed-cube (odd, skew)

            # --- 'v2i' : v2g + v2h (curated products AND asymmetric nonlinearities) ---
            if self.interact_set == 'v2i':
                # v2g curated products
                inter.append(hnr_d * long_loud_1d)
                inter.append((1.0 - hnr_peak) * flux_abs_long)
                inter.append(flat * long_loud_1d)
                inter.append(flat * flux_abs_long)
                nFh = self.n_filters
                b_lo = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b_hi = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                inter.append((b_hi - b_lo) * flux_abs_long)
                inter.append(best_strength * flat)
                inter.append(best_strength * (b_hi - b_lo))
                inter.append(f12_disp * long_loud_1d)
                inter.append(log_f0 * flat)
                inter.append(hnr_d * flux_abs_long)
                # v2h asymmetric nonlinearities on zero-mean key signals
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)

            # --- 'v2j' : v2g + CAUSAL CROSS-LAG products — flux at t * loud at t-2 etc
            # (precede / follow structure ridge cannot synthesize from per-frame features).
            if self.interact_set == 'v2j':
                # v2g curated products
                inter.append(hnr_d * long_loud_1d)
                inter.append((1.0 - hnr_peak) * flux_abs_long)
                inter.append(flat * long_loud_1d)
                inter.append(flat * flux_abs_long)
                nFh = self.n_filters
                b_lo = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b_hi = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                inter.append((b_hi - b_lo) * flux_abs_long)
                inter.append(best_strength * flat)
                inter.append(best_strength * (b_hi - b_lo))
                inter.append(f12_disp * long_loud_1d)
                inter.append(log_f0 * flat)
                inter.append(hnr_d * flux_abs_long)
                # cross-lag products (k=2 frames ≈ 67 ms — phonetic boundary scale)
                def _lag(sig, k):
                    return torch.cat([sig[:k], sig[:-k]], dim=0)
                # voicing-onset preceded by flux: flux[t-2] * hnr_d[t]
                inter.append(_lag(flux_abs_long, 2) * hnr_d)                    # 1
                # flux preceded by voicing-onset: hnr_d[t-2] * flux[t]
                inter.append(_lag(hnr_d, 2) * flux_abs_long)                    # 1
                # loud preceded by flux: flux[t-2] * loud[t]
                inter.append(_lag(flux_abs_long, 2) * long_loud_1d)             # 1
                # pitch_strength followed by F0 change: best_strength[t] * f0_d1[t+2]
                fwd_f0d1 = torch.cat([f0_d1[2:], f0_d1[-1:].expand(2, -1)], dim=0)
                inter.append(best_strength * fwd_f0d1)                          # 1
                # loud followed by flux: long_loud[t] * flux[t+2]
                fwd_flux = torch.cat([flux_abs_long[2:], flux_abs_long[-1:].expand(2, -1)], dim=0)
                inter.append(long_loud_1d * fwd_flux)                           # 1
                # voicing followed by un-voicing (transition): hnr_peak[t] * (1-hnr_peak)[t+2]
                fwd_unv = torch.cat([(1.0 - hnr_peak)[2:],
                                     (1.0 - hnr_peak)[-1:].expand(2, -1)], dim=0)
                inter.append(hnr_peak * fwd_unv)                                # 1

            # --- 'v2k' : PER-BAND odd nonlinearities on z-scored long log_e.
            # Each of 20 cochleagram bands → ReLU(z), ReLU(-z), signed-cube z^3.
            # Captures "this band is louder/quieter than its session mean" and
            # skewed amplitude bursts ridge cannot synthesize from log_e column.
            if self.interact_set == 'v2k':
                m = log_e_long.mean(dim=0, keepdim=True)
                s = log_e_long.std(dim=0, keepdim=True).clamp(min=1e-6)
                z_le = (log_e_long - m) / s                                     # (T, 20)
                inter.append(z_le.clamp(min=0.0))                               # 20  ReLU above-mean per band
                inter.append((-z_le).clamp(min=0.0))                            # 20  ReLU below-mean per band
                inter.append(z_le * z_le * z_le)                                # 20  signed-cube per band

            # --- 'v2l' : v2g + v2j + v2k — full curated kit (products + cross-lag + per-band odd) ---
            if self.interact_set == 'v2l':
                # v2g products
                inter.append(hnr_d * long_loud_1d)
                inter.append((1.0 - hnr_peak) * flux_abs_long)
                inter.append(flat * long_loud_1d)
                inter.append(flat * flux_abs_long)
                nFh = self.n_filters
                b_lo = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b_hi = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                inter.append((b_hi - b_lo) * flux_abs_long)
                inter.append(best_strength * flat)
                inter.append(best_strength * (b_hi - b_lo))
                inter.append(f12_disp * long_loud_1d)
                inter.append(log_f0 * flat)
                inter.append(hnr_d * flux_abs_long)
                # v2j cross-lag
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
                # v2k per-band odd
                m_le = log_e_long.mean(dim=0, keepdim=True)
                s_le = log_e_long.std(dim=0, keepdim=True).clamp(min=1e-6)
                z_le = (log_e_long - m_le) / s_le
                inter.append(z_le.clamp(min=0.0))
                inter.append((-z_le).clamp(min=0.0))
                inter.append(z_le * z_le * z_le)

            # --- 'v2m' : v2h asymmetric + v2j cross-lag — both moved needle
            # individually to 0.0764, test additive composition.
            if self.interact_set == 'v2m':
                # v2h asymmetric on 10 zero-mean key signals
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # v2j cross-lag products
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

            # --- 'v2n' : v2g products + v2h asymmetric + v2j cross-lag.
            # Tests whether all three additively compose vs v2m best of 0.0766.
            if self.interact_set == 'v2n':
                # v2g curated products (10)
                inter.append(hnr_d * long_loud_1d)
                inter.append((1.0 - hnr_peak) * flux_abs_long)
                inter.append(flat * long_loud_1d)
                inter.append(flat * flux_abs_long)
                nFh = self.n_filters
                b_lo = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b_hi = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                inter.append((b_hi - b_lo) * flux_abs_long)
                inter.append(best_strength * flat)
                inter.append(best_strength * (b_hi - b_lo))
                inter.append(f12_disp * long_loud_1d)
                inter.append(log_f0 * flat)
                inter.append(hnr_d * flux_abs_long)
                # v2h asymmetric on 10 zero-mean key signals (30)
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # v2j cross-lag (6)
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

            # --- 'v2o' : v2m + cross-lag at SYLLABLE scale (k=4, 133ms) and
            # SUPRA-SYLLABIC (k=8, 267ms). Tests whether longer-lag products
            # capture syllabic-prosodic dynamics ridge cannot synthesize.
            if self.interact_set == 'v2o':
                # v2h asymmetric
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # multi-scale cross-lag (k=2, 4, 8 frames = 67/133/267 ms)
                def _lag(sig, k):
                    return torch.cat([sig[:k], sig[:-k]], dim=0)
                def _fwd(sig, k):
                    return torch.cat([sig[k:], sig[-1:].expand(k, -1)], dim=0)
                for k in (2, 4, 8):
                    inter.append(_lag(flux_abs_long, k) * hnr_d)
                    inter.append(_lag(hnr_d, k) * flux_abs_long)
                    inter.append(_lag(flux_abs_long, k) * long_loud_1d)
                    inter.append(best_strength * _fwd(f0_d1, k))
                    inter.append(long_loud_1d * _fwd(flux_abs_long, k))
                    inter.append(hnr_peak * _fwd(1.0 - hnr_peak, k))

            # --- 'v2p' : v2m + signed-cube of formant features (formant skew) ---
            if self.interact_set == 'v2p':
                # v2h asymmetric
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # v2j cross-lag k=2
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
                # asymmetric on formant 1-D features
                for sig in (f1_loc, f2_loc, f1_amp, f2_amp):
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)

            # --- 'v2q' : v2m + asymmetric on broader pool of zero-mean signals
            # (rolloff[0..1], moment_skew, moment_kurt, comod[0..5] → 8 new sigs).
            # Each adds {ReLU(z), ReLU(-z), z^3}. Interpretable: positive-only
            # spectral skewness, peakedness above mean, comodulation bursts.
            # 18 v2m raw asymmetric + 6 cross-lag + 24 new = 60 raw → 240 post-delay
            if self.interact_set == 'v2q':
                # v2m asymmetric (10) + extra 8 = 18 total
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp,
                            roll[:, 0:1], roll[:, 1:2],
                            mom[:, 2:3], mom[:, 3:4]]
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                for j in range(6):
                    sig = comod[:, j:j+1]
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # v2j cross-lag
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

            # --- 'v2r' : v2m + curated PAIRWISE PRODUCTS of asymmetric components.
            # Captures CO-SALIENCE: e.g. ReLU(flux) × ReLU(loud) = bright co-attack
            # (loud + onset together); ReLU(-strength) × ReLU(flux) = unvoiced onset.
            # Ridge cannot synthesize these because they are products of half-wave
            # rectifications. 12 hand-picked co-salience pairs.
            if self.interact_set == 'v2r':
                # v2m asymmetric (30)
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                z_list = []
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    z_list.append(z)
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # v2j cross-lag
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
                # PAIRWISE PRODUCTS OF ASYMMETRIC COMPONENTS (12 co-salience pairs):
                # Index map: 0=loud 1=strength 2=logf0 3=flux 4=hnr_peak
                #            5=centroid 6=flat 7=hnr_dvox 8=f12_disp 9=f12_amp
                rp = [z.clamp(min=0.0) for z in z_list]
                rn = [(-z).clamp(min=0.0) for z in z_list]
                # Loud-onset co-salience: ReLU(flux+) × ReLU(loud+)
                inter.append(rp[3] * rp[0])
                # Voiced-loud-pitch convergence: ReLU(strength+) × ReLU(loud+)
                inter.append(rp[1] * rp[0])
                # Unvoiced-flux burst: ReLU(strength-) × ReLU(flux+)
                inter.append(rn[1] * rp[3])
                # Voiced-onset: ReLU(hnr_peak+) × ReLU(flux+)
                inter.append(rp[4] * rp[3])
                # Quiet-tonal: ReLU(loud-) × ReLU(flat-)  (low energy AND tonal)
                inter.append(rn[0] * rn[6])
                # Noisy-loud: ReLU(loud+) × ReLU(flat+)
                inter.append(rp[0] * rp[6])
                # Bright-loud: ReLU(centroid+) × ReLU(loud+)
                inter.append(rp[5] * rp[0])
                # F0-attack: ReLU(strength+) × ReLU(flux+)
                inter.append(rp[1] * rp[3])
                # F1-F2-spread-voiced: ReLU(f12_disp+) × ReLU(hnr_peak+)
                inter.append(rp[8] * rp[4])
                # Formant-amp loud: ReLU(f12_amp+) × ReLU(loud+)
                inter.append(rp[9] * rp[0])
                # High-pitch-tonal: ReLU(logf0+) × ReLU(flat-)
                inter.append(rp[2] * rn[6])
                # Voicing-loss flux: ReLU(hnr_peak-) × ReLU(flux+)
                inter.append(rn[4] * rp[3])

            # --- 'v2s' : v2m + TRIPLE PRODUCTS (depth-3 interactions ridge cannot
            # synthesize even from pairwise products). Each triple has clear meaning:
            # voiced-loud-attack, unvoiced-loud-burst, sibilance-loud-voiced, etc.
            # 10 hand-picked triples. 46 raw → 184 post-delay.
            if self.interact_set == 'v2s':
                # v2m asymmetric (30)
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # v2j cross-lag
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
                # TRIPLE PRODUCTS — named depth-3 interactions
                nFh = self.n_filters
                b_lo = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b_hi = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                # 1. voiced-loud-attack
                inter.append(flux_abs_long * hnr_peak * long_loud_1d)
                # 2. unvoiced-loud-burst
                inter.append(flux_abs_long * (1.0 - hnr_peak) * long_loud_1d)
                # 3. pitch-strength × pitch × loud (voiced-loud-pitch cube)
                inter.append(best_strength * log_f0 * long_loud_1d)
                # 4. broadband-noisy-burst
                inter.append(flat * long_loud_1d * flux_abs_long)
                # 5. voiced-sibilance: (hi-lo) × loud × voicing
                inter.append((b_hi - b_lo) * long_loud_1d * hnr_peak)
                # 6. unvoiced-sibilance: (hi-lo) × loud × (1-voicing)
                inter.append((b_hi - b_lo) * long_loud_1d * (1.0 - hnr_peak))
                # 7. voiced-formant-bright-vowel
                inter.append(f12_disp * long_loud_1d * hnr_peak)
                # 8. voiced-formant-amp
                inter.append(f12_amp * long_loud_1d * hnr_peak)
                # 9. flux × hnr_d × loud — voiced-onset transient
                inter.append(flux_abs_long * hnr_d * long_loud_1d)
                # 10. strength × f0_d1 × loud — voiced-pitch-change × loud
                inter.append(best_strength * f0_d1 * long_loud_1d)

            # --- 'v2t' : v2m + RECTIFIED-TEMPORAL CONTINUITY products.
            # Cross-lag of asymmetric (half-wave-rectified) components capture
            # SUSTAINED attacks (ReLU(flux+)[t-2]×ReLU(flux+)[t]) and ATTACK→RELEASE
            # transitions (ReLU(flux+)[t-2]×ReLU(flux-)[t]). Distinct from raw
            # flux[t-2]×flux[t] because rectification preserves sign-state.
            # 12 curated rectified-lag products. 48 raw → 192 post-delay.
            if self.interact_set == 'v2t':
                # v2m asymmetric (30) — store z for reuse
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, long_centroid, flat, hnr_dvox,
                            f12_disp, f12_amp]
                z_list = []
                for sig in key_sigs:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    z_list.append(z)
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                # v2j cross-lag
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
                # RECTIFIED-TEMPORAL CONTINUITY:
                # Index map: 0=loud 1=strength 3=flux 4=hnr_peak 6=flat
                rp = [z.clamp(min=0.0) for z in z_list]
                rn = [(-z).clamp(min=0.0) for z in z_list]
                # SUSTAINED attacks: ReLU(sig+)[t-2] × ReLU(sig+)[t]
                inter.append(_lag(rp[0], 2) * rp[0])  # sustained loud
                inter.append(_lag(rp[3], 2) * rp[3])  # sustained flux
                inter.append(_lag(rp[4], 2) * rp[4])  # sustained voicing
                inter.append(_lag(rp[1], 2) * rp[1])  # sustained pitch-strength
                # ATTACK→RELEASE transitions: ReLU(sig+)[t-2] × ReLU(sig-)[t]
                inter.append(_lag(rp[0], 2) * rn[0])  # loud→quiet
                inter.append(_lag(rp[3], 2) * rn[3])  # flux→still
                # RELEASE→ATTACK (rest→activity): ReLU(sig-)[t-2] × ReLU(sig+)[t]
                inter.append(_lag(rn[0], 2) * rp[0])  # quiet→loud (true onset)
                inter.append(_lag(rn[4], 2) * rp[4])  # unvoiced→voiced (V onset)
                inter.append(_lag(rp[4], 2) * rn[4])  # voiced→unvoiced (V offset)
                # Sustained quiet (ReLU(sig-)[t-2] × ReLU(sig-)[t])
                inter.append(_lag(rn[0], 2) * rn[0])
                inter.append(_lag(rn[3], 2) * rn[3])
                # Cross-feature rectified lag: loud rises while flux rises
                inter.append(_lag(rp[3], 2) * rp[0])

            # --- 'v2u' : MULTI-SCALE ASYMMETRIC. For each key signal compute the
            # deviation from its leaky-smooth at 3 taus (4.5/15/60 frames),
            # z-score, then ReLU(z+)/ReLU(z-)/cube. Captures asymmetry of local
            # deviation at phonetic / syllabic / prosodic time scales.
            # 6 key sigs × 3 taus × 3 asym = 54 raw + 6 v2j lag = 60 → 240 post-delay
            if self.interact_set == 'v2u':
                key_sigs = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                            hnr_peak, flat]
                for sig in key_sigs:
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v2j cross-lag
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

            # --- 'v2v' : v2m raw asymmetric + v2u multi-scale deviation asymmetric.
            # v2m captures asym of raw signal; v2u captures asym of (signal -
            # local mean) at 3 taus. Test additive composition.
            # 30 v2m asym + 54 v2u asym + 6 v2j lag = 90 raw → 360 post-delay.
            if self.interact_set == 'v2v':
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
                # v2u multi-scale deviation asym
                key_sigs6 = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                             hnr_peak, flat]
                for sig in key_sigs6:
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v2j cross-lag
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

            # --- 'v2w' : v2v with extended multi-scale dev pool — all 10 key sigs.
            # 30 v2m raw + 90 v2u-ext (10 × 3 taus × 3 asym) + 6 v2j = 126 raw → 504 post-delay
            if self.interact_set == 'v2w':
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

            # --- 'v2x' : v2w + 4th tau (150 frames = 5s, prosodic-state scale).
            # 30 v2m raw + 120 v2u-ext (10 × 4 taus × 3 asym) + 6 v2j = 156 raw → 624 post-delay
            if self.interact_set == 'v2x':
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
                    for tau in (4.5, 15.0, 60.0, 150.0):
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

            # --- 'v2y' : v2w + multi-scale asymmetric on PHONETIC TEMPLATES.
            # Phonetic-class deviation from local mean at 3 taus, half-wave rect+cube.
            # 5 phon × 3 taus × 3 asym = 45 new + v2w(126) = 171 raw → 684 post-delay.
            if self.interact_set == 'v2y':
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
                # PHONETIC TEMPLATES multi-scale asym (5 × 3 × 3 = 45)
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

            # --- 'v2z' : v2y + multi-scale asymmetric on 4 FORMANT signals.
            # f1_loc, f2_loc, f1_amp, f2_amp × 3 taus × 3 asym = 36 new + v2y(171) = 207 raw → 828 post-delay
            if self.interact_set == 'v2z':
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
                # Formant multi-scale (4 × 3 × 3 = 36)
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

            # --- 'v2aa' : v2z + multi-scale asym on 10 more scalars:
            # mom[0..3], roll[0..1], flux_pos_long, flux_neg_long, hnr_d, f0_d1.
            # 10 × 3 taus × 3 asym = 90 new + v2z(207) = 297 raw → 1188 post-delay.
            if self.interact_set == 'v2aa':
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
                # NEW: 10 more scalars × 3 taus × 3 asym = 90
                more_sigs = [mom[:, 0:1], mom[:, 1:2], mom[:, 2:3], mom[:, 3:4],
                             roll[:, 0:1], roll[:, 1:2],
                             flux_pos_long, flux_long[:, 1:2], hnr_d, f0_d1]
                for sig in more_sigs:
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

            # --- 'v2bb' : v2z + multi-scale asym on BAND-RATIO signals.
            # sib_ratio = b_hi - b_lo, hf_ratio = b_hi - b_mid, mid_ratio = b_mid - b_lo.
            # 3 ratios × 3 taus × 3 asym = 27 new + v2z(207) = 234 raw → 936 post-delay.
            if self.interact_set == 'v2bb':
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
                # NEW: band-ratio multi-scale asym
                nFh = self.n_filters
                b_lo = log_e_long[:, :nFh//4].mean(dim=1, keepdim=True)
                b_mid = log_e_long[:, nFh//4:3*nFh//4].mean(dim=1, keepdim=True)
                b_hi = log_e_long[:, 3*nFh//4:].mean(dim=1, keepdim=True)
                sib_ratio = b_hi - b_lo
                hf_ratio = b_hi - b_mid
                mid_ratio = b_mid - b_lo
                for sig in (sib_ratio, hf_ratio, mid_ratio):
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

            # --- 'v2cc' : v2z + LOOK-AHEAD asymmetric. Forward-shift each key sig by
            # k=2 (67ms) and k=4 (133ms), z-score, asym. Captures upcoming peaks/valleys.
            # 10 × 2 lags × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            if self.interact_set == 'v2cc':
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
                # LOOK-AHEAD asymmetric
                def _fwd(sig, k):
                    return torch.cat([sig[k:], sig[-1:].expand(k, -1)], dim=0)
                for sig in key_sigs10:
                    for k in (2, 4):
                        fsig = _fwd(sig, k)
                        m = fsig.mean(dim=0, keepdim=True)
                        s = fsig.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (fsig - m) / s
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

            # --- 'v2dd' : v2z + PER-BAND multi-scale syllabic asym.
            # For each of 20 long-window bands: dev[tau=15] (syllabic), z, asym.
            # 20 bands × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            if self.interact_set == 'v2dd':
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
                # PER-BAND syllabic dev asym (tau=15)
                band_dev = log_e_long - leaky_lowpass(log_e_long, tau_frames=15.0)
                m = band_dev.mean(dim=0, keepdim=True)
                s = band_dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                z_bd = (band_dev - m) / s
                inter.append(z_bd.clamp(min=0.0))
                inter.append((-z_bd).clamp(min=0.0))
                inter.append(z_bd * z_bd * z_bd)
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

            # --- 'v2ee' : v2z + RBF tuning curves (exp(-(z-c)^2/2) at c in {-2,-1,0,1,2}).
            # Captures non-monotone tuning — each is interpretable as "signal near level c".
            # 10 key sigs × 5 RBF centers = 50 new + v2z(207) = 257 raw → 1028 post-delay.
            if self.interact_set == 'v2ee':
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
                # RBF tuning curves
                centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0],
                                       device=x.device, dtype=torch.float32)
                for sig in key_sigs10:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    for c in centers:
                        inter.append(torch.exp(-(z - c) ** 2 / 2.0))
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

            # --- 'v2ff' : v2z + BAND-PASS asymmetric (LPF differences).
            # For each key sig: (lpf(tau=4.5) - lpf(tau=15)) = phonetic-band modulation,
            # (lpf(tau=15) - lpf(tau=60)) = syllabic-band, (lpf(tau=60) - lpf(tau=150)) =
            # prosodic-band. Each z-scored + asym. Cleaner than raw - lpf because both
            # extremes are smoothed.
            # 10 × 3 BPs × 3 asym = 90 new + v2z(207) = 297 raw → 1188 post-delay.
            if self.interact_set == 'v2ff':
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
                # BAND-PASS asymmetric
                bp_pairs = ((4.5, 15.0), (15.0, 60.0), (60.0, 150.0))
                for sig in key_sigs10:
                    for (tau_lo, tau_hi) in bp_pairs:
                        bp = leaky_lowpass(sig, tau_frames=tau_lo) - leaky_lowpass(sig, tau_frames=tau_hi)
                        m = bp.mean(dim=0, keepdim=True)
                        s = bp.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (bp - m) / s
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

            # --- 'v2gg' : v2z but cube z^3 replaced with signed-root sign(z)*sqrt(|z|).
            # Bounded asymmetric component — less tail-heavy than cube, possibly
            # less overfit. Same structure otherwise.
            if self.interact_set == 'v2gg':
                def _asym(sig):
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(torch.sign(z) * torch.sqrt(torch.abs(z) + 0.01))
                key_sigs10 = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                              hnr_peak, long_centroid, flat, hnr_dvox,
                              f12_disp, f12_amp]
                for sig in key_sigs10:
                    _asym(sig)
                for sig in key_sigs10:
                    for tau in (4.5, 15.0, 60.0):
                        _asym(sig - leaky_lowpass(sig, tau_frames=tau))
                for j in range(phon.shape[1]):
                    sig = phon[:, j:j+1]
                    for tau in (4.5, 15.0, 60.0):
                        _asym(sig - leaky_lowpass(sig, tau_frames=tau))
                for sig in (f1_loc, f2_loc, f1_amp, f2_amp):
                    for tau in (4.5, 15.0, 60.0):
                        _asym(sig - leaky_lowpass(sig, tau_frames=tau))
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

            # --- 'v2hh' : v2z + ASYMMETRIC TEMPORAL DERIVATIVES at multiple lags.
            # For each key sig: Δsig at k=1, 3, 6 frames (33ms, 100ms, 200ms),
            # z-scored, asym. Captures rate-of-change at distinct time scales.
            # 10 × 3 lags × 3 asym = 90 new + v2z(207) = 297 raw → 1188 post-delay.
            if self.interact_set == 'v2hh':
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
                # Multi-lag temporal derivatives asym
                for sig in key_sigs10:
                    for k in (1, 3, 6):
                        d_sig = torch.zeros_like(sig)
                        d_sig[k:] = sig[k:] - sig[:-k]
                        m = d_sig.mean(dim=0, keepdim=True)
                        s = d_sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (d_sig - m) / s
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

            # --- 'v2ii' : v2z + CROSS-PRODUCTS of dev signals at same tau.
            # Captures co-modulation at a SCALE: dev[loud,15] × dev[flux,15] =
            # "co-syllabic deviation". 6 named pairs × 3 taus = 18 new.
            if self.interact_set == 'v2ii':
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
                # Compute and store all dev z's at 3 taus per sig
                dev_z = {}
                for i, sig in enumerate(key_sigs10):
                    dev_z[i] = {}
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        dev_z[i][tau] = z
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
                # CROSS-PRODUCTS at same tau (6 named pairs × 3 taus)
                # idx: 0=loud 1=strength 2=logf0 3=flux 4=hnr_peak 6=flat
                pairs = [(0, 3), (0, 4), (1, 3), (3, 4), (0, 6), (3, 6)]
                for (i, j) in pairs:
                    for tau in (4.5, 15.0, 60.0):
                        inter.append(dev_z[i][tau] * dev_z[j][tau])
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

            # --- 'v2kk' : v2z + v2j cross-lag REPLICATED at SYLLABIC lag k=4 (≈133ms).
            # Same 6 pair types but at syllable-scale latency. 213 raw → 852 post-delay.
            if self.interact_set == 'v2kk':
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
                def _lagk(sig, k):
                    return torch.cat([sig[:k], sig[:-k]], dim=0)
                def _fwdk(sig, k):
                    return torch.cat([sig[k:], sig[-1:].expand(k, -1)], dim=0)
                for k in (2, 4):
                    inter.append(_lagk(flux_abs_long, k) * hnr_d)
                    inter.append(_lagk(hnr_d, k) * flux_abs_long)
                    inter.append(_lagk(flux_abs_long, k) * long_loud_1d)
                    inter.append(best_strength * _fwdk(f0_d1, k))
                    inter.append(long_loud_1d * _fwdk(flux_abs_long, k))
                    inter.append(hnr_peak * _fwdk(1.0 - hnr_peak, k))

            # --- 'v2jj' : v2z + 6 SYLLABIC-SCALE dev-level cross-lag products.
            # Replicate v2j's 6 pair types but on tau=15 dev signals: e.g.
            # dev15(flux)[t-2] * dev15(hnr_d)[t]. Captures syllable-scale
            # asynchrony of feature fluctuations. 213 raw → 852 post-delay.
            if self.interact_set == 'v2jj':
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
                # Dev-level syllabic cross-lag (6 new products)
                def _dev15(sig):
                    return sig - leaky_lowpass(sig, tau_frames=15.0)
                d_flux = _dev15(flux_abs_long)
                d_hnrd = _dev15(hnr_d)
                d_loud = _dev15(long_loud_1d)
                d_strg = _dev15(best_strength)
                d_voic = _dev15(hnr_peak)
                d_f0d1 = _dev15(f0_d1)
                inter.append(_lag(d_flux, 2) * d_hnrd)
                inter.append(_lag(d_hnrd, 2) * d_flux)
                inter.append(_lag(d_flux, 2) * d_loud)
                fwd_df0d1 = torch.cat([d_f0d1[2:], d_f0d1[-1:].expand(2, -1)], dim=0)
                inter.append(d_strg * fwd_df0d1)
                fwd_dflux = torch.cat([d_flux[2:], d_flux[-1:].expand(2, -1)], dim=0)
                inter.append(d_loud * fwd_dflux)
                fwd_dunv = torch.cat([(-d_voic)[2:], (-d_voic)[-1:].expand(2, -1)], dim=0)
                inter.append(d_voic * fwd_dunv)

            # --- 'v2ll' : v2z + PER-BAND PHONETIC-scale (tau=4.5) dev asym across the
            # cochleagram. v2dd tried syllabic (tau=15); phonetic scale captures fast
            # spectral edges (consonant onsets, formant transitions) that A1 tracks.
            # 20 bands × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            if self.interact_set == 'v2ll':
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
                # PER-BAND phonetic-scale dev asym (tau=4.5, ~150ms)
                band_dev = log_e_long - leaky_lowpass(log_e_long, tau_frames=4.5)
                m = band_dev.mean(dim=0, keepdim=True)
                s = band_dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                z_bd = (band_dev - m) / s
                inter.append(z_bd.clamp(min=0.0))
                inter.append((-z_bd).clamp(min=0.0))
                inter.append(z_bd * z_bd * z_bd)
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

            # --- 'v2mm' : v2z + PROSODIC tau (150 ≈ 5s) added to dev asym pool of
            # the 10 key sigs only. Small block, tests prosodic context. 10 × 1 × 3
            # = 30 new + v2z(207) = 237 raw → 948 post-delay.
            if self.interact_set == 'v2mm':
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
                # PROSODIC tau (150) dev asym on key sigs
                for sig in key_sigs10:
                    dev = sig - leaky_lowpass(sig, tau_frames=150.0)
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

            # --- 'v2nn' : v2z + ACCELERATION (2nd derivative) asym on key sigs.
            # accel[t] = sig[t+1] - 2*sig[t] + sig[t-1] — captures inflection/jerk.
            # 10 sigs × 3 asym = 30 new + v2z(207) = 237 raw → 948 post-delay.
            if self.interact_set == 'v2nn':
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
                # ACCELERATION asym on key sigs (centered 2nd difference)
                for sig in key_sigs10:
                    # pad both ends to keep length
                    sig_fwd = torch.cat([sig[1:], sig[-1:]], dim=0)
                    sig_bwd = torch.cat([sig[:1], sig[:-1]], dim=0)
                    accel = sig_fwd - 2.0 * sig + sig_bwd
                    m = accel.mean(dim=0, keepdim=True)
                    s = accel.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (accel - m) / s
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

            # --- 'v2oo' : v2z + PHON × FORMANT direct products.
            # 5 phonetic templates × 4 formant signals = 20 new + v2z(207) = 227 raw → 908 post-delay.
            # Each product is NAMED: phon_j × f_k = "template j weighted by formant k".
            if self.interact_set == 'v2oo':
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
                # PHON × FORMANT products (curated 20 dim)
                for j in range(phon.shape[1]):
                    pj = phon[:, j:j+1]
                    for fsig in (f1_loc, f2_loc, f1_amp, f2_amp):
                        inter.append(pj * fsig)
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

            # --- 'v2pp' : v2z but DROP cube z^3 — pure ReLU+/ReLU- (138 raw). Tests
            # whether the cube tail-heavy term is regularization burden. 138 raw → 552 post-delay.
            if self.interact_set == 'v2pp':
                key_sigs10 = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                              hnr_peak, long_centroid, flat, hnr_dvox,
                              f12_disp, f12_amp]
                for sig in key_sigs10:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                for sig in key_sigs10:
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                for j in range(phon.shape[1]):
                    sig = phon[:, j:j+1]
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                for sig in (f1_loc, f2_loc, f1_amp, f2_amp):
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
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

            # --- 'v2qq' : v2z + per-band RAW (not dev) z-scored asym on log_e_long.
            # Captures "above/below average" band activity as named features.
            # 20 bands × 3 asym = 60 new + v2z(207) = 267 raw → 1068 post-delay.
            if self.interact_set == 'v2qq':
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
                # PER-BAND RAW z-scored asym (no detrending)
                m = log_e_long.mean(dim=0, keepdim=True)
                s = log_e_long.std(dim=0, keepdim=True).clamp(min=1e-6)
                z_band = (log_e_long - m) / s
                inter.append(z_band.clamp(min=0.0))
                inter.append((-z_band).clamp(min=0.0))
                inter.append(z_band * z_band * z_band)
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

            # --- 'v2rr' : v2z with finer taus (3, 9, 30) instead of (4.5, 15, 60).
            # Same dim count (207 raw → 828 post-delay). Tests whether our 3-tau
            # choice is at the right time scales.
            if self.interact_set == 'v2rr':
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
                    for tau in (3.0, 9.0, 30.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for j in range(phon.shape[1]):
                    sig = phon[:, j:j+1]
                    for tau in (3.0, 9.0, 30.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for sig in (f1_loc, f2_loc, f1_amp, f2_amp):
                    for tau in (3.0, 9.0, 30.0):
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

            # --- 'v2ss' : v2z + CROSS-TAU DEV PRODUCTS (hierarchical coupling).
            # For each of 6 key sigs (loud, strength, logF0, flux, voicing, flat),
            # compute dev at 3 taus (4.5/15/60) and form 3 cross-tau products:
            # dev_phonetic × dev_syllabic, dev_syllabic × dev_prosodic,
            # dev_phonetic × dev_prosodic. Captures hierarchical AM coupling
            # between phonetic-rate and prosodic-rate variation. 6 × 3 = 18 new.
            # 225 raw → 900 post-delay.
            if self.interact_set == 'v2ss':
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
                # CROSS-TAU DEV PRODUCTS
                cross_sigs = [long_loud_1d, best_strength, log_f0,
                              flux_abs_long, hnr_peak, flat]
                for sig in cross_sigs:
                    devs = []
                    for tau in (4.5, 15.0, 60.0):
                        d = sig - leaky_lowpass(sig, tau_frames=tau)
                        s = d.std(dim=0, keepdim=True).clamp(min=1e-6)
                        devs.append((d - d.mean(dim=0, keepdim=True)) / s)
                    # 3 cross-tau products
                    inter.append(devs[0] * devs[1])
                    inter.append(devs[1] * devs[2])
                    inter.append(devs[0] * devs[2])

            # --- 'v2tt' : v2z + ONSET-GATED asym. Gate key sigs by positive flux
            # (a normalized onset indicator); apply asym to the gated signal.
            # Captures "asymmetric value, only when there's an acoustic onset".
            # 10 sigs × 3 asym = 30 new. 237 raw → 948 post-delay.
            if self.interact_set == 'v2tt':
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
                # ONSET-GATED asym: gate = z-scored positive flux
                gate = flux_pos_long
                gm = gate.mean(dim=0, keepdim=True)
                gs = gate.std(dim=0, keepdim=True).clamp(min=1e-6)
                gz = ((gate - gm) / gs).clamp(min=0.0)
                for sig in key_sigs10:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(gz * z.clamp(min=0.0))
                    inter.append(gz * (-z).clamp(min=0.0))
                    inter.append(gz * z * z * z)

            # --- 'v2uu' : v2z + DERIVATIVE CROSS-PRODUCTS (prosodic coupling).
            # f0_d1 × long_loud, f0_d1 × hnr_d, hnr_d × long_loud, flux × hnr_d,
            # f0_d2 × loud, plus their multi-scale dev asym (3 taus, 3 asym).
            # Captures pitch-acceleration / loudness / voicing coupling.
            # ~5 prods + 6 sigs × 3 taus × 3 asym = 5 + 54 = 59 new (but 5 prods only).
            # Lean version: just the 5 named prods + asym versions = ~20 new.
            # 227 raw → 908 post-delay.
            if self.interact_set == 'v2uu':
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
                # DERIVATIVE CROSS-PRODUCTS (prosodic coupling): each product as a
                # raw value + signed-rectified components (3-fold asym), captures
                # rising/falling synchrony between modalities.
                loud_d = torch.zeros_like(long_loud_1d)
                loud_d[1:] = long_loud_1d[1:] - long_loud_1d[:-1]
                prod_pairs = [
                    f0_d1 * long_loud_1d,    # pitch-velocity × loudness
                    f0_d1 * hnr_d,           # pitch-velocity × voicing change
                    hnr_d * long_loud_1d,    # voicing change × loudness
                    hnr_d * flux_abs_long,   # voicing change × flux
                    loud_d * flux_abs_long,  # loudness change × flux
                    loud_d * best_strength,  # loudness change × pitch strength
                    f0_d2 * long_loud_1d,    # pitch acceleration × loudness
                ]
                for p in prod_pairs:
                    m = p.mean(dim=0, keepdim=True)
                    s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (p - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)

            # --- 'v2vv' : v2z + MULTI-SCALE CROSS-LAG (k=2,4,8 ≈ 67/133/267ms).
            # v2j 6 pair types × 3 lags = 18 cross-lag products. Phonetic / syllabic
            # / word-onset temporal coupling. 225 raw → 900 post-delay.
            if self.interact_set == 'v2vv':
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
                # Multi-lag cross-products replacing v2j (now 6 pairs × 3 lags = 18)
                def _lag(sig, k):
                    return torch.cat([sig[:k], sig[:-k]], dim=0)
                def _fwd(sig, k):
                    return torch.cat([sig[k:], sig[-1:].expand(k, -1)], dim=0)
                for k in (2, 4, 8):
                    inter.append(_lag(flux_abs_long, k) * hnr_d)
                    inter.append(_lag(hnr_d, k) * flux_abs_long)
                    inter.append(_lag(flux_abs_long, k) * long_loud_1d)
                    inter.append(best_strength * _fwd(f0_d1, k))
                    inter.append(long_loud_1d * _fwd(flux_abs_long, k))
                    inter.append(hnr_peak * _fwd(1.0 - hnr_peak, k))

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


model_shorthand_name = "INTERP-v2vv-interact"
model_description = (
    "v2vv: v2z + MULTI-SCALE CROSS-LAG. v2j 6 pairs × 3 lags (k=2,4,8 ≈ 67/133/267ms) "
    "= 18 cross-lag products (phonetic / syllabic / word-onset). 225 raw → 900 "
    "post-delay. STUB:"
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
    parser.add_argument("--interact-set", default="v2z")
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
