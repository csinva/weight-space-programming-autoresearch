"""
PHONOLOGICAL paradigm v7 — LEAN backbone + peakRate-gated sample-and-hold.

Strategy: previous iterations overfit (test_corr plateau at 0.067 with train
climbing past 0.099). Strip to minimal effective phon features and add one
genuinely novel idea:

  * peakRate-gated phonological SAMPLE-AND-HOLD: STG is known to entrain to
    acoustic edges (Oganian & Chang 2019, Khalighinejad 2017). At each
    envelope-onset landmark, we SNAPSHOT the current phonological state
    (manner, place, formant location, voicing) and hold that snapshot
    (with exponential decay) until the next landmark. This makes phon
    features stable WITHIN a phoneme but discretely change at edges —
    matching the STG response profile better than instantaneous values.

Plus: lean baseline phon detector set, plus pitch/HNR, plus contrastive
relative intonation, plus phonetic template matches (single mid scale).
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_C"
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


def gammatone_powerbank(win, n_filters, f_min=80.0, f_max=8000.0, order=4):
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32)), fcs


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
    return torch.flip(leaky_lowpass(torch.flip(x, dims=[0]), tau_frames), dims=[0])


# ---------------------------------------------------------------------------
# Spectrum / pitch utilities
# ---------------------------------------------------------------------------

def harmonic_comb(power: torch.Tensor, win: int, f0_cands: torch.Tensor,
                   n_harm: int = 10) -> torch.Tensor:
    freq_res = SR / win
    log_p = torch.log(torch.clamp(power, min=1e-10))
    tot = log_p.mean(dim=1, keepdim=True)
    out = []
    for f0 in f0_cands:
        ks = torch.arange(1, n_harm + 1, device=power.device, dtype=torch.float32)
        bins = (f0 * ks / freq_res).long().clamp(max=power.shape[1] - 1)
        out.append(log_p[:, bins].mean(dim=1, keepdim=True) - tot)
    return torch.cat(out, dim=1)


def hnr_from_acf(power: torch.Tensor, win: int,
                  f_min: float = 75.0, f_max: float = 450.0) -> torch.Tensor:
    full = torch.fft.irfft(power, n=win)
    acf0 = full[:, :1] + 1e-12
    acf_n = full / acf0
    lag_min = max(1, int(SR / f_max))
    lag_max = min(win - 1, int(SR / f_min))
    region = acf_n[:, lag_min:lag_max + 1]
    peak_val, peak_lag_rel = region.max(dim=1, keepdim=True)
    peak_val = peak_val.clamp(min=0.0, max=0.999)
    hnr_db = 10.0 * torch.log10(peak_val.clamp(min=1e-3) /
                                  (1.0 - peak_val).clamp(min=1e-3))
    best_lag = (peak_lag_rel.float() + lag_min)
    best_f0 = SR / best_lag.clamp(min=1.0)
    log_f0 = torch.log(best_f0.clamp(min=20.0))
    return torch.cat([peak_val, hnr_db, log_f0], dim=1)


def spectral_moments(power: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    p = power / (power.sum(dim=1, keepdim=True) + 1e-12)
    mu = (p * freqs).sum(dim=1, keepdim=True)
    df = freqs - mu
    var = (p * df ** 2).sum(dim=1, keepdim=True)
    std = torch.sqrt(var + 1e-12)
    skew = (p * df ** 3).sum(dim=1, keepdim=True) / (std ** 3 + 1e-12)
    kurt = (p * df ** 4).sum(dim=1, keepdim=True) / (var ** 2 + 1e-12)
    log_p = torch.log(torch.clamp(power, min=1e-10))
    geo = log_p.mean(dim=1, keepdim=True)
    arith = torch.log(power.mean(dim=1, keepdim=True) + 1e-10)
    flatness = geo - arith
    cs = torch.cumsum(power, dim=1)
    total = cs[:, -1:] + 1e-12
    idx85 = (cs >= 0.85 * total).float().argmax(dim=1)
    idx95 = (cs >= 0.95 * total).float().argmax(dim=1)
    r85 = torch.log(freqs[idx85].clamp(min=1.0)).unsqueeze(1)
    r95 = torch.log(freqs[idx95].clamp(min=1.0)).unsqueeze(1)
    return torch.cat([torch.log(mu.clamp(min=1.0)), torch.log(std.clamp(min=1.0)),
                       skew.clamp(-10, 10), kurt.clamp(0, 50),
                       flatness, r85, r95], dim=1)


# ---------------------------------------------------------------------------
# Phonological detectors (LEAN set)
# ---------------------------------------------------------------------------

def _band(log_e, fcs, fl, fh):
    mask = (fcs >= fl) & (fcs <= fh)
    if not mask.any():
        return log_e[:, :1] * 0.0
    idx = torch.from_numpy(np.where(mask)[0]).long().to(log_e.device)
    return log_e[:, idx].mean(dim=1, keepdim=True)


def manner_detectors(log_e, fcs, hnr, comb_max):
    lo = _band(log_e, fcs, 80, 500)
    mid = _band(log_e, fcs, 500, 2500)
    hi = _band(log_e, fcs, 2500, 5000)
    very_hi = _band(log_e, fcs, 5000, 8000)
    full = log_e.mean(dim=1, keepdim=True)
    spread = log_e.std(dim=1, keepdim=True)
    peak_val = hnr[:, 0:1]
    voiced = peak_val * 2.0 + comb_max
    silence = -full
    vowel = (lo + mid) * 0.5 + voiced - spread * 0.5
    sonorant = (lo + mid) * 0.5 + voiced
    fricative = (hi + very_hi) * 0.5 - lo - 0.5 * voiced
    sibilant = very_hi - mid - 0.5 * voiced
    plosive = spread + (very_hi * 0.3) - voiced * 0.5
    nasal = lo - mid - hi * 0.5 + voiced * 0.3
    approximant = mid + voiced - spread * 0.3 - very_hi
    return torch.cat([vowel, sonorant, fricative, sibilant, plosive, nasal,
                       approximant, silence, voiced], dim=1)


def place_detectors(log_e, fcs):
    f1_low = _band(log_e, fcs, 250, 450)
    f1_high = _band(log_e, fcs, 550, 900)
    f2_low = _band(log_e, fcs, 700, 1100)
    f2_high = _band(log_e, fcs, 1700, 2700)
    high_burst = _band(log_e, fcs, 3500, 6500)
    low_burst = _band(log_e, fcs, 80, 500)
    return torch.cat([f2_high - f2_low, f2_low - f2_high,
                       f1_low - f1_high, f1_high - f1_low,
                       high_burst - low_burst, low_burst - high_burst], dim=1)


def formant_tracking(log_e, fcs):
    log_fcs = torch.from_numpy(np.log(fcs).astype(np.float32)).to(log_e.device)

    def soft_peak(fl, fh, beta=4.0):
        mask = (fcs >= fl) & (fcs <= fh)
        if not mask.any():
            return torch.zeros_like(log_e[:, :1]), torch.zeros_like(log_e[:, :1])
        idx = torch.from_numpy(np.where(mask)[0]).long().to(log_e.device)
        band = log_e[:, idx]
        w = torch.softmax(band * beta, dim=1)
        log_f_band = log_fcs[idx].view(1, -1)
        loc = (w * log_f_band).sum(dim=1, keepdim=True)
        amp = (w * band).sum(dim=1, keepdim=True)
        return loc, amp

    f1l, f1a = soft_peak(200, 900)
    f2l, f2a = soft_peak(800, 2700)
    f3l, f3a = soft_peak(2200, 3800)
    return torch.cat([f1l, f2l, f3l, f1a, f2a, f3a,
                       f2l - f1l, f3l - f2l, f3l - f1l], dim=1)


def peakrate_train(loudness: torch.Tensor, tau: float = 5.0) -> torch.Tensor:
    """Sparse peakRate landmark train. Returns (T, 1) with non-zero spikes
    where d/dt smoothed loudness is a positive local maximum."""
    smooth = leaky_lowpass(loudness, tau)
    d = torch.zeros_like(smooth)
    d[1:] = smooth[1:] - smooth[:-1]
    d_prev = torch.cat([d[:1], d[:-1]], dim=0)
    d_next = torch.cat([d[1:], d[-1:]], dim=0)
    is_max = ((d > d_prev) & (d > d_next) & (d > 0)).float()
    return is_max * d


def relative_intonation(log_f0, voiced_mask, taus=(15.0, 60.0, 150.0)):
    out = [log_f0, voiced_mask]
    d1 = torch.zeros_like(log_f0); d1[1:] = log_f0[1:] - log_f0[:-1]
    d2 = torch.zeros_like(log_f0); d2[1:] = d1[1:] - d1[:-1]
    d1, d2 = d1 * voiced_mask, d2 * voiced_mask
    out += [d1, d2, d1.clamp(min=0.0)]
    for tau in taus:
        wsum = leaky_lowpass(voiced_mask, tau)
        mu = leaky_lowpass(log_f0 * voiced_mask, tau) / (wsum + 1e-3)
        sq = leaky_lowpass((log_f0 ** 2) * voiced_mask, tau)
        var = (sq / (wsum + 1e-3) - mu ** 2).clamp(min=1e-6)
        sigma = torch.sqrt(var)
        out.append(torch.cat([(log_f0 - mu) * voiced_mask,
                                ((log_f0 - mu) / sigma) * voiced_mask,
                                mu, sigma], dim=1))
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# Phonetic 2-D TEMPLATE matched filters
# ---------------------------------------------------------------------------

def _gauss(fnorm, mu, sigma):
    return np.exp(-((fnorm - mu) / sigma) ** 2)


def make_phonetic_templates(n_bands: int, n_time: int, fcs: np.ndarray):
    """Compact 14-prototype hand-designed phoneme template bank."""
    T, F = n_time, n_bands
    fcs_log = np.log(fcs)
    fnorm = (fcs_log - fcs_log.min()) / (fcs_log.max() - fcs_log.min() + 1e-9)
    mid = T // 2
    protos = []

    # Vowels
    for vc in [(0.10, 0.55, "i"), (0.10, 0.25, "u"), (0.25, 0.40, "a"),
                (0.20, 0.40, "schwa"), (0.15, 0.50, "e")]:
        f1c, f2c, _ = vc
        p = np.zeros((T, F))
        for t in range(T):
            p[t] = _gauss(fnorm, f1c, 0.08) + 0.7 * _gauss(fnorm, f2c, 0.08)
        protos.append(p)

    # Fricatives
    for fl, fh, _ in [(0.95, 0.05, "s"), (0.78, 0.08, "sh"),
                       (0.65, 0.25, "f"), (0.5, 0.40, "h")]:
        p = np.zeros((T, F))
        amp = 1.0 if _ != "h" else 0.3
        for t in range(T):
            p[t] = amp * _gauss(fnorm, fl, fh)
        protos.append(p)

    # Stops (burst + silence)
    for burst_loc, _ in [(0.30, "p"), (0.75, "t"), (0.55, "k")]:
        p = np.zeros((T, F))
        p[mid] = _gauss(fnorm, burst_loc, 0.20)
        p[:mid] = -0.3
        protos.append(p)

    # Nasal
    p = np.zeros((T, F))
    for t in range(T):
        p[t] = _gauss(fnorm, 0.05, 0.08) - 0.5 * _gauss(fnorm, 0.35, 0.15)
    protos.append(p)

    # /r/ approximant
    p = np.zeros((T, F))
    for t in range(T):
        p[t] = (_gauss(fnorm, 0.18, 0.08) + 0.6 * _gauss(fnorm, 0.42, 0.08)
                 - 0.5 * _gauss(fnorm, 0.62, 0.06))
    protos.append(p)

    proto_arr = np.stack(protos, axis=0)
    proto_arr -= proto_arr.mean(axis=(1, 2), keepdims=True)
    norms = np.linalg.norm(proto_arr.reshape(proto_arr.shape[0], -1), axis=1)
    proto_arr /= (norms[:, None, None] + 1e-9)
    return torch.from_numpy(proto_arr.astype(np.float32)).unsqueeze(1)


def phonetic_template_match(log_e: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    Tt = templates.shape[2]
    pad_t = Tt // 2
    x = log_e.unsqueeze(0).unsqueeze(0)
    y = torch.nn.functional.conv2d(x, templates, padding=(pad_t, 0))
    return y.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# STMF (Chi-Shamma 2-D spectro-temporal Gabors) on log cochleagram
# ---------------------------------------------------------------------------

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


def make_stmf_kernels(rates, device, fs=30.0):
    """Pre-build STMF 2-D Gabor cos/sin kernels for each rate-scale pair."""
    kernels = []
    for omega_t, omega_f in rates:
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
        cos_k -= cos_k.mean()
        sin_k -= sin_k.mean()
        kernels.append((
            torch.from_numpy(cos_k).to(device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1),
            torch.from_numpy(sin_k).to(device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1),
            Lt, Lf,
        ))
    return kernels


def make_gabor_1d(rate_hz, fs=30.0):
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
    return cos_k.astype(np.float32), sin_k.astype(np.float32)


def gabor_envelope_1d(x: torch.Tensor, cos_k_t: torch.Tensor,
                       sin_k_t: torch.Tensor) -> torch.Tensor:
    """Gabor bandpass envelope on a (T, C) signal along time."""
    L = cos_k_t.numel()
    pad = L // 2
    C = x.shape[1]
    xp = x.transpose(0, 1).unsqueeze(0)
    ck = cos_k_t.view(1, 1, -1).expand(C, 1, L).contiguous()
    sk = sin_k_t.view(1, 1, -1).expand(C, 1, L).contiguous()
    xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    c = torch.nn.functional.conv1d(xp_p, ck, groups=C)[..., :x.shape[0]]
    s = torch.nn.functional.conv1d(xp_p, sk, groups=C)[..., :x.shape[0]]
    return torch.sqrt(c * c + s * s + 1e-12).squeeze(0).transpose(0, 1).contiguous()


def stmf_envelope(log_e: torch.Tensor, cos_k, sin_k, Lt, Lf) -> torch.Tensor:
    """Apply a single STMF 2-D Gabor and return the envelope (T, F)."""
    xp = log_e.unsqueeze(0).unsqueeze(0)
    c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
    s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
    T, F = log_e.shape
    env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
    return env[:T, :F]


# ---------------------------------------------------------------------------
# v8 NEW: Cardinal vowel RBF similarity + vowel-space trajectory topology
# ---------------------------------------------------------------------------

# Cardinal vowel anchor positions in (log_F1_Hz, log_F2_Hz) space (adult averages)
CARDINAL_VOWELS = {
    "i":   (np.log(280.0),  np.log(2250.0)),   # high front
    "I":   (np.log(400.0),  np.log(1920.0)),
    "e":   (np.log(450.0),  np.log(2050.0)),
    "E":   (np.log(550.0),  np.log(1770.0)),
    "ae":  (np.log(690.0),  np.log(1660.0)),   # low front
    "a":   (np.log(700.0),  np.log(1100.0)),   # low central/back
    "o":   (np.log(470.0),  np.log(870.0)),    # mid back
    "u":   (np.log(310.0),  np.log(870.0)),    # high back
    "U":   (np.log(450.0),  np.log(1100.0)),
    "schwa": (np.log(500.0), np.log(1500.0)),  # neutral
}


def cardinal_vowel_similarity(f1l: torch.Tensor, f2l: torch.Tensor,
                                voicing: torch.Tensor,
                                sigma_log_hz: float = 0.25) -> torch.Tensor:
    """RBF similarity to each cardinal vowel anchor in (log F1, log F2) space.

    f1l, f2l: (T, 1) log-Hz formant locations.
    voicing: (T, 1) [0..1] voicing gate.
    Returns (T, n_cardinal) similarity scores, each in [0,1] gated by voicing.
    """
    out = []
    for name, (mu_f1, mu_f2) in CARDINAL_VOWELS.items():
        d2 = ((f1l - mu_f1) ** 2 + (f2l - mu_f2) ** 2) / (sigma_log_hz ** 2)
        sim = torch.exp(-d2)
        out.append(sim * voicing)
    return torch.cat(out, dim=1)


def ipa_vowel_grid(f1l: torch.Tensor, f2l: torch.Tensor, voicing: torch.Tensor,
                     n_f1: int = 6, n_f2: int = 6, sigma: float = 0.20) -> torch.Tensor:
    """IPA-topographic Gaussian receptive-field grid over the vowel chart.

    Grid spans typical F1 in [200, 900] Hz and F2 in [700, 2700] Hz (log).
    Each cell is a 2-D Gaussian RBF; the response is sim_ij(t) gated by voicing.
    Returns (T, n_f1*n_f2)."""
    f1_min, f1_max = np.log(200.0), np.log(900.0)
    f2_min, f2_max = np.log(700.0), np.log(2700.0)
    f1_centers = np.linspace(f1_min, f1_max, n_f1)
    f2_centers = np.linspace(f2_min, f2_max, n_f2)
    out = []
    for f1c in f1_centers:
        for f2c in f2_centers:
            d = ((f1l - f1c) ** 2 + (f2l - f2c) ** 2) / (sigma ** 2)
            out.append(torch.exp(-d) * voicing)
    return torch.cat(out, dim=1)


def ipa_consonant_grid(manner: torch.Tensor, place: torch.Tensor,
                         silence_gate: torch.Tensor,
                         n_m: int = 4, n_p: int = 4, sigma: float = 1.5) -> torch.Tensor:
    """Topographic Gaussian RBF code over a (manner, place) 4×4 chart.

    manner: (T, 9) detector outputs. We project to a 1-D "manner score":
        manner_score = soft-argmax position over manner detectors.
    place:  similarly project to a 1-D "place score".
    Then 2D RBF over the (manner_score, place_score) plane gated by 1-silence.
    Returns (T, n_m*n_p)."""
    # Soft positions
    m_w = torch.softmax(manner * 2.0, dim=1)
    p_w = torch.softmax(place * 2.0, dim=1)
    m_pos_idx = torch.arange(manner.shape[1], device=manner.device,
                              dtype=torch.float32).view(1, -1)
    p_pos_idx = torch.arange(place.shape[1], device=place.device,
                              dtype=torch.float32).view(1, -1)
    m_score = (m_w * m_pos_idx).sum(dim=1, keepdim=True)
    p_score = (p_w * p_pos_idx).sum(dim=1, keepdim=True)
    m_centers = np.linspace(0, manner.shape[1] - 1, n_m)
    p_centers = np.linspace(0, place.shape[1] - 1, n_p)
    out = []
    for mc in m_centers:
        for pc in p_centers:
            d = ((m_score - mc) ** 2 + (p_score - pc) ** 2) / (sigma ** 2)
            out.append(torch.exp(-d) * silence_gate)
    return torch.cat(out, dim=1)


def vowel_space_trajectory(f1l: torch.Tensor, f2l: torch.Tensor,
                             voicing: torch.Tensor) -> torch.Tensor:
    """Vowel-space topology features.

    pos = (f1l, f2l)
    velocity = d/dt pos
    speed = ||velocity||
    angle = atan2(d_f2, d_f1) encoded as (sin, cos)
    curvature ~= |d2 pos| / (|d pos|^3 + eps)

    Returns (T, 8) gated by voicing.
    """
    d_f1 = torch.zeros_like(f1l); d_f1[1:] = f1l[1:] - f1l[:-1]
    d_f2 = torch.zeros_like(f2l); d_f2[1:] = f2l[1:] - f2l[:-1]
    speed = torch.sqrt(d_f1 ** 2 + d_f2 ** 2 + 1e-9)
    angle = torch.atan2(d_f2, d_f1)
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    dd_f1 = torch.zeros_like(d_f1); dd_f1[1:] = d_f1[1:] - d_f1[:-1]
    dd_f2 = torch.zeros_like(d_f2); dd_f2[1:] = d_f2[1:] - d_f2[:-1]
    curv = (d_f1 * dd_f2 - d_f2 * dd_f1) / (speed ** 3 + 1e-6)
    curv = curv.clamp(-20, 20)
    # Sparse "gestural endpoint" event: local minimum of speed (vowel target)
    s_prev = torch.cat([speed[:1], speed[:-1]], dim=0)
    s_next = torch.cat([speed[1:], speed[-1:]], dim=0)
    is_min = ((speed < s_prev) & (speed < s_next)).float()
    return torch.cat([speed * voicing, sin_a * voicing, cos_a * voicing,
                       curv * voicing, is_min * voicing,
                       d_f1 * voicing, d_f2 * voicing,
                       speed], dim=1)


# ---------------------------------------------------------------------------
# NEW v7: peakRate-gated SAMPLE-AND-HOLD of phonological features
# ---------------------------------------------------------------------------

def peakrate_sample_hold(phon_feat: torch.Tensor, peak_train: torch.Tensor,
                          decay_tau: float = 12.0,
                          threshold_quantile: float = 0.7) -> torch.Tensor:
    """SAMPLE-AND-HOLD phonological features at peakRate landmarks.

    For each time t, this returns the phon_feat value at the most recent
    peakRate landmark before t, decayed by exp(-Δt / decay_tau) frames.

    This turns continuous phon features into step-function-like signals
    that are constant within a phoneme (between landmarks) and change
    discretely at edges — matching the STG response to acoustic edges.

    Implementation: a vectorized exponential causal recurrence:
        held[t] = α_t * phon[t] + (1 - α_t) * held[t-1] * decay
    where α_t = strong if t is a landmark, weak otherwise.

    We avoid Python time loops by using a CUMULATIVE recurrence trick:
    let g[t] = 1 if landmark fires at t else decay_factor.
    Then held[t] = phon[landmark_t] * (decay_factor)^(t - landmark_t).
    This is implementable as:
        running_geom = cumprod of g; flip-cumsum-flip.

    Returns (T, n_feat) sample-and-hold feature stream.
    """
    T, F = phon_feat.shape
    device = phon_feat.device
    # Soft landmark gate: peak_train is positive at landmarks, zero else.
    # Convert to a 0..1 weight
    p = peak_train.squeeze(-1) if peak_train.dim() == 2 and peak_train.shape[1] == 1 else peak_train
    p = p / (p.abs().max() + 1e-9)
    # Mask: landmark = 1 where peak_train > threshold, else 0
    # Use percentile threshold relative to non-zero spikes
    nz = p[p > 0]
    if nz.numel() > 0:
        thr = torch.quantile(nz, threshold_quantile)
    else:
        thr = torch.tensor(0.0, device=device)
    landmark = (p > thr).float().view(T, 1)  # (T,1)

    # Compute "time since last landmark" via cumulative sum reset trick.
    # idx vector: for each t, find largest s<=t where landmark[s] > 0.
    # Use a running argmax approach.
    # Easier: build sample-and-hold via per-frame iteration in vectorized form
    # using a manual EMA with adaptive alpha:
    #   alpha[t] = 1 if landmark[t] else 0
    #   held[t] = alpha[t] * phon[t] + (1 - alpha[t]) * decay * held[t-1]
    # The recurrence is linear with time-varying coefficient — must be done
    # with a python loop (T is ~3500 max per chunk, fine).
    decay = float(np.exp(-1.0 / decay_tau))
    held = torch.zeros_like(phon_feat)
    cur = phon_feat[0:1].clone()
    for t in range(T):
        a = landmark[t:t + 1]
        cur = a * phon_feat[t:t + 1] + (1.0 - a) * decay * cur
        held[t:t + 1] = cur
    return held


def peakrate_sample_hold_fast(phon_feat: torch.Tensor, peak_train: torch.Tensor,
                                decay_tau: float = 12.0,
                                threshold_quantile: float = 0.7) -> torch.Tensor:
    """Vectorized variant of peakrate_sample_hold using prefix-product trick.

    For each time t:
        idx(t)   = max s <= t with landmark[s] = 1
        held[t]  = phon[idx(t)] * decay^(t - idx(t))

    Vectorized:
      • landmark_t boolean: 1 at landmarks
      • Compute "last landmark index" via cumulative max of (t * landmark)
      • Gather phon[idx(t)] via index_select
      • Compute decay^(t - idx(t))
    """
    T, F = phon_feat.shape
    device = phon_feat.device
    p = peak_train.squeeze(-1) if peak_train.dim() == 2 and peak_train.shape[1] == 1 else peak_train
    p = p / (p.abs().max() + 1e-9)
    nz = p[p > 0]
    if nz.numel() > 0:
        thr = torch.quantile(nz, threshold_quantile)
    else:
        thr = torch.tensor(0.0, device=device)
    landmark = (p > thr)  # (T,) bool

    # Last landmark index up to t
    tarange = torch.arange(T, device=device)
    masked_idx = torch.where(landmark, tarange, torch.full_like(tarange, -1))
    last_idx, _ = masked_idx.cummax(dim=0)  # (T,) — last landmark idx, or -1 if none
    # Where no landmark yet, fall back to idx=0
    last_idx_safe = last_idx.clamp(min=0)
    held = phon_feat[last_idx_safe]  # (T, F)
    # Apply decay by elapsed frames
    elapsed = (tarange - last_idx_safe).float()
    elapsed = torch.where(last_idx < 0, torch.zeros_like(elapsed), elapsed)
    decay = float(np.exp(-1.0 / decay_tau))
    held = held * (decay ** elapsed).unsqueeze(1)
    # If no landmark ever fired before t, set held=0
    held = held * (last_idx >= 0).float().unsqueeze(1)
    return held


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0
    WIN_LIST = (1024, 2048, 4096)
    N_FILT = 40
    F0_CANDIDATES = (75.0, 90.0, 110.0, 135.0, 165.0, 200.0, 245.0,
                      295.0, 355.0, 425.0)
    ADAPT_TAUS = (1.5, 4.5, 15.0, 60.0, 150.0)

    def __init__(self, device: str = "cuda"):
        self.device = device
        fbs = []
        self.fcs = None
        for w in self.WIN_LIST:
            fb, fcs = gammatone_powerbank(w, n_filters=self.N_FILT)
            fbs.append(fb.to(device))
            if w == 4096:
                self.fcs = fcs.astype(np.float32)
        self.fbs = fbs
        # Coarse 20-band cochleagram for STMF (matches Chi-Shamma scale)
        self.fb_stmf, self.fcs_stmf = gammatone_powerbank(4096, n_filters=20)
        self.fb_stmf = self.fb_stmf.to(device)
        self.fcs_stmf = self.fcs_stmf.astype(np.float32)
        self.long_freqs = torch.from_numpy(
            np.fft.rfftfreq(4096, d=1.0 / SR).astype(np.float32)).to(device)
        self.f0_cands = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32,
                                       device=device)
        self.templates = make_phonetic_templates(self.N_FILT, n_time=9,
                                                   fcs=self.fcs).to(device)
        self.stmf_kernels = make_stmf_kernels(STMF_RATES, device=device)
        # Gabor 1-D modulation kernels for prosodic-rate envelope features
        self.gabor_1d = {}
        for rate in (2.0, 4.0, 6.0, 10.0):
            c, s = make_gabor_1d(rate)
            self.gabor_1d[rate] = (torch.from_numpy(c).to(device),
                                    torch.from_numpy(s).to(device))

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []

        # ====== Multi-window cochleagram + adaptation deltas (BACKBONE) ======
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
            for tau in self.ADAPT_TAUS:
                feats.append(log_e - leaky_lowpass(log_e, tau_frames=tau))

        p_long = powers[-1]
        log_e_long = log_es[-1]

        # ====== Pitch / voicing ======
        comb = harmonic_comb(p_long, 4096, self.f0_cands, n_harm=10)
        comb_max, _ = comb.max(dim=1, keepdim=True)
        hnr_long = hnr_from_acf(p_long, 4096, f_min=75.0, f_max=450.0)
        voicing = torch.sigmoid((hnr_long[:, 0:1] - 0.4) * 8.0)
        log_f0 = hnr_long[:, 2:3]
        feats += [comb, comb_max, hnr_long, voicing]

        # ====== Phonological detectors ======
        manner = manner_detectors(log_e_long, self.fcs, hnr_long, comb_max)
        place = place_detectors(log_e_long, self.fcs)
        fmt = formant_tracking(log_e_long, self.fcs)
        feats += [manner, place, fmt]

        # Spectral moments
        feats.append(spectral_moments(p_long, self.long_freqs))

        # ====== Loudness + peakRate ======
        loud_long = log_e_long.mean(dim=1, keepdim=True)
        feats.append(loud_long)
        # peakRate train at one tau (sparse landmark train)
        peakRate = peakrate_train(loud_long, tau=5.0)
        feats.append(peakRate)
        feats.append(leaky_lowpass(peakRate, 4.0))
        feats.append(leaky_lowpass(peakRate, 12.0))

        # Loudness derivatives (continuous)
        d_loud = torch.zeros_like(loud_long); d_loud[1:] = loud_long[1:] - loud_long[:-1]
        feats += [d_loud, d_loud.clamp(min=0.0), (-d_loud).clamp(min=0.0)]

        # ====== Relative / speaker-normalized intonation ======
        feats.append(relative_intonation(log_f0, voicing, taus=(15.0, 60.0, 150.0)))

        # ====== Contrastive formants (multi-tau) ======
        f_loc = fmt[:, :3]; f_amp = fmt[:, 3:6]; f_diff = fmt[:, 6:9]
        for tau in (15.0, 60.0, 150.0):
            wsum = leaky_lowpass(voicing, tau)
            mu_f_loc = leaky_lowpass(f_loc * voicing, tau) / (wsum + 1e-3)
            feats.append((f_loc - mu_f_loc) * voicing)
        f_d1 = torch.zeros_like(f_loc); f_d1[1:] = f_loc[1:] - f_loc[:-1]
        feats += [f_d1 * voicing, f_d1.clamp(min=0), (-f_d1).clamp(min=0)]

        # ====== Spectral flux ======
        d = torch.zeros_like(log_e_long); d[1:] = log_e_long[1:] - log_e_long[:-1]
        feats.append(torch.cat([d.clamp(min=0.0).sum(dim=1, keepdim=True),
                                 (-d).clamp(min=0.0).sum(dim=1, keepdim=True),
                                 d.abs().sum(dim=1, keepdim=True)], dim=1))

        # ====== Slow tonic context ======
        for tau in (15.0, 60.0):
            feats += [leaky_lowpass(manner, tau),
                       leaky_lowpass(place, tau),
                       leaky_lowpass(fmt[:, :6], tau)]
        feats.append(acausal_leaky_lowpass(log_e_long, tau_frames=60.0))

        # ====== Phonetic template matching ======
        tm = phonetic_template_match(log_e_long, self.templates)
        feats.append(tm)
        tm_d = torch.zeros_like(tm); tm_d[1:] = tm[1:] - tm[:-1]
        feats += [tm_d.clamp(min=0.0), (-tm_d).clamp(min=0.0)]
        feats.append(leaky_lowpass(tm, 15.0))

        # ====== [v8] STMF Chi-Shamma 2-D Gabors on coarse cochleagram ======
        log_e_stmf = torch.log(torch.clamp(p_long @ self.fb_stmf, min=1e-10))
        feats.append(log_e_stmf)  # 20
        for cos_k, sin_k, Lt, Lf in self.stmf_kernels:
            feats.append(stmf_envelope(log_e_stmf, cos_k, sin_k, Lt, Lf))

        # ====== [v9] Cube-root cochleagram + adaptation deltas ======
        cube_long = torch.pow(p_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)
        feats.append(cube_long)
        for tau in (4.5, 15.0, 60.0):
            feats.append(cube_long - leaky_lowpass(cube_long, tau_frames=tau))

        # ====== [v9] Future + past per-band deltas at lags 3, 6, 12 ======
        for k in (3, 6, 12):
            fut = torch.cat([log_e_long[k:], log_e_long[-1:].expand(k, -1)], dim=0)
            past = torch.cat([log_e_long[:1].expand(k, -1), log_e_long[:-k]], dim=0)
            fut_d = fut - log_e_long
            past_d = log_e_long - past
            feats += [fut_d, fut_d.clamp(min=0.0), (-fut_d).clamp(min=0.0), past_d]
        # Cube-root future delta at lag 6
        fut_c = torch.cat([cube_long[6:], cube_long[-1:].expand(6, -1)], dim=0)
        feats.append(fut_c - cube_long)

        # ====== [v9] Per-band modulation depth (HWR of high-pass envelope) ======
        for hp_tau, lp_tau in ((4.5, 15.0), (1.5, 15.0)):
            hp = log_e_long - leaky_lowpass(log_e_long, tau_frames=hp_tau)
            feats.append(leaky_lowpass(hp.abs(), tau_frames=lp_tau))

        # ====== [v9] Gabor 1-D modulation envelopes on 1-D prosodic signals ======
        log_e_w = torch.softmax(log_e_long, dim=1)
        ch_idx = torch.arange(self.N_FILT, dtype=torch.float32,
                                device=log_e_long.device).view(1, -1)
        centroid = (log_e_w * ch_idx).sum(dim=1, keepdim=True)
        mod_signals = [loud_long, centroid, hnr_long, comb_max, log_f0,
                        manner, place, fmt[:, :6]]
        mod_pack = torch.cat(mod_signals, dim=1)
        for rate, (ck, sk) in self.gabor_1d.items():
            feats.append(gabor_envelope_1d(mod_pack, ck, sk))

        # ====== [v8 NEW] Cardinal vowel RBF similarity ======
        f1l = fmt[:, 0:1]
        f2l = fmt[:, 1:2]
        cv_sim = cardinal_vowel_similarity(f1l, f2l, voicing)
        feats.append(cv_sim)  # 10
        # Multi-sigma RBF for narrower / broader receptive fields
        for sigma in (0.10, 0.45):
            feats.append(cardinal_vowel_similarity(f1l, f2l, voicing,
                                                     sigma_log_hz=sigma))
        # Slow tonic of cardinal vowels (which vowel was sustained recently)
        feats.append(leaky_lowpass(cv_sim, 8.0))
        feats.append(leaky_lowpass(cv_sim, 30.0))

        # ====== [v8 NEW] Vowel-space trajectory topology ======
        vst = vowel_space_trajectory(f1l, f2l, voicing)
        feats.append(vst)  # 8
        feats.append(leaky_lowpass(vst, 8.0))

        # ====== [v9 NEW] IPA topographic population code ======
        # 6x6 Gaussian RBF grid over vowel chart (F1, F2)
        ipa_v = ipa_vowel_grid(f1l, f2l, voicing, n_f1=6, n_f2=6, sigma=0.20)
        feats.append(ipa_v)  # 36
        # Slow tonic (which vowel cell was recently active)
        feats.append(leaky_lowpass(ipa_v, 8.0))
        feats.append(leaky_lowpass(ipa_v, 30.0))
        # 4x4 manner-place grid for consonants (gated by inverse-silence)
        silence_gate = torch.sigmoid(-(manner[:, 7:8] * 4.0))  # 1 - silence
        ipa_c = ipa_consonant_grid(manner, place, silence_gate, n_m=4, n_p=4)
        feats.append(ipa_c)  # 16
        feats.append(leaky_lowpass(ipa_c, 8.0))

        # ====== [v7] peakRate-gated SAMPLE-AND-HOLD of phonological state ======
        # Bundle the core phonological state into a single vector:
        # (manner, place, formants, voicing, log_f0, template_match)
        phon_state = torch.cat([manner, place, fmt, voicing, log_f0, tm], dim=1)
        # Sample-and-hold at peakRate landmarks at 3 decay scales
        for decay_tau, thr_q in [(8.0, 0.5), (20.0, 0.5), (50.0, 0.5)]:
            sh = peakrate_sample_hold_fast(phon_state, peakRate,
                                             decay_tau=decay_tau,
                                             threshold_quantile=thr_q)
            feats.append(sh)
            # Also delta-from-held: how much current value differs from held snapshot
            feats.append(phon_state - sh)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "PHON-v9"
model_description = (
    "PHON v9: adds (a) cube-root cochleagram + adaptation deltas, (b) per-band "
    "future/past deltas at lags 3/6/12, (c) per-band modulation depth (HWR "
    "high-pass), (d) Gabor 1-D modulation envelopes on prosodic 1-D signals "
    "at 4 rates; PLUS new IPA topographic population code: 6x6 Gaussian RBF "
    "vowel chart + 4x4 manner-place consonant chart. v8=0.0678."
)


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
