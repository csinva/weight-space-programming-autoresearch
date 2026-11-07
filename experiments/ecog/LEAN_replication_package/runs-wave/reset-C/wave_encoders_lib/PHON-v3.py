"""
PHONOLOGICAL paradigm v3 — hybrid: lean cochleagram backbone (anchors ridge)
PLUS bold phonological structures on top.

v3 additions over v1 (PHON-v1 = 0.0645):
  • Cochleagram adaptation backbone (multi-tau deltas per band) — anchors the
    bottom-up acoustic signal that pure phon features dropped.
  • Phonetic 2-D TEMPLATE matched filters: closed-form (T, F) prototypes for
    /vowel-hold/, /stop-burst/, /fricative-plateau/, /nasal-murmur/,
    /vowel-to-vowel transition/, /stop-to-vowel onset/ convolved over the
    log cochleagram (so each frame gets a similarity score to each prototype
    phoneme class as a moving template).
  • Articulatory gesture-space trajectory: each frame mapped to a 5-D
    closed-form articulator space (tongue height, frontness, lip rounding,
    velum, voicing) → velocity, acceleration, peak-velocity events.
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
# Phonological detectors
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


def peakrate_landmarks(loudness, taus=(2.0, 5.0, 12.0)):
    out = []
    for tau in taus:
        smooth = leaky_lowpass(loudness, tau)
        d = torch.zeros_like(smooth)
        d[1:] = smooth[1:] - smooth[:-1]
        pr_pos = d.clamp(min=0.0)
        pr_neg = (-d).clamp(min=0.0)
        d_prev = torch.cat([d[:1], d[:-1]], dim=0)
        d_next = torch.cat([d[1:], d[-1:]], dim=0)
        is_max = ((d > d_prev) & (d > d_next) & (d > 0)).float()
        sparse_peak = is_max * d
        out.append(torch.cat([pr_pos, pr_neg, sparse_peak,
                               leaky_lowpass(sparse_peak, 1.5),
                               leaky_lowpass(sparse_peak, 4.0),
                               leaky_lowpass(sparse_peak, 12.0)], dim=1))
    return torch.cat(out, dim=1)


def relative_intonation(log_f0, voiced_mask, taus=(15.0, 60.0, 150.0, 400.0)):
    out = [log_f0, voiced_mask]
    d1 = torch.zeros_like(log_f0); d1[1:] = log_f0[1:] - log_f0[:-1]
    d2 = torch.zeros_like(log_f0); d2[1:] = d1[1:] - d1[:-1]
    d3 = torch.zeros_like(log_f0); d3[1:] = d2[1:] - d2[:-1]
    d1, d2, d3 = d1 * voiced_mask, d2 * voiced_mask, d3 * voiced_mask
    out += [d1, d2, d3, d1.clamp(min=0.0)]
    for tau in taus:
        wsum = leaky_lowpass(voiced_mask, tau)
        mu = leaky_lowpass(log_f0 * voiced_mask, tau) / (wsum + 1e-3)
        sq = leaky_lowpass((log_f0 ** 2) * voiced_mask, tau)
        var = (sq / (wsum + 1e-3) - mu ** 2).clamp(min=1e-6)
        sigma = torch.sqrt(var)
        rel = (log_f0 - mu) * voiced_mask
        z = ((log_f0 - mu) / sigma) * voiced_mask
        slope = torch.zeros_like(mu); slope[1:] = mu[1:] - mu[:-1]
        out.append(torch.cat([rel, z, mu, sigma, slope], dim=1))
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# NEW: Phonetic 2-D TEMPLATE matched filters
# ---------------------------------------------------------------------------

def make_phonetic_templates(n_bands: int, n_time: int = 9, fcs: np.ndarray = None):
    """Closed-form 2-D (time, frequency) phonetic prototype templates.

    Each template is a hand-shaped spectro-temporal pattern in log-energy
    space, designed to match prototypical phoneme classes. Convolving these
    with the log cochleagram gives a similarity score at every frame.

    Returns a (P, 1, n_time, n_bands) tensor (P prototypes).
    """
    T = n_time
    F = n_bands
    fcs_log = np.log(fcs)
    # Normalize freq axis to [0,1] across log-frequency
    fnorm = (fcs_log - fcs_log.min()) / (fcs_log.max() - fcs_log.min() + 1e-9)
    tnorm = (np.arange(T) - (T - 1) / 2) / max(1.0, (T - 1) / 2)

    protos = []

    # 1. Vowel HOLD: sustained low-F1 + mid-F2 peaks, stable over time
    p = np.zeros((T, F))
    f1_peak = np.exp(-((fnorm - 0.15) / 0.08) ** 2)
    f2_peak = np.exp(-((fnorm - 0.45) / 0.10) ** 2)
    for t in range(T):
        p[t] = f1_peak + 0.7 * f2_peak
    protos.append(p)

    # 2. Stop BURST: silence → broadband impulse at center
    p = np.zeros((T, F))
    p[T // 2] = np.exp(-((fnorm - 0.7) / 0.3) ** 2)  # high-freq burst
    p[:T // 2] = -0.3  # preceding silence
    protos.append(p)

    # 3. Fricative SUSTAIN: stable high-freq plateau
    p = np.zeros((T, F))
    for t in range(T):
        p[t] = np.exp(-((fnorm - 0.85) / 0.12) ** 2)
    protos.append(p)

    # 4. Nasal MURMUR: low-band sustained, weak/notched mid
    p = np.zeros((T, F))
    for t in range(T):
        p[t] = np.exp(-((fnorm - 0.05) / 0.08) ** 2) - 0.5 * np.exp(-((fnorm - 0.35) / 0.15) ** 2)
    protos.append(p)

    # 5. Vowel-to-vowel TRANSITION (F2 sweep up)
    p = np.zeros((T, F))
    for t in range(T):
        center = 0.30 + 0.25 * (t / max(1, T - 1))
        p[t] = np.exp(-((fnorm - center) / 0.12) ** 2)
    protos.append(p)

    # 6. Vowel-to-vowel TRANSITION (F2 sweep down)
    p = np.zeros((T, F))
    for t in range(T):
        center = 0.55 - 0.25 * (t / max(1, T - 1))
        p[t] = np.exp(-((fnorm - center) / 0.12) ** 2)
    protos.append(p)

    # 7. Stop-to-vowel ONSET: burst at t<T/2 then sustained low-F1
    p = np.zeros((T, F))
    mid = T // 2
    for t in range(T):
        if t < mid:
            p[t] = 0.6 * np.exp(-((fnorm - 0.75) / 0.25) ** 2)
        else:
            p[t] = np.exp(-((fnorm - 0.15) / 0.10) ** 2) + 0.5 * np.exp(-((fnorm - 0.45) / 0.12) ** 2)
    protos.append(p)

    # 8. Vowel-to-stop OFFSET: vowel then silence
    p = np.zeros((T, F))
    for t in range(T):
        if t < mid:
            p[t] = np.exp(-((fnorm - 0.20) / 0.12) ** 2)
        else:
            p[t] = -0.3
    protos.append(p)

    # 9. Approximant /l/-/r/ — F3 dip
    p = np.zeros((T, F))
    for t in range(T):
        p[t] = (np.exp(-((fnorm - 0.20) / 0.10) ** 2)
                + 0.6 * np.exp(-((fnorm - 0.50) / 0.10) ** 2)
                - 0.4 * np.exp(-((fnorm - 0.65) / 0.08) ** 2))
    protos.append(p)

    # 10. Glottal stop / creaky voice — narrow low-freq impulse train
    p = np.zeros((T, F))
    p[mid] = np.exp(-((fnorm - 0.05) / 0.08) ** 2)
    protos.append(p)

    # 11. Affricate /tʃ/ — burst then high-freq sibilant
    p = np.zeros((T, F))
    p[mid - 1] = np.exp(-((fnorm - 0.75) / 0.20) ** 2)  # sudden burst
    for t in range(mid, T):
        p[t] = np.exp(-((fnorm - 0.88) / 0.10) ** 2)
    protos.append(p)

    # 12. Diphthong /aj/ — F1 sweep down, F2 sweep up
    p = np.zeros((T, F))
    for t in range(T):
        f1c = 0.30 - 0.15 * (t / max(1, T - 1))
        f2c = 0.35 + 0.25 * (t / max(1, T - 1))
        p[t] = np.exp(-((fnorm - f1c) / 0.10) ** 2) + 0.7 * np.exp(-((fnorm - f2c) / 0.10) ** 2)
    protos.append(p)

    proto_arr = np.stack(protos, axis=0)  # (P, T, F)
    # Zero-mean each template (so DC offset doesn't dominate)
    proto_arr -= proto_arr.mean(axis=(1, 2), keepdims=True)
    # Normalize each prototype to unit Frobenius norm
    norms = np.linalg.norm(proto_arr.reshape(proto_arr.shape[0], -1), axis=1)
    proto_arr /= (norms[:, None, None] + 1e-9)
    return torch.from_numpy(proto_arr.astype(np.float32)).unsqueeze(1)  # (P, 1, T, F)


def phonetic_template_match(log_e: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    """Convolve log-cochleagram (T_in, F) with phonetic templates (P, 1, T_t, F).
    Returns (T_in, P) similarity scores at each frame (same-time padding)."""
    Tt = templates.shape[2]
    pad_t = Tt // 2
    x = log_e.unsqueeze(0).unsqueeze(0)  # (1, 1, T_in, F)
    # conv2d with kernel covering full freq dim -> output has freq dim = 1
    y = torch.nn.functional.conv2d(x, templates, padding=(pad_t, 0))
    # y shape: (1, P, T_in, 1)
    return y.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # (T_in, P)


# ---------------------------------------------------------------------------
# NEW: Articulatory gesture-space trajectory
# ---------------------------------------------------------------------------

def articulatory_trajectory(fmt: torch.Tensor, hnr: torch.Tensor,
                              manner: torch.Tensor, voicing: torch.Tensor) -> torch.Tensor:
    """5-D articulatory space + trajectory features.

    Coordinates:
      tongue_height ~ -F1   (low F1 = high tongue body)
      tongue_front  ~  F2   (high F2 = front tongue)
      lip_round     ~ -F3   (low F3 = lip rounding)
      velum_open    =  manner[nasal]
      glottal       =  hnr_peak (voicing)
    Then velocity + acceleration + "gestural peak" (sparse local max of speed).
    Returns (T, 5 + 5 + 5 + 1 + 5) = (T, 21).
    """
    pos = torch.cat([-fmt[:, 0:1], fmt[:, 1:2], -fmt[:, 2:3],
                      manner[:, 5:6], voicing], dim=1)  # (T,5)
    v = torch.zeros_like(pos); v[1:] = pos[1:] - pos[:-1]
    a = torch.zeros_like(v); a[1:] = v[1:] - v[:-1]
    # Trajectory speed
    speed = torch.sqrt((v ** 2).sum(dim=1, keepdim=True) + 1e-9)
    # Sparse local-max gesture peak in speed
    s_prev = torch.cat([speed[:1], speed[:-1]], dim=0)
    s_next = torch.cat([speed[1:], speed[-1:]], dim=0)
    is_max = ((speed > s_prev) & (speed > s_next) & (speed > 0)).float()
    sparse = is_max * speed
    return torch.cat([pos, v, a, speed, sparse * 1.0,
                       leaky_lowpass(sparse, 4.0),
                       leaky_lowpass(sparse, 12.0),
                       v * (voicing)], dim=1)


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
        self.long_freqs = torch.from_numpy(
            np.fft.rfftfreq(4096, d=1.0 / SR).astype(np.float32)).to(device)
        self.f0_cands = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32,
                                       device=device)
        # Phonetic 2-D template bank
        self.templates_short = make_phonetic_templates(self.N_FILT, n_time=5,
                                                         fcs=self.fcs).to(device)
        self.templates_mid = make_phonetic_templates(self.N_FILT, n_time=9,
                                                       fcs=self.fcs).to(device)
        self.templates_long = make_phonetic_templates(self.N_FILT, n_time=17,
                                                        fcs=self.fcs).to(device)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []

        # ====== Multi-window cochleagram backbone + adaptation deltas ======
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

        # ====== Loudness landmarks ======
        loud_long = log_e_long.mean(dim=1, keepdim=True)
        feats.append(loud_long)
        feats.append(peakrate_landmarks(loud_long, taus=(2.0, 5.0, 12.0)))

        # ====== Relative / speaker-normalized intonation ======
        feats.append(relative_intonation(log_f0, voicing,
                                            taus=(15.0, 60.0, 150.0, 400.0)))

        # ====== Contrastive formants (multi-tau) ======
        f_loc = fmt[:, :3]; f_amp = fmt[:, 3:6]; f_diff = fmt[:, 6:9]
        for tau in (15.0, 60.0, 150.0, 400.0):
            wsum = leaky_lowpass(voicing, tau)
            mu_f_loc = leaky_lowpass(f_loc * voicing, tau) / (wsum + 1e-3)
            mu_f_amp = leaky_lowpass(f_amp * voicing, tau) / (wsum + 1e-3)
            mu_f_diff = leaky_lowpass(f_diff * voicing, tau) / (wsum + 1e-3)
            feats.append((f_loc - mu_f_loc) * voicing)
            feats.append((f_amp - mu_f_amp) * voicing)
            feats.append((f_diff - mu_f_diff) * voicing)
        # Formant velocity & accel
        f_d1 = torch.zeros_like(f_loc); f_d1[1:] = f_loc[1:] - f_loc[:-1]
        f_d2 = torch.zeros_like(f_loc); f_d2[1:] = f_d1[1:] - f_d1[:-1]
        feats += [f_d1 * voicing, f_d2 * voicing,
                   f_d1.clamp(min=0), (-f_d1).clamp(min=0)]

        # ====== Phon-feature delta dynamics (lookback / lookahead) ======
        phon_pack = torch.cat([manner, place, fmt[:, :6]], dim=1)
        for k in (3, 6, 12):
            future = torch.cat([phon_pack[k:], phon_pack[-1:].expand(k, -1)], dim=0)
            past = torch.cat([phon_pack[:1].expand(k, -1), phon_pack[:-k]], dim=0)
            feats += [future - phon_pack, phon_pack - past]

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

        # ====== [v3 NEW] Phonetic 2-D template matched filters ======
        # Match the cochleagram against hand-designed phoneme templates at
        # three temporal scales. Each gives (T, P) phoneme-class similarities.
        tm_short = phonetic_template_match(log_e_long, self.templates_short)
        tm_mid = phonetic_template_match(log_e_long, self.templates_mid)
        tm_long = phonetic_template_match(log_e_long, self.templates_long)
        feats += [tm_short, tm_mid, tm_long]
        # Slow tonic of the template match (which phonemes were happening recently)
        feats += [leaky_lowpass(tm_mid, 15.0),
                   leaky_lowpass(tm_mid, 60.0)]
        # Future template match (upcoming phoneme context)
        feats.append(acausal_leaky_lowpass(tm_mid, 5.0))

        # ====== [v3 NEW] Articulatory gesture-space trajectory ======
        artic = articulatory_trajectory(fmt, hnr_long, manner, voicing)
        feats.append(artic)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "PHON-v3"
model_description = (
    "PHON v3 hybrid: cochleagram backbone (3 win x 5-tau adaptation) + "
    "phonological detectors + NEW phonetic 2-D template matched filters "
    "(12 prototypes x 3 time scales) + articulatory gesture-space trajectory. "
    "v1=0.0645, v2=0.0638."
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
