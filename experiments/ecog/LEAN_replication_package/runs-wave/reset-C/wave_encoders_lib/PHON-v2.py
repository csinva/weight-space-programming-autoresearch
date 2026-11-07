"""
PHONOLOGICAL / LINGUISTIC front-end for patient1 waveform -> ECoG high-gamma.

Paradigm: instead of bottom-up gammatone+modulations, build a top-down stream of
closed-form LINGUISTIC feature detectors on the hypothesis that STG encodes
phonological features and prosody directly.

Components (all closed-form, no learning):
  A. Articulatory MANNER detectors (vowel/sonorant/fricative/sibilant/stop/
     nasal/approximant/silence) from band-energy contrasts on log-power spectrum.
  B. PLACE of articulation: low/mid/high spectral subband contrasts (front/back
     vowel axis; labial/coronal/dorsal stop axis).
  C. VOICING: HNR + harmonic-comb salience + low-frequency-energy ratio.
  D. RELATIVE / speaker-normalized INTONATION: log-F0 minus & divided by a
     slow running baseline; pitch velocity, acceleration, jerk.
  E. SYLLABIC peakRate landmark train (impulses at envelope-derivative maxima).
  F. FORMANT tracking F1/F2/F3 from spectral-band local maxima.
  G. CONTRASTIVE vowel-space coordinates (F1/F2/F3 minus running mean).
  H. PHONOLOGICAL feature vector (binary distinctive features via thresholds).
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
# DSP building blocks (all closed-form)
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
    """Causal EMA along time."""
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def symm_lowpass(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric EMA = causal + acausal averaged (zero-phase)."""
    fwd = leaky_lowpass(x, tau)
    bwd = torch.flip(leaky_lowpass(torch.flip(x, [0]), tau), [0])
    return 0.5 * (fwd + bwd)


# ---------------------------------------------------------------------------
# Phonological / linguistic feature extractors
# ---------------------------------------------------------------------------

def harmonic_comb(power: torch.Tensor, win: int, f0_cands: torch.Tensor,
                   n_harm: int = 10) -> torch.Tensor:
    """Comb salience per F0 candidate. (T, n_F0)."""
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
    """Harmonics-to-noise ratio from normalized autocorrelation peak in pitch lag range.
    Returns (T, 3): [peak_val, hnr_db, best_lag_log]."""
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
    """log centroid, log spread, skew, kurt, flatness, rolloff85, rolloff95.
    Returns (T, 7)."""
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
# MANNER / PLACE / VOICING detectors (closed-form)
# ---------------------------------------------------------------------------

def manner_detectors(log_e: torch.Tensor, fcs: np.ndarray,
                      hnr: torch.Tensor, comb_max: torch.Tensor) -> torch.Tensor:
    """Closed-form articulatory manner detectors from log band-energies.

    Bands defined by Hz cutoffs over the ERB-spaced cochleagram channels.
    Returns (T, 9): [vowel, sonorant, fricative, sibilant, plosive, nasal,
                     approximant, silence, voiced].
    """
    def band(fl, fh):
        mask = (fcs >= fl) & (fcs <= fh)
        if not mask.any():
            return log_e[:, :1] * 0.0
        idx = torch.from_numpy(np.where(mask)[0]).long().to(log_e.device)
        return log_e[:, idx].mean(dim=1, keepdim=True)

    lo = band(80, 500)        # F1 region / nasal murmur
    mid = band(500, 2500)     # F2 / vowel core
    hi = band(2500, 5000)     # F3 / fricative onset
    very_hi = band(5000, 8000)  # sibilant
    full = log_e.mean(dim=1, keepdim=True)
    spread = log_e.std(dim=1, keepdim=True)

    peak_val = hnr[:, 0:1]
    hnr_db = hnr[:, 1:2]
    cmax = comb_max  # (T,1)

    # Manner detectors (sign-stable analytic combinations)
    voiced = peak_val * 2.0 + cmax  # voicing proxy
    silence = -full
    vowel = (lo + mid) * 0.5 + voiced - spread * 0.5
    sonorant = (lo + mid) * 0.5 + voiced
    fricative = (hi + very_hi) * 0.5 - lo - 0.5 * voiced
    sibilant = very_hi - mid - 0.5 * voiced
    plosive = spread + (very_hi * 0.3) - voiced * 0.5  # noisy onset, unvoiced
    nasal = lo - mid - hi * 0.5 + voiced * 0.3        # narrow low + voiced
    approximant = mid + voiced - spread * 0.3 - very_hi
    return torch.cat([vowel, sonorant, fricative, sibilant, plosive, nasal,
                       approximant, silence, voiced], dim=1)


def place_detectors(log_e: torch.Tensor, fcs: np.ndarray) -> torch.Tensor:
    """Place of articulation: front/back vowel axis (F2 frequency), high/low (F1).

    Returns (T, 6): [front_vowel, back_vowel, high_vowel, low_vowel,
                     coronal_stop, labial_stop]. Each is a contrast of
                     subbands picked to align with classical phonetic acoustics.
    """
    def band(fl, fh):
        mask = (fcs >= fl) & (fcs <= fh)
        if not mask.any():
            return log_e[:, :1] * 0.0
        idx = torch.from_numpy(np.where(mask)[0]).long().to(log_e.device)
        return log_e[:, idx].mean(dim=1, keepdim=True)

    f1_low = band(250, 450)     # high vowels: F1 low (/i/, /u/)
    f1_high = band(550, 900)    # low vowels: F1 high (/a/)
    f2_low = band(700, 1100)    # back vowels: F2 low (/u/, /o/)
    f2_high = band(1700, 2700)  # front vowels: F2 high (/i/, /e/)
    f3 = band(2400, 3500)       # rhotic / palatalization
    high_burst = band(3500, 6500)  # coronal burst
    low_burst = band(80, 500)      # labial burst

    front_vowel = f2_high - f2_low
    back_vowel = f2_low - f2_high
    high_vowel = f1_low - f1_high
    low_vowel = f1_high - f1_low
    coronal_stop = high_burst - low_burst
    labial_stop = low_burst - high_burst
    return torch.cat([front_vowel, back_vowel, high_vowel, low_vowel,
                       coronal_stop, labial_stop], dim=1)


def formant_tracking(log_e: torch.Tensor, fcs: np.ndarray) -> torch.Tensor:
    """F1/F2/F3 location and amplitude via soft-argmax in canonical formant bands.
    Returns (T, 9): [logF1, logF2, logF3, A1, A2, A3, F2-F1, F3-F2, F3-F1]."""
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


def per_band_peakrate(log_e: torch.Tensor, taus=(2.0, 5.0, 12.0),
                        smooth_taus=(2.0, 5.0)) -> torch.Tensor:
    """Per-band syllabic / phonetic LANDMARK CASCADE.

    For each cochlear band (40 channels) and each smoothing tau:
      - peakRate_pos: HWR(d/dt smoothed envelope)
      - peakRate_neg: HWR(-d/dt)
      - sparse local-max landmark of d/dt
      - sparse local-max smoothed
    This breaks the syllabic landmark concept apart by frequency band, so
    each ERB channel contributes its own discrete event train — capturing
    e.g. burst-in-high-band (=stop) vs plateau-in-low-band (=vowel).
    Returns (T, n_bands * len(taus) * 3 + n_bands * len(taus) * len(smooth_taus)).
    """
    out = []
    for tau in taus:
        smooth = leaky_lowpass(log_e, tau)
        d = torch.zeros_like(smooth)
        d[1:] = smooth[1:] - smooth[:-1]
        d_prev = torch.cat([d[:1], d[:-1]], dim=0)
        d_next = torch.cat([d[1:], d[-1:]], dim=0)
        is_max = ((d > d_prev) & (d > d_next) & (d > 0)).float()
        sparse = is_max * d
        out.append(d.clamp(min=0.0))    # per-band peakRate_pos (T, n_bands)
        out.append((-d).clamp(min=0.0))  # per-band peakRate_neg
        out.append(sparse)
        for stau in smooth_taus:
            out.append(leaky_lowpass(sparse, stau))
    return torch.cat(out, dim=1)


def distinctive_features(log_e: torch.Tensor, fcs: np.ndarray,
                          hnr: torch.Tensor, comb_max: torch.Tensor,
                          fmt: torch.Tensor) -> torch.Tensor:
    """16-dim closed-form distinctive feature vector + pairwise interactions.

    Each feature is a normalized contrast designed so its sign approximates
    the +/- of classical SPE/Jakobson distinctive features.
    Returns (T, 16 + 16*15/2) = (T, 136).
    """
    def band(fl, fh):
        mask = (fcs >= fl) & (fcs <= fh)
        if not mask.any():
            return log_e[:, :1] * 0.0
        idx = torch.from_numpy(np.where(mask)[0]).long().to(log_e.device)
        return log_e[:, idx].mean(dim=1, keepdim=True)

    lo = band(80, 500)
    mid = band(500, 1800)
    hi = band(1800, 3500)
    sib = band(3500, 8000)
    full = log_e.mean(dim=1, keepdim=True)
    spread = log_e.std(dim=1, keepdim=True)
    voiced = hnr[:, 0:1] * 2.0 + comb_max
    f1, f2, f3 = fmt[:, 0:1], fmt[:, 1:2], fmt[:, 2:3]
    f2_f1 = fmt[:, 6:7]  # log F2 - log F1 (back-front + low-high mixed)

    # Distinctive features (normalize to roughly zero-centered, unit-scale)
    cons = -voiced + spread           # consonantal: noisy, unvoiced bias
    son = voiced - spread             # sonorant: voiced, narrow spectrum
    cont = -spread * 0.5 + (hi - lo)  # continuant: sustained (negative of stop)
    voice = voiced
    nas = lo - mid - hi * 0.5 + voiced * 0.3
    strid = sib - lo                  # strident: sibilant high-frequency dominance
    lab = lo - sib                    # labial: low-frequency dominant
    cor = sib - lo                    # coronal: high-frequency dominant
    dors = mid - lo - hi              # dorsal: midband concentration
    high_vow = -f1                    # high vowel: low F1
    low_vow = f1                      # low vowel: high F1
    back_vow = -f2                    # back vowel: low F2
    front_vow = f2                    # front vowel: high F2
    round_vow = -f3                   # rounding: low F3
    tense = (f2 - f1)                 # tense: peripheral vowel space
    anterior = sib + lo - mid         # anterior place

    df = torch.cat([cons, son, cont, voice, nas, strid, lab, cor, dors,
                     high_vow, low_vow, back_vow, front_vow, round_vow,
                     tense, anterior], dim=1)  # (T, 16)

    # Pairwise interaction terms (df_i * df_j) for i < j → 16*15/2 = 120
    T, K = df.shape
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append((df[:, i:i + 1] * df[:, j:j + 1]))
    inter = torch.cat(pairs, dim=1)  # (T, 120)
    return torch.cat([df, inter], dim=1)


def peakrate_landmarks(loudness: torch.Tensor, taus=(2.0, 5.0, 12.0)) -> torch.Tensor:
    """Syllabic PeakRate landmark trains.

    For each smoothing tau:
      d = d/dt of smoothed loudness
      peakRate = HWR(d) (positive envelope-onset rate)
      peakRateNeg = HWR(-d) (offset rate)
      sparse_peak = 1 if d[t] > d[t-1] and d[t] > d[t+1], else 0
                    (local-max isolation of peakRate)
    Returns (T, 6 * len(taus))."""
    out = []
    for tau in taus:
        smooth = leaky_lowpass(loudness, tau)
        d = torch.zeros_like(smooth)
        d[1:] = smooth[1:] - smooth[:-1]
        pr_pos = d.clamp(min=0.0)
        pr_neg = (-d).clamp(min=0.0)
        # Local-maximum isolation
        T = d.shape[0]
        d_prev = torch.cat([d[:1], d[:-1]], dim=0)
        d_next = torch.cat([d[1:], d[-1:]], dim=0)
        is_max = ((d > d_prev) & (d > d_next) & (d > 0)).float()
        sparse_peak = is_max * d  # height-weighted spike
        # Multi-tau smoothing of the sparse train (so ridge can see it)
        spike_smooth1 = leaky_lowpass(sparse_peak, 1.5)
        spike_smooth2 = leaky_lowpass(sparse_peak, 4.0)
        spike_smooth3 = leaky_lowpass(sparse_peak, 12.0)
        out.append(torch.cat([pr_pos, pr_neg, sparse_peak,
                               spike_smooth1, spike_smooth2, spike_smooth3], dim=1))
    return torch.cat(out, dim=1)


def relative_intonation(log_f0: torch.Tensor, voiced_mask: torch.Tensor,
                         taus=(15.0, 60.0, 150.0, 400.0)) -> torch.Tensor:
    """Speaker-normalized intonation: log-F0 minus & divided by running baseline.

    Returns (T, 6 + 6 * len(taus)):
      [logF0, voicing_gate, dF0, ddF0, dddF0, dF0_signed,
       (log_f0 - mu_tau, (log_f0 - mu_tau)/sigma_tau, log_f0/mu_tau,
        mu_tau, sigma_tau, contour_slope_tau) for tau in taus]
    """
    out = []
    out.append(log_f0)
    out.append(voiced_mask)

    d1 = torch.zeros_like(log_f0)
    d1[1:] = log_f0[1:] - log_f0[:-1]
    d2 = torch.zeros_like(log_f0)
    d2[1:] = d1[1:] - d1[:-1]
    d3 = torch.zeros_like(log_f0)
    d3[1:] = d2[1:] - d2[:-1]
    # Mask derivatives by voicing
    d1 = d1 * voiced_mask
    d2 = d2 * voiced_mask
    d3 = d3 * voiced_mask
    out.append(d1)
    out.append(d2)
    out.append(d3)
    out.append(d1.clamp(min=0.0))  # rising-pitch signed

    # For baselines, use voiced-only mean via masked smoothing trick:
    # mu_tau = lowpass(f0 * voiced) / lowpass(voiced + eps)
    for tau in taus:
        wsum = leaky_lowpass(voiced_mask, tau)
        msum = leaky_lowpass(log_f0 * voiced_mask, tau)
        mu = msum / (wsum + 1e-3)
        # Spread via masked second moment
        sq = leaky_lowpass((log_f0 ** 2) * voiced_mask, tau)
        var = (sq / (wsum + 1e-3) - mu ** 2).clamp(min=1e-6)
        sigma = torch.sqrt(var)
        rel = (log_f0 - mu) * voiced_mask
        z = ((log_f0 - mu) / sigma) * voiced_mask
        ratio = (log_f0 / (mu + 1e-3)) * voiced_mask
        # Slope: mu - mu_shifted
        slope = torch.zeros_like(mu)
        slope[1:] = mu[1:] - mu[:-1]
        out.append(torch.cat([rel, z, ratio, mu, sigma, slope], dim=1))
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Phonological / linguistic front-end for patient1."""

    win = 4096
    n_params = 0
    # Two analysis windows for FFT: short (transients) + long (formants, F0)
    WIN_LIST = (1024, 4096)
    N_FILT = 40  # ERB-spaced cochleagram channels
    F0_CANDIDATES = (75.0, 90.0, 110.0, 135.0, 165.0, 200.0, 245.0,
                      295.0, 355.0, 425.0)

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
        self.short_freqs = torch.from_numpy(
            np.fft.rfftfreq(1024, d=1.0 / SR).astype(np.float32)).to(device)
        self.f0_cands = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32,
                                       device=device)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []

        # ====== Long-window analysis (formants, F0, manner, place) ======
        p_long = power_spectrum(x, 4096, self.device)
        log_e_long = torch.log(torch.clamp(p_long @ self.fbs[1], min=1e-10))

        # Pitch: harmonic-comb + ACF
        comb = harmonic_comb(p_long, 4096, self.f0_cands, n_harm=10)
        comb_max, comb_argmax = comb.max(dim=1, keepdim=True)
        comb_f0 = torch.log(self.f0_cands[comb_argmax.squeeze(1)].unsqueeze(1).clamp(min=20.0))
        hnr_long = hnr_from_acf(p_long, 4096, f_min=75.0, f_max=450.0)
        # Voicing gate (soft) from HNR peak
        voicing = torch.sigmoid((hnr_long[:, 0:1] - 0.4) * 8.0)
        # Combine comb_f0 and ACF log_f0 (ACF lag-derived) weighted by voicing
        log_f0_acf = hnr_long[:, 2:3]
        log_f0 = log_f0_acf  # use ACF lag-derived F0 as primary
        feats.append(comb)            # 10
        feats.append(comb_max)        # 1
        feats.append(hnr_long)        # 3
        feats.append(voicing)         # 1

        # Manner detectors
        manner = manner_detectors(log_e_long, self.fcs, hnr_long, comb_max)
        feats.append(manner)          # 9

        # Place detectors
        place = place_detectors(log_e_long, self.fcs)
        feats.append(place)           # 6

        # Formants
        fmt = formant_tracking(log_e_long, self.fcs)
        feats.append(fmt)             # 9

        # Spectral moments (extra phonological context)
        mom = spectral_moments(p_long, self.long_freqs)
        feats.append(mom)             # 7

        # ====== Short-window analysis (onset detection, plosives) ======
        start = (4096 - 1024) // 2
        xs = x[:, start:start + 1024]
        p_short = power_spectrum(xs, 1024, self.device)
        log_e_short = torch.log(torch.clamp(p_short @ self.fbs[0], min=1e-10))
        mom_short = spectral_moments(p_short, self.short_freqs)
        feats.append(mom_short)       # 7
        loud_short = log_e_short.mean(dim=1, keepdim=True)
        feats.append(loud_short)      # 1

        # ====== Loudness ======
        loud_long = log_e_long.mean(dim=1, keepdim=True)
        feats.append(loud_long)       # 1

        # ====== Syllabic PeakRate landmark train ======
        landmarks = peakrate_landmarks(loud_long, taus=(2.0, 5.0, 12.0))
        feats.append(landmarks)       # 18

        # ====== Relative / speaker-normalized intonation ======
        rel_int = relative_intonation(log_f0, voicing,
                                        taus=(15.0, 60.0, 150.0, 400.0))
        feats.append(rel_int)         # 6 + 6*4 = 30

        # ====== Contrastive formant trajectories ======
        # F1/F2/F3 minus running mean (multi-tau, voicing-gated)
        f_loc = fmt[:, :3]   # log F1, F2, F3
        f_amp = fmt[:, 3:6]
        f_diff = fmt[:, 6:9]
        for tau in (15.0, 60.0, 150.0, 400.0):
            wsum = leaky_lowpass(voicing, tau)
            mu_f_loc = leaky_lowpass(f_loc * voicing, tau) / (wsum + 1e-3)
            mu_f_amp = leaky_lowpass(f_amp * voicing, tau) / (wsum + 1e-3)
            mu_f_diff = leaky_lowpass(f_diff * voicing, tau) / (wsum + 1e-3)
            feats.append((f_loc - mu_f_loc) * voicing)
            feats.append((f_amp - mu_f_amp) * voicing)
            feats.append((f_diff - mu_f_diff) * voicing)

        # Formant velocity / acceleration (diphthong gestures)
        f_d1 = torch.zeros_like(f_loc)
        f_d1[1:] = f_loc[1:] - f_loc[:-1]
        f_d2 = torch.zeros_like(f_loc)
        f_d2[1:] = f_d1[1:] - f_d1[:-1]
        feats.append(f_d1 * voicing)
        feats.append(f_d2 * voicing)
        feats.append(f_d1.clamp(min=0))
        feats.append((-f_d1).clamp(min=0))

        # ====== Phonological feature deltas at multiple lags ======
        # Manner / place dynamics at lag 3, 6, 12 (delta features)
        phon_pack = torch.cat([manner, place, fmt[:, :6]], dim=1)
        for k in (3, 6, 12):
            future = torch.cat([phon_pack[k:], phon_pack[-1:].expand(k, -1)], dim=0)
            past = torch.cat([phon_pack[:1].expand(k, -1), phon_pack[:-k]], dim=0)
            feats.append(future - phon_pack)
            feats.append(phon_pack - past)

        # ====== Spectral flux (broad onset) ======
        d = torch.zeros_like(log_e_long)
        d[1:] = log_e_long[1:] - log_e_long[:-1]
        feats.append(torch.cat([d.clamp(min=0.0).sum(dim=1, keepdim=True),
                                 (-d).clamp(min=0.0).sum(dim=1, keepdim=True),
                                 d.abs().sum(dim=1, keepdim=True)], dim=1))

        # Final cochleagram (anchors ridge to bottom-up energy)
        feats.append(log_e_long)      # 40
        feats.append(log_e_short)     # 40

        # ====== Slow tonic state of phonological features ======
        for tau in (15.0, 60.0):
            feats.append(leaky_lowpass(manner, tau))
            feats.append(leaky_lowpass(place, tau))
            feats.append(leaky_lowpass(fmt[:, :6], tau))

        # ====== [v2] Per-band peakRate landmark cascade ======
        # For each of 40 cochlear channels, compute envelope-derivative landmarks
        # at multiple time scales. Captures band-specific phonetic events (e.g.
        # stop burst in high band, vowel hold in low-mid band).
        per_band_pr = per_band_peakrate(log_e_long, taus=(2.0, 5.0, 12.0),
                                          smooth_taus=(2.0, 5.0))
        feats.append(per_band_pr)

        # ====== [v2] Distinctive feature vector + pairwise interactions ======
        df = distinctive_features(log_e_long, self.fcs, hnr_long, comb_max, fmt)
        feats.append(df)
        # Slow tonic of distinctive features
        for tau in (15.0, 60.0):
            feats.append(leaky_lowpass(df[:, :16], tau))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "PHON-v2"
model_description = (
    "PHON v2 (phon paradigm): adds per-band peakRate landmark cascade "
    "(40 bands x 3 taus x 5 quantities = 600 dim) + distinctive feature "
    "vector (16) with pairwise interactions (120) + slow tonic of DF. "
    "PHON v1 was 0.0645 / 1608 dim."
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
