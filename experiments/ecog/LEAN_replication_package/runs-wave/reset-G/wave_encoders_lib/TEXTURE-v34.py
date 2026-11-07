"""
McDermott-Simoncelli SOUND-TEXTURE summary statistics, computed as
CAUSAL RUNNING quantities at 30 Hz via leaky lowpass accumulators.

Paradigm: every statistic is a NONLINEAR functional of the cochlear envelope
(higher moments, cross-band products, modulation-band powers, cross-channel
modulation correlations). A linear ridge over the bands cannot reconstruct
any of these, so they add genuinely new information to the readout.
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_G"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cochlear filterbank
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=32, f_min=50.0, f_max=8000.0, order=4):
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
# Causal leaky lowpass — the only smoothing primitive
# ---------------------------------------------------------------------------

def leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Causal exponential moving average of (T, C) at timescale `tau_frames`."""
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Running marginal moments  (var, skew, kurt are NONLINEAR in x)
# ---------------------------------------------------------------------------

def running_moments(x: torch.Tensor, tau: float):
    """Returns (mu, var, skew, kurt-3) of (T, C) at causal timescale tau.
    All four are computed via EMA-of-x-and-its-powers."""
    mu = leaky_lowpass(x, tau)
    d = x - mu
    var = leaky_lowpass(d * d, tau).clamp(min=1e-10)
    std = torch.sqrt(var)
    m3 = leaky_lowpass(d ** 3, tau)
    m4 = leaky_lowpass(d ** 4, tau)
    skew = m3 / (std ** 3 + 1e-10)
    kurt = m4 / (var ** 2 + 1e-10) - 3.0
    return mu, var, skew, kurt


def running_corr(x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Running normalized cross-correlation rho_xy(t) of (T, C_pairs)."""
    mux = leaky_lowpass(x, tau)
    muy = leaky_lowpass(y, tau)
    dx = x - mux
    dy = y - muy
    cov = leaky_lowpass(dx * dy, tau)
    vx = leaky_lowpass(dx * dx, tau).clamp(min=1e-10)
    vy = leaky_lowpass(dy * dy, tau).clamp(min=1e-10)
    return cov / torch.sqrt(vx * vy)


# ---------------------------------------------------------------------------
# Temporal modulation Gabor filterbank
# ---------------------------------------------------------------------------

def make_temporal_gabor(rate_hz: float, fs: float, device, dtype):
    period = fs / max(rate_hz, 0.1)
    sigma = period
    L = int(np.ceil(4 * sigma))
    t = np.arange(-L, L + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
    sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
    cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
    n = np.linalg.norm(cos_k) + 1e-12
    cos_k /= n; sin_k /= n
    return (torch.from_numpy(cos_k.astype(np.float32)).to(device),
            torch.from_numpy(sin_k.astype(np.float32)).to(device))


def conv_temporal(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve (T, C) along time with a 1-D kernel; same length out."""
    L = kernel.numel()
    pad = L // 2
    C = x.shape[1]
    xp = x.transpose(0, 1).unsqueeze(0)
    kk = kernel.view(1, 1, -1).expand(C, 1, L).contiguous()
    xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    y = torch.nn.functional.conv1d(xp_p, kk, groups=C)
    return y[..., :x.shape[0]].squeeze(0).transpose(0, 1).contiguous()


def modulation_envelope(x: torch.Tensor, rate_hz: float, fs: float = 30.0):
    """Bandpass (T, C) at temporal rate `rate_hz` using a complex Gabor; return
    (cos_part, sin_part, envelope=|complex|)."""
    cos_k, sin_k = make_temporal_gabor(rate_hz, fs, x.device, x.dtype)
    c = conv_temporal(x, cos_k)
    s = conv_temporal(x, sin_k)
    env = torch.sqrt(c * c + s * s + 1e-12)
    return c, s, env


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """McDermott-Simoncelli sound-texture statistics, causally accumulated at 30 Hz.

    For each frame at 30 Hz we expose:

      1. raw compressed log-cochleagram c_b(t)        ............... LINEAR
      2. per-band running moments  (var, skew, kurt)
         at multiple timescales tau ........................ NONLINEAR x_b^k
      3. cross-band normalized envelope correlations
         (selected near + long-range pairs)
         at multiple timescales ........................... NONLINEAR x_a * x_b / ...
      4. modulation-band power per (band, mod_rate)
         (mean, variance) at multiple timescales ........... NONLINEAR
      5. cross-channel correlations at matched mod rate (C1)
         (selected pairs) ................................. NONLINEAR

    Every smoothing operator is a causal leaky EMA — no re-windowing.
    """

    win = 4096
    n_params = 0

    # Cochlear bands
    N_BANDS = 24
    COCHLEAR_COMPRESSION = 0.3

    # Running-moment timescales (frames @ 30 Hz)
    MOMENT_TAUS = (4.5, 15.0, 60.0)                # ~150 ms / 500 ms / 2 s

    # Modulation rates  (Hz) — McDermott uses 1/2, 1, 2, 4, 8, 16, 32 Hz
    # v34: dropped 16Hz to free dim budget.
    MOD_RATES = (2.0, 4.0, 8.0)
    # SLOW modulation rates for the loudness/centroid signals (syllable tempo)
    SLOW_RATES = (0.5, 1.0)

    # Cross-band pair offsets (in band index). Captures local + long-range comodulation.
    PAIR_OFFSETS = (1, 2, 4, 8)

    # 2-D STMF rate-scale (Chi-Shamma) — (temporal Hz, spectral cyc/channel).
    # Positive scale = upward sweep; negative = downward sweep; 0 = pure temporal.
    # v34: restored full set (trimming in v33 hurt 5pt).
    STMF_RS = (
        (1.0, 0.0),  (2.0, 0.0),  (4.0, 0.0),  (8.0, 0.0),  (12.0, 0.0),
        (2.0, 0.125), (2.0, -0.125),
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.125), (4.0, -0.125),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (0.0, 0.125), (0.0, 0.25), (0.0, 0.5),
    )

    SHORT_WIN = 1024     # second cochleagram window (~23 ms) — finer temporal grain
    SHORT_BANDS = 20

    # Within-window envelope-modulation-spectrum sub-bands (Hz). Per band
    # we sum |Env(f)|^2 over each interval — captures syllable/pitch/voicing
    # content of the WITHIN-WINDOW envelope at multiple temporal scales.
    ENV_MOD_BANDS = (
        (4.0, 12.0),     # theta / syllable rate
        (12.0, 32.0),    # beta / fast-transition rate
        (32.0, 100.0),   # gamma / phonation onset rate
        (100.0, 300.0),  # F0 fundamental range
        (300.0, 800.0),  # high-pitched voicing / unresolved harmonics
    )

    # Within-window cross-band envelope correlation offsets.
    CROSS_BAND_OFFSETS = (1, 4, 8)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.N_BANDS).to(device)
        self.fb_short = gammatone_powerbank(self.SHORT_WIN,
                                             n_filters=self.SHORT_BANDS).to(device)
        # AMPLITUDE-domain gammatone (sqrt of the power bank) — used for the
        # analytic per-band signal (rfft -> H -> ifft) in the within-window
        # envelope-statistics block. NOT the same as power bank.
        self.fb_amp = torch.sqrt(self.fb.clamp(min=0.0)).to(device)
        # Pre-cached hann window for FFT
        self._hann = torch.hann_window(self.win, periodic=True, device=device)
        # Precompute frequency axis and weighted-frequency banks for per-band
        # spectral-centroid / bandwidth (instantaneous-frequency proxies)
        freqs_np = np.fft.rfftfreq(self.win, d=1.0 / SR).astype(np.float32)
        self.freq_axis = torch.from_numpy(freqs_np).to(device)            # (n_freqs,)
        log_f = np.log(np.clip(freqs_np, 20.0, None)).astype(np.float32)
        self.log_freq_axis = torch.from_numpy(log_f).to(device)
        self.freq_H = self.freq_axis.unsqueeze(1) * self.fb               # (n_freqs, F)
        self.freq2_H = (self.freq_axis ** 2).unsqueeze(1) * self.fb       # (n_freqs, F)
        self.logfreq_H = self.log_freq_axis.unsqueeze(1) * self.fb        # (n_freqs, F)
        # Precompute modulation Gabor kernels (fast + slow)
        self._mod_kernels = {}
        for r in self.MOD_RATES:
            self._mod_kernels[r] = make_temporal_gabor(r, 30.0, device, torch.float32)
        for r in self.SLOW_RATES:
            self._mod_kernels[r] = make_temporal_gabor(r, 30.0, device, torch.float32)
        self._stmf_cache: dict[tuple, tuple] = {}

    def _cochleagram(self, windows_t: torch.Tensor):
        """Compute (T, F) compressed envelope, (T, n_freqs) power spectrum,
        and (T, n_freqs) complex rfft spectrum from per-frame windows."""
        X = torch.fft.rfft(windows_t * self._hann, n=self.win)       # (T, n_freqs) complex
        p = (X.real ** 2 + X.imag ** 2)                              # (T, n_freqs)
        e = p @ self.fb                                              # (T, F)
        c = (e + 1e-10) ** self.COCHLEAR_COMPRESSION
        return c, p, X

    def _within_window_envelope_stats(self, X: torch.Tensor):
        """One-pass within-window per-band analytic stats: envelope moments,
        asymmetry, pitch ACF, env-mod sub-band powers, and CROSS-BAND envelope
        coherence at selected offsets (pitch-rate coherence — voiced/harmonic
        coupling across bands sharing F0)."""
        T = X.shape[0]
        n_freqs = X.shape[1]
        N = self.win
        F = self.N_BANDS
        Hamp = self.fb_amp                                           # (n_freqs, F)
        # Build all-bands analytic spectrum at once
        X_b = X.unsqueeze(1) * Hamp.t().unsqueeze(0)                 # (T, F, n_freqs)
        Y = torch.zeros((T, F, N), dtype=torch.complex64, device=X.device)
        Y[:, :, :n_freqs] = X_b
        Y[:, :, 1:n_freqs - 1] *= 2.0
        del X_b
        z = torch.fft.ifft(Y, n=N, dim=2)                            # (T, F, N) complex
        del Y
        env = z.abs()                                                 # (T, F, N) real
        # Unit phasor zn = z / |z|  — only PHASE info (carrier phase per sample)
        zn = z / env.clamp(min=1e-9)                                  # (T, F, N) complex
        del z
        # Envelope moments
        m1 = env.mean(dim=2)                                          # (T, F)
        d = env - m1.unsqueeze(2)
        m2 = (d * d).mean(dim=2).clamp(min=1e-12)
        m3 = (d * d * d).mean(dim=2)
        m4 = (d * d * d * d).mean(dim=2)
        std = torch.sqrt(m2)
        cov = std / m1.clamp(min=1e-6)
        skew = (m3 / (std * m2 + 1e-12)).clamp(-10.0, 10.0)
        kurt = (m4 / (m2 * m2 + 1e-12) - 3.0).clamp(-5.0, 50.0)
        del d, m3, m4
        # Asymmetry: first half CoV vs second half CoV
        h = N // 2
        env_h1 = env[..., :h]
        env_h2 = env[..., h:]
        cov_h1 = env_h1.std(dim=2) / env_h1.mean(dim=2).clamp(min=1e-6)
        cov_h2 = env_h2.std(dim=2) / env_h2.mean(dim=2).clamp(min=1e-6)
        asym = cov_h1 - cov_h2
        del env_h1, env_h2, cov_h1, cov_h2
        # Cross-band PHASE-LOCK VALUE (PLV) — true harmonic phase coupling.
        # For each selected offset off: zn_a · conj(zn_b) is a unit phasor; its
        # mean over n gives the complex coherence; |mean|=PLV in [0,1].  Voiced
        # harmonic bands lock phase (PLV~1); noise yields PLV~0.
        plv_mag_l, plv_re_l, plv_im_l = [], [], []
        for offset in self.CROSS_BAND_OFFSETS:
            za = zn[:, :F - offset, :]
            zb = zn[:, offset:, :]
            crossp = za * zb.conj()                                   # (T, F-off, N)
            mean_p = crossp.mean(dim=2)                               # (T, F-off) complex
            plv_mag_l.append(mean_p.abs())
            plv_re_l.append(mean_p.real)
            plv_im_l.append(mean_p.imag)
        plv_mag = torch.cat(plv_mag_l, dim=1)
        plv_re = torch.cat(plv_re_l, dim=1)
        plv_im = torch.cat(plv_im_l, dim=1)
        del zn
        # Envelope spectrum → mod-band powers + pitch ACF
        Env = torch.fft.rfft(env, n=N, dim=2)                         # (T, F, n_freqs)
        del env
        Penv = (Env.real * Env.real + Env.imag * Env.imag)
        del Env
        df_env = SR / N
        mod_band_outs = []
        for (lo_hz, hi_hz) in self.ENV_MOD_BANDS:
            lo = max(1, int(np.floor(lo_hz / df_env)))
            hi = min(Penv.shape[2], max(lo + 1, int(np.ceil(hi_hz / df_env))))
            pwr = Penv[..., lo:hi].sum(dim=2).clamp(min=1e-10)
            mod_band_outs.append(torch.log(pwr))
        ACF = torch.fft.irfft(Penv, n=N, dim=2)                       # (T, F, N)
        del Penv
        ACF_n = ACF / ACF[..., :1].clamp(min=1e-12)
        lag_min = max(2, int(SR / 500.0))
        lag_max = min(N - 1, int(SR / 75.0))
        region = ACF_n[..., lag_min:lag_max + 1]
        peak_val, peak_idx = region.max(dim=2)
        peak_val = peak_val.clamp(min=0.0, max=0.999)
        lag_actual = peak_idx.float() + lag_min
        f0 = SR / lag_actual.clamp(min=1.0)
        log_f0 = torch.log(f0.clamp(min=20.0))
        hnr = 10.0 * torch.log10(
            peak_val.clamp(min=1e-3) / (1.0 - peak_val).clamp(min=1e-3))
        out = {
            'log_mean': torch.log(m1.clamp(min=1e-6)),
            'cov': cov,
            'skew': skew,
            'kurt': kurt,
            'asym': asym,
            'pitch_log_f0': log_f0,
            'pitch_hnr': hnr,
            'plv_mag': plv_mag,
            'plv_re': plv_re,
            'plv_im': plv_im,
        }
        for i, _b in enumerate(self.ENV_MOD_BANDS):
            out[f'mod{i}'] = mod_band_outs[i]
        return out

    F0_CANDIDATES = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0, 230.0, 270.0,
                      320.0, 380.0, 450.0)

    def _pitch_block(self, p: torch.Tensor) -> list[torch.Tensor]:
        """Closed-form harmonic-comb pitch salience + autocorrelation HNR
        derived from the per-window power spectrum (no re-windowing).

        Harmonic salience: for each F0 candidate, sum log-power at the first n
        harmonic bins minus the global log-power baseline. The argmax over
        candidates gives the running pitch estimate.

        HNR: normalized autocorrelation from IFFT of power spectrum;
        peak/(1-peak) is the harmonic-to-noise ratio."""
        feats = []
        n_freqs = p.shape[1]
        freq_res = SR / self.win
        log_p = torch.log(torch.clamp(p, min=1e-10))
        baseline = log_p.mean(dim=1, keepdim=True)
        # Harmonic comb for each F0 candidate
        cand_feats = []
        f0_t = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32, device=p.device)
        for i, f0 in enumerate(self.F0_CANDIDATES):
            ks = np.arange(1, 9)
            bins = np.clip((f0 * ks / freq_res).astype(int), 0, n_freqs - 1)
            bins_t = torch.from_numpy(bins).to(p.device)
            sal = log_p[:, bins_t].mean(dim=1, keepdim=True) - baseline
            cand_feats.append(sal)
        salience = torch.cat(cand_feats, dim=1)                        # (T, F0)
        feats.append(salience)
        # Best F0 and its strength + log of pitch
        best_strength, best_idx = salience.max(dim=1, keepdim=True)
        log_f0 = torch.log(f0_t[best_idx.squeeze(1)].unsqueeze(1).clamp(min=20.0))
        feats.append(best_strength)
        feats.append(log_f0)
        # Multi-tau dynamics (highpass) of pitch + strength
        for tau in (4.5, 15.0, 60.0):
            feats.append(log_f0 - leaky_lowpass(log_f0, tau))
            feats.append(best_strength - leaky_lowpass(best_strength, tau))

        # Autocorrelation HNR
        full = torch.fft.irfft(p, n=self.win)                          # (T, win)
        acf0 = full[:, :1] + 1e-12
        acf_n = full / acf0
        lag_min = max(1, int(SR / 400.0))
        lag_max = min(self.win - 1, int(SR / 75.0))
        region = acf_n[:, lag_min:lag_max + 1]
        peak_val, peak_lag = region.max(dim=1, keepdim=True)
        peak_val = peak_val.clamp(min=0.0, max=0.999)
        hnr_db = 10.0 * torch.log10(peak_val.clamp(min=1e-3) /
                                      (1.0 - peak_val).clamp(min=1e-3))
        feats.append(peak_val)
        feats.append(hnr_db)
        feats.append(torch.log(peak_lag.float().clamp(min=1.0) + 0.0))
        # HNR derivative (voicing onset/offset)
        hd = torch.zeros_like(hnr_db)
        hd[1:] = hnr_db[1:] - hnr_db[:-1]
        feats.append(hd)
        feats.append(hd.clamp(min=0))
        feats.append((-hd).clamp(min=0))
        # Multi-tau dynamics of HNR
        for tau in (4.5, 15.0, 60.0):
            feats.append(hnr_db - leaky_lowpass(hnr_db, tau))
        return feats

    def _band_moments_block(self, c: torch.Tensor) -> list[torch.Tensor]:
        """For each tau, return [var, skew, kurt] — each (T, F)."""
        out = []
        for tau in self.MOMENT_TAUS:
            mu, var, skew, kurt = running_moments(c, tau)
            out.append(torch.log(var.clamp(min=1e-10)))
            out.append(skew.clamp(-10.0, 10.0))
            out.append(kurt.clamp(-5.0, 50.0))
            # also the running mu itself (still useful, although near-linear)
            out.append(mu)
        return out

    def _cross_band_correlations(self, c: torch.Tensor) -> list[torch.Tensor]:
        """Running normalized correlation between band pairs at multiple lags
        and multiple timescales."""
        T, F = c.shape
        feats = []
        for offset in self.PAIR_OFFSETS:
            a = c[:, :F - offset]
            b = c[:, offset:]
            for tau in self.MOMENT_TAUS:
                rho = running_corr(a, b, tau)
                feats.append(rho)
        return feats

    def _modulation_block(self, c: torch.Tensor) -> list[torch.Tensor]:
        """For each band & each modulation rate:
              env_{b,r}(t) = |Gabor_r * c_b|
           Then compress and exponentially-smooth.
           ALSO compute running variance of env_{b,r} (modulation 'depth' nonlinearity).
           v34: trimmed to ONE tau per rate (60.0) for memory.
        """
        feats = []
        envs_by_rate = {}                          # store for cross-channel C1
        for r in self.MOD_RATES:
            cos_k, sin_k = self._mod_kernels[r]
            cc = conv_temporal(c, cos_k)
            ss = conv_temporal(c, sin_k)
            env = torch.sqrt(cc * cc + ss * ss + 1e-12)       # (T, F)
            log_env = torch.log(env.clamp(min=1e-8))
            envs_by_rate[r] = env
            feats.append(log_env)                              # all bands' mod power
            # single tau; nonlinear running mean + variance + CoV
            tau = 60.0
            mu = leaky_lowpass(env, tau)
            var = leaky_lowpass((env - mu) ** 2, tau).clamp(min=1e-10)
            feats.append(torch.log(mu.clamp(min=1e-8)))
            feats.append(torch.log(var))
            feats.append(torch.log((var / (mu * mu + 1e-10) + 1e-8)))   # mod CoV
        return feats, envs_by_rate

    def _cross_channel_modulation_corr(
            self, envs_by_rate: dict[float, torch.Tensor]) -> list[torch.Tensor]:
        """C1: cross-channel running correlation of modulation envelopes at the
        SAME modulation rate. Selected band offsets. Single tau (lean)."""
        feats = []
        for r, env in envs_by_rate.items():
            F = env.shape[1]
            for offset in (2, 4, 8):
                a = env[:, :F - offset]
                b = env[:, offset:]
                rho = running_corr(a, b, 60.0)
                feats.append(rho)
        return feats

    def _stmf_envelope(self, log_e: torch.Tensor, omega_t: float, omega_f: float,
                        fs: float = 30.0) -> torch.Tensor:
        """Closed-form Chi-Shamma 2-D Gabor envelope on (T, F) cochleagram.
        Returns (T, F). Stride-1 conv with cached kernels."""
        key = (round(omega_t, 3), round(omega_f, 3))
        if key not in self._stmf_cache:
            period_t = fs / max(abs(omega_t), 0.5)
            sigma_t = period_t / (2 * np.pi)
            Lt = max(1, int(np.ceil(3 * sigma_t)))
            sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi)
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
        xp = log_e.unsqueeze(0).unsqueeze(0)            # (1,1,T,F)
        c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
        s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
        T, F = log_e.shape
        env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
        return env[:T, :F]

    def _stmf_block(self, log_c: torch.Tensor) -> list[torch.Tensor]:
        """For each (omega_t, omega_f), compute the 2-D Gabor envelope —
        full (T, F) feature map per filter."""
        feats = []
        for (ot, of) in self.STMF_RS:
            feats.append(self._stmf_envelope(log_c, ot, of))
        return feats

    def _within_channel_cross_rate_corr(
            self, envs_by_rate: dict[float, torch.Tensor]) -> list[torch.Tensor]:
        """C2: within-channel correlation between modulation envelopes at
        DIFFERENT modulation rates (amplitude-amplitude coupling across rates)."""
        feats = []
        rates = list(envs_by_rate.keys())
        # Pair rate i with rate i+1 (adjacent octave)
        for i in range(len(rates) - 1):
            ea = envs_by_rate[rates[i]]
            eb = envs_by_rate[rates[i + 1]]
            for tau in (30.0, 150.0):
                rho = running_corr(ea, eb, tau)
                feats.append(rho)
        return feats

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        c, p, X_spec = self._cochleagram(x)                     # (T, F), (T, n_freqs), complex
        log_c = torch.log(c.clamp(min=1e-10))

        feats: list[torch.Tensor] = []
        # 1. raw compressed envelope (linear input the ridge can use directly)
        feats.append(log_c)
        # First-difference of log envelope (delta) — basic onset signal
        d1 = torch.zeros_like(log_c)
        d1[1:] = log_c[1:] - log_c[:-1]
        feats.append(d1)

        # 2. per-band running moments  (NONLINEAR x^k)
        feats.extend(self._band_moments_block(log_c))

        # 3. cross-band envelope correlations  (NONLINEAR products / ratios)
        feats.extend(self._cross_band_correlations(log_c))

        # 3b. NONLINEAR MATCHED-FILTER pattern detectors (L1) +
        #     CASCADED RANDOM-PROJECTION ReLU layer (L2).
        #     L1: 24 closed-form Gaussian templates over (band-center, band-width)
        #     grid → mf, with ReLU + softmax + EMA outputs (closed-form, NONLINEAR).
        #     L2: fixed-seed random projection of L1 ReLU codes through K2=40
        #     random Gaussian directions + bias → ReLU. This is a true 2-stage
        #     nonlinear feature stack — the linear ridge cannot synthesize it.
        F = log_c.shape[1]
        if not hasattr(self, "_mf_templates"):
            centers = np.linspace(2.0, F - 3.0, 8).astype(np.float32)
            widths = (1.5, 3.0, 5.0)
            tmps = []
            b_axis = np.arange(F, dtype=np.float32)
            for s in widths:
                for c0 in centers:
                    t = np.exp(-((b_axis - c0) ** 2) / (2.0 * s * s))
                    t -= t.mean()
                    n = np.linalg.norm(t) + 1e-9
                    t /= n
                    tmps.append(t)
            T_mat = np.stack(tmps).astype(np.float32)              # (K1, F)
            self._mf_templates = torch.from_numpy(T_mat).to(log_c.device).t()
            K1 = T_mat.shape[0]
            # Fixed-seed random projection for L2 (K1 -> K2)
            K2 = 40
            rng = np.random.RandomState(20260605)
            W2 = rng.randn(K1, K2).astype(np.float32) / np.sqrt(K1)
            b2 = rng.randn(K2).astype(np.float32) * 0.3
            self._mf_W2 = torch.from_numpy(W2).to(log_c.device)
            self._mf_b2 = torch.from_numpy(b2).to(log_c.device)
            # second random projection (for softmax-input L2)
            W2b = rng.randn(K1, K2).astype(np.float32) / np.sqrt(K1)
            b2b = rng.randn(K2).astype(np.float32) * 0.3
            self._mf_W2b = torch.from_numpy(W2b).to(log_c.device)
            self._mf_b2b = torch.from_numpy(b2b).to(log_c.device)
        mf = log_c @ self._mf_templates                            # (T, K1)
        mf_bias = leaky_lowpass(mf, 60.0)
        mf_dev = mf - mf_bias                                       # zero-mean
        mf_pos = mf_dev.clamp(min=0.0)                              # NONLINEAR
        mf_neg = (-mf_dev).clamp(min=0.0)                           # NONLINEAR
        mf_sm = torch.softmax(mf_dev * 1.5, dim=1)                  # WTA-soft
        feats.append(mf_pos)
        feats.append(mf_neg)
        feats.append(mf_sm)
        feats.append(leaky_lowpass(mf_pos, 30.0))
        feats.append(leaky_lowpass(mf_sm, 30.0))
        # (L2 cascade dropped — added noise in v22.)

        # 4. modulation filterbank + per-(band, rate) statistics
        mod_feats, envs_by_rate = self._modulation_block(log_c)
        feats.extend(mod_feats)

        # 5. C1 cross-channel modulation correlations
        feats.extend(self._cross_channel_modulation_corr(envs_by_rate))

        # (6. C2 within-channel cross-rate corr dropped — marginal contribution
        # per v15 ablation; freeing ~144 dim for within-window pitch block.)

        # 7. 2-D STMF spectro-temporal modulation envelopes (rate-scale Gabors).
        #    Both on log_c (multiplicative) and on raw compressed c (additive).
        stmf_log = self._stmf_block(log_c)
        feats.extend(stmf_log)
        # 7b. NONLINEAR STMF POOLING — "which spectro-temporal pattern dominates
        #     at each frame, and at what band location?"  Soft argmax across the
        #     band axis (per filter), softmax across filters (cross-filter WTA),
        #     and ReLU half-wave of band-max.
        stmf_stack = torch.stack(stmf_log, dim=1)                 # (T, K, F)
        T_st, K_st, F_st = stmf_stack.shape
        # Per-filter peak-position soft-argmax + sharpness (within-band)
        sm_band = torch.softmax(stmf_stack * 2.0, dim=2)          # (T, K, F)
        b_axis_t = torch.arange(F_st, device=stmf_stack.device,
                                 dtype=stmf_stack.dtype).view(1, 1, -1)
        peak_pos = (sm_band * b_axis_t).sum(dim=2)                # (T, K) soft argmax band
        peak_spr = torch.sqrt(((sm_band * (b_axis_t - peak_pos.unsqueeze(2)) ** 2)
                                .sum(dim=2)).clamp(min=1e-6))      # spread
        # Band-max value per filter (soft max via logsumexp)
        band_max = torch.logsumexp(stmf_stack * 2.0, dim=2) / 2.0  # (T, K)
        band_max_relu = band_max.clamp(min=0.0)                    # NONLINEAR
        # Softmax across filters (which filter wins?)
        sm_filter = torch.softmax(band_max * 2.0, dim=1)           # (T, K)
        feats.append(peak_pos)
        feats.append(peak_spr)
        feats.append(band_max_relu)
        feats.append(sm_filter)
        feats.append(leaky_lowpass(sm_filter, 30.0))
        feats.append(leaky_lowpass(band_max_relu, 30.0))

        # 7c. NONLINEAR POOLING of MODULATION ENVELOPES — "where in the spectrum
        #     is the X-Hz modulation peaking?"  Per modulation rate r, compute
        #     soft-argmax peak band and band-max; softmax across rates picks
        #     the dominant modulation rate at each frame.
        rate_band_max = []
        for r in self.MOD_RATES:
            env_r = envs_by_rate[r]                                 # (T, F)
            sm_b = torch.softmax(env_r * 2.0, dim=1)                # (T, F)
            bax = torch.arange(env_r.shape[1], device=env_r.device,
                                dtype=env_r.dtype).view(1, -1)
            pp = (sm_b * bax).sum(dim=1, keepdim=True)
            ps = torch.sqrt(((sm_b * (bax - pp) ** 2).sum(dim=1, keepdim=True))
                              .clamp(min=1e-6))
            bm = torch.logsumexp(env_r * 2.0, dim=1, keepdim=True) / 2.0
            feats.append(pp)
            feats.append(ps)
            feats.append(bm)
            rate_band_max.append(bm)
        rate_stack = torch.cat(rate_band_max, dim=1)                # (T, R)
        rate_winner = torch.softmax(rate_stack * 2.0, dim=1)        # (T, R)
        feats.append(rate_winner)
        feats.append(leaky_lowpass(rate_winner, 30.0))

        # 7d. NONLINEAR POOLING of LATERAL-INHIBITED cochleagram (lat).
        #     Computed below (block 9b2); placeholder here. Inserted post-hoc.

        # 8. Pitch / HNR autocorrelation block — strongly nonlinear ratios of
        #    ACF peaks and harmonic-comb sums of log power.
        feats.extend(self._pitch_block(p))

        # 9. ON/OFF rectified AGC: half-wave-rectified (log_c - EMA_mu)/EMA_std,
        #    pos and neg channels at one tau. Compact, biophysically motivated.
        mu_t = leaky_lowpass(log_c, 15.0)
        var_t = leaky_lowpass((log_c - mu_t) ** 2, 15.0).clamp(min=1e-10)
        z = (log_c - mu_t) / torch.sqrt(var_t)
        feats.append(z.clamp(min=0.0))
        feats.append((-z).clamp(min=0.0))

        # 9b. SLOW MODULATION (1 Hz) on per-band log envelope — syllable tempo.
        cos_k, sin_k = self._mod_kernels[1.0]
        cc = conv_temporal(log_c, cos_k)
        ss = conv_temporal(log_c, sin_k)
        env = torch.sqrt(cc * cc + ss * ss + 1e-12)
        feats.append(torch.log(env.clamp(min=1e-8)))

        # 9b2. LATERAL INHIBITION across bands (DoG conv) on log_c.
        #      lat[b] = narrow_avg(log_c)[b] - wide_avg(log_c)[b]
        #      Sharpens spectral peaks. Closed-form Gaussian kernels along band axis.
        #      Also include the squared & ON/OFF rectified versions.
        def _gauss_band_conv(x: torch.Tensor, sigma_b: float) -> torch.Tensor:
            k = int(np.ceil(3 * sigma_b))
            coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
            g = torch.exp(-(coords ** 2) / (2 * sigma_b ** 2))
            g = (g / g.sum()).view(1, 1, -1)
            return torch.nn.functional.conv1d(
                x.unsqueeze(1), g, padding=k).squeeze(1)
        narrow = _gauss_band_conv(log_c, 0.8)
        wide = _gauss_band_conv(log_c, 3.0)
        lat = narrow - wide                                          # (T, F)
        feats.append(lat)
        feats.append(lat.clamp(min=0.0))                             # peaks only
        # Temporal delta of lat — onset of spectral peaks
        lat_d = torch.zeros_like(lat)
        lat_d[1:] = lat[1:] - lat[:-1]
        feats.append(lat_d.clamp(min=0.0))
        feats.append((-lat_d).clamp(min=0.0))
        # 9b2c. NONLINEAR POOLING of lat — soft-argmax band (sharpened peak
        #       location), soft band-max amplitude, deltas + EMA.
        sm_lat = torch.softmax(lat * 3.0, dim=1)
        bax_lat = torch.arange(lat.shape[1], device=lat.device,
                                dtype=lat.dtype).view(1, -1)
        lat_peak_pos = (sm_lat * bax_lat).sum(dim=1, keepdim=True)
        lat_peak_spr = torch.sqrt(((sm_lat * (bax_lat - lat_peak_pos) ** 2)
                                     .sum(dim=1, keepdim=True)).clamp(min=1e-6))
        lat_band_max = torch.logsumexp(lat * 3.0, dim=1, keepdim=True) / 3.0
        feats.append(lat_peak_pos)
        feats.append(lat_peak_spr)
        feats.append(lat_band_max)
        # Multi-tau dynamics of lat_peak_pos (formant glide speed)
        for tau in (4.5, 15.0, 60.0):
            feats.append(lat_peak_pos - leaky_lowpass(lat_peak_pos, tau))
        # 9b3. STMF on the lateral-inhibited cochleagram — captures rate-scale
        #      modulations of the SHARPENED spectral pattern. Use a focused
        #      subset of rate-scale (compact).
        for (ot, of) in ((4.0, 0.25), (4.0, -0.25),
                          (8.0, 0.25), (8.0, -0.25),
                          (4.0, 0.5)):
            feats.append(self._stmf_envelope(lat, ot, of))

        # (9c0 PER-BAND IF / centroid block dropped — was neutral in v18
        # ablation; freeing dim for the stronger within-window analytic stats.)

        # 9c00. WITHIN-WINDOW ANALYTIC-SIGNAL stats per band:
        #         log_mean, CoV, skew, kurt of envelope (voicing peakiness)
        #         asymmetry (CoV first-half - second-half: onset vs decay)
        #         per-band ENVELOPE-ACF pitch (log_f0) + HNR
        ww = self._within_window_envelope_stats(X_spec)
        # log_mean is redundant with band power log_c; skip.
        for k in ('cov', 'skew', 'kurt', 'asym',
                  'pitch_log_f0', 'pitch_hnr'):
            feats.append(ww[k])
        # (env-mod-band powers dropped — neutral in v27 ablation)
        # multi-tau highpass of the strongest voicing-discriminative stats only
        for k in ('cov', 'pitch_hnr'):
            feats.append(ww[k] - leaky_lowpass(ww[k], 15.0))

        # 9c01. VOICING-GATED CHANNEL SPLIT — multiply selected features by
        #       per-band voicing gate sigmoid(HNR-3) and 1-gate. The ridge
        #       gets SEPARATE voiced & unvoiced "channels" for the same
        #       feature — an interaction it cannot synthesize from HNR and
        #       feature alone.
        voiced_gate = torch.sigmoid((ww['pitch_hnr'] - 3.0) * 0.5)     # (T, F)
        unvoiced_gate = 1.0 - voiced_gate
        feats.append(voiced_gate * log_c)
        feats.append(unvoiced_gate * log_c)
        feats.append(voiced_gate * ww['cov'])
        feats.append(unvoiced_gate * ww['kurt'])
        # Across-band global gate
        global_gate = voiced_gate.mean(dim=1, keepdim=True)
        feats.append(global_gate * log_c)
        feats.append(global_gate)
        # 9c02. COMPACT bilinear interactions (v33). Memory-conservative:
        # only scalar / very-low-dim additions to fit within 192GB cgroup.
        # (a) opponent-band PRODUCT: low_pool × high_pool (bilinear, scalar)
        Fhalf = log_c.shape[1] // 2
        low_pool = log_c[:, :Fhalf].mean(dim=1, keepdim=True)
        hi_pool = log_c[:, Fhalf:].mean(dim=1, keepdim=True)
        global_voiced = global_gate                                      # (T,1)
        global_unvoiced = 1.0 - global_voiced                            # (T,1)
        spec_tilt = low_pool - hi_pool
        # 7 scalar interaction features
        feats.append(low_pool * hi_pool)
        feats.append(spec_tilt * global_voiced)
        feats.append(spec_tilt * global_unvoiced)
        feats.append(low_pool * global_voiced)
        feats.append(hi_pool * global_voiced)
        feats.append(low_pool * global_unvoiced)
        feats.append(hi_pool * global_unvoiced)
        # (b) Per-band transient gate (peaky envelope = plosive/burst)
        transient_gate = torch.sigmoid((ww['kurt'] - 1.0) * 0.5)         # (T,F)
        feats.append(transient_gate * log_c)                             # (T,F)
        feats.append(transient_gate * unvoiced_gate)                     # (T,F): unvoiced bursts
        # (c) Per-band onset gate × voicing complement (unvoiced onsets esp.
        # important: fricatives, plosives, transients)
        onset_gate = torch.sigmoid(d1 * 4.0)                             # (T,F)
        feats.append(onset_gate * unvoiced_gate)                         # (T,F)
        feats.append(onset_gate * voiced_gate)                           # (T,F)

        # 9d. CEPSTRUM SOURCE-ONLY BLOCK (v32 trimmed). The MFCC-like low-quefrency
        # cepstral coefficients were neutral (v32 = v30); only the
        # high-quefrency excitation peak adds genuinely new pitch info.
        log_p = torch.log(p.clamp(min=1e-10))
        n_freqs = log_p.shape[1]
        cep_full = torch.fft.irfft(log_p.to(torch.complex64),
                                    n=2 * (n_freqs - 1), dim=1).real
        q_lo = max(2, int(SR / 500.0))
        q_hi = min(cep_full.shape[1] - 1, int(SR / 75.0))
        cep_src = cep_full[:, q_lo:q_hi + 1]
        sm_q = torch.softmax(cep_src * 5.0, dim=1)
        q_axis = torch.arange(cep_src.shape[1], device=cep_src.device,
                                dtype=cep_src.dtype).view(1, -1) + q_lo
        q_peak_pos = (sm_q * q_axis).sum(dim=1, keepdim=True)
        q_peak_val = torch.logsumexp(cep_src * 5.0, dim=1, keepdim=True) / 5.0
        log_f0_cep = torch.log((SR / q_peak_pos.clamp(min=1.0)).clamp(min=20.0))
        cpp = q_peak_val - cep_src.mean(dim=1, keepdim=True)
        feats.append(log_f0_cep)
        feats.append(q_peak_val)
        feats.append(cpp)
        for tau in (4.5, 15.0, 60.0):
            feats.append(log_f0_cep - leaky_lowpass(log_f0_cep, tau))
            feats.append(cpp - leaky_lowpass(cpp, tau))
        del cep_full, cep_src, sm_q, log_p

        # 9e. NESTED MODULATION (v34) — modulation OF modulation.
        # For each band, take its 4-Hz modulation envelope env4 (already from
        # mod block) and bandpass IT at 0.5 Hz to get the slow envelope of the
        # fast modulation. This is a TWO-STAGE nonlinear cascade (|.| applied
        # twice) — a ridge cannot reconstruct it from band powers OR from env4.
        # Captures rhythmic structure of fast modulations (e.g. burst trains).
        env4 = envs_by_rate[4.0]                                        # (T, F)
        env8 = envs_by_rate[8.0]
        log_env4 = torch.log(env4.clamp(min=1e-8))
        log_env8 = torch.log(env8.clamp(min=1e-8))
        for r_slow in (0.5, 1.0):
            cos_k, sin_k = self._mod_kernels[r_slow]
            cc4 = conv_temporal(log_env4, cos_k)
            ss4 = conv_temporal(log_env4, sin_k)
            nested4 = torch.sqrt(cc4 * cc4 + ss4 * ss4 + 1e-12)
            feats.append(torch.log(nested4.clamp(min=1e-8)))
            cc8 = conv_temporal(log_env8, cos_k)
            ss8 = conv_temporal(log_env8, sin_k)
            nested8 = torch.sqrt(cc8 * cc8 + ss8 * ss8 + 1e-12)
            feats.append(torch.log(nested8.clamp(min=1e-8)))
        # Running mean & variance of these (texture of nested modulation)
        for tau in (60.0,):
            mu = leaky_lowpass(log_env4, tau)
            var = leaky_lowpass((log_env4 - mu) ** 2, tau).clamp(min=1e-10)
            feats.append(torch.log(var))
            mu = leaky_lowpass(log_env8, tau)
            var = leaky_lowpass((log_env8 - mu) ** 2, tau).clamp(min=1e-10)
            feats.append(torch.log(var))

        # 9c. FORMANT-LIKE PEAK TRACKING on log_c — compact: F1, F2, F2-F1, amps.
        nF = self.N_BANDS
        f1_band = log_c[:, :nF // 2]
        f2_band = log_c[:, nF // 2:]
        w1 = torch.softmax(f1_band * 2.0, dim=1)
        w2 = torch.softmax(f2_band * 2.0, dim=1)
        ch1 = torch.arange(nF // 2, device=log_c.device,
                            dtype=log_c.dtype).view(1, -1)
        ch2 = torch.arange(nF // 2, device=log_c.device,
                            dtype=log_c.dtype).view(1, -1) + nF // 2
        f1_loc = (w1 * ch1).sum(dim=1, keepdim=True)
        f2_loc = (w2 * ch2).sum(dim=1, keepdim=True)
        f1_amp = (w1 * f1_band).sum(dim=1, keepdim=True)
        f2_amp = (w2 * f2_band).sum(dim=1, keepdim=True)
        formants = torch.cat([f1_loc, f2_loc, f2_loc - f1_loc, f1_amp, f2_amp], dim=1)
        feats.append(formants)

        # 10. PER-FRAME spectral pattern statistics — nonlinear summaries of the
        #     instantaneous cochleagram pattern.
        F = log_c.shape[1]
        # Softmax distribution over bands → entropy + Gini-style sparsity
        p_dist = torch.softmax(log_c, dim=1).clamp(min=1e-10)         # (T, F)
        entropy = -(p_dist * torch.log(p_dist)).sum(dim=1, keepdim=True)
        sparsity = (p_dist ** 2).sum(dim=1, keepdim=True)             # 1/N_effective
        feats.append(entropy)
        feats.append(sparsity)
        feats.append(torch.log(sparsity.clamp(min=1e-8)))
        # Spectral centroid (band index weighted by p) and spread
        band_idx = torch.arange(F, device=log_c.device,
                                 dtype=log_c.dtype).view(1, -1)
        centroid = (p_dist * band_idx).sum(dim=1, keepdim=True)
        spread = torch.sqrt(((p_dist * (band_idx - centroid) ** 2)
                              .sum(dim=1, keepdim=True)).clamp(min=1e-8))
        feats.append(centroid)
        feats.append(spread)
        # Multi-tau dynamics of centroid + entropy + spread (highpass)
        ce_block = torch.cat([centroid, spread, entropy, torch.log(sparsity.clamp(min=1e-8))], dim=1)
        feats.append(ce_block)
        for tau in (4.5, 15.0, 60.0):
            feats.append(ce_block - leaky_lowpass(ce_block, tau))
        # Spectral CONTRAST: log of running max-band minus running min-band ratio.
        # Max/min via softmax(log_c * k) for differentiable proxy, then their log diff.
        max_band = (torch.softmax(log_c * 5.0, dim=1) * log_c).sum(dim=1, keepdim=True)
        min_band = (torch.softmax(-log_c * 5.0, dim=1) * log_c).sum(dim=1, keepdim=True)
        contrast = max_band - min_band
        feats.append(contrast)
        feats.append(max_band)
        feats.append(min_band)
        for tau in (15.0, 60.0):
            feats.append(contrast - leaky_lowpass(contrast, tau))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "TEXTURE-v34"
model_description = (
    "TEXTURE v34: restored full STMF (22 filters, trim hurt 5pt in v33). "
    "Trimmed cepstrum to source-only pitch/CPP features (MFCC-like c1..c24 "
    "added nothing in v32). Added NESTED MODULATION block: |gabor 0.5Hz|, "
    "|gabor 1Hz| of log_env at rates 4Hz & 8Hz — modulation OF modulation, "
    "TWO-STAGE nonlinear cascade (|.| of |.|) that captures rhythmic structure "
    "of fast modulations (burst trains, syllable substructure). + running "
    "variance of log_env4 and log_env8. v30=0.0690, v32=0.0691, v33=0.0685."
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

    # Quick dim probe
    dummy = np.zeros((10, embedder.win), dtype=np.float32)
    sample = embedder(dummy)
    print(f"feature dim per frame = {sample.shape[1]}")

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
