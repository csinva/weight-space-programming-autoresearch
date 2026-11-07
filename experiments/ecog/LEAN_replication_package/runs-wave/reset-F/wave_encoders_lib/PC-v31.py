"""
PC-v1: Predictive-coding auditory front-end.

Paradigm: auditory cortex / STG encodes PREDICTION ERROR, not the stimulus.
Build fixed-coefficient predictors of the cochleagram at multiple horizons
and emit GENUINELY NONLINEAR functions of the residuals (rectified, squared,
precision-weighted, cross-band products) — quantities a linear ridge over
the bare bands cannot synthesize from its FIR-delayed inputs.

Components:
  1. Long-window log-cochleagram (40 ERB bands) as a linear-baseline anchor.
  2. Running precision (EMA of |x-EMA(x)|) at multiple taus — log(sigma).
  3. Per-band AR residuals at horizons k in {1,2,4,8,16} for TWO predictors:
       - hold:   pred[t] = x[t-k]      (zero-order)
       - lin:    pred[t] = 2*x[t-k] - x[t-2k]  (linear extrapolation)
     Each residual emitted as: pos-rectified, neg-rectified, squared,
     precision-weighted z, and (z)^2 — all nonlinear in the bands.
  4. Cross-band divisive normalization (residual against running broadband
     loudness) — log(band) - log(EMA(loudness)).
  5. Spatial (channel-axis) contextual residual: band minus mean of its
     nearest neighbors — captures spectral edges & deviance.
  6. Surprise scalars: across-band sums of |r|, r^2, pos(r), neg(r) per k.
  7. Cross-horizon residual products r_k1 * r_k2 (synergy of multi-scale
     surprise).

Closed-form weights only; no training/PCA/etc. Ridge readout (the only
data-fit step) is done by the harness.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.data import SR
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR = "/path/to/scratch/feat_cache_reset_F"
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


def gammatone_powerbank(win, n_filters=40, f_min=50.0, f_max=8000.0, order=4):
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))


def power_spectrum(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2


def complex_stft(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win)                  # complex (T, F)


def causal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Causal exponential moving average via FIR conv along time (axis 0)."""
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = F.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = F.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def past_shift(x: torch.Tensor, k: int) -> torch.Tensor:
    """x[t-k] with replicate padding for t<k."""
    if k <= 0:
        return x
    T = x.shape[0]
    if k >= T:
        return x[:1].expand(T, -1).contiguous()
    return torch.cat([x[:1].expand(k, -1), x[:-k]], dim=0)


# ---------------------------------------------------------------------------
# Predictive-coding embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0
    N_FILTERS = 22
    HORIZONS = (1, 2, 4, 8, 16)                # short AR prediction horizons (frames)
    LONG_HORIZONS = (30, 60, 90)               # long AR horizons (1s, 2s, 3s)
    PRECISION_TAUS = (15.0, 60.0)              # for log(sigma)
    Z_TAUS = (30.0,)                           # for precision-weighting residuals
    SPATIAL_SIGMAS = (1.0, 3.0)                # channel-axis context smoothing
    KL_HORIZONS = (1, 4, 16, 60)               # for spectral-distribution KL surprise
    KL_EMA_TAUS = (30.0, 120.0)                # for KL against running spectral profile

    SHORT_WIN = 1024
    MED_WIN = 2048                                                   # [v24]

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.N_FILTERS).to(device)
        # [v10] Short window for better temporal resolution (~23 ms)
        self.fb_short = gammatone_powerbank(self.SHORT_WIN, n_filters=self.N_FILTERS).to(device)
        # [v24] Medium window (~46 ms) — bridges long (93 ms) and short (23 ms)
        self.fb_med = gammatone_powerbank(self.MED_WIN, n_filters=self.N_FILTERS).to(device)
        # Waveform-level fixed AR-predictor residual responses |H_p(ω)|^2
        # for p-th finite-difference predictors (polynomial extrap order p).
        # H_p(ω) = (1 - e^{-iω})^p  =>  |H_p|^2 = (2 sin(ω/2))^(2p)
        omega = 2.0 * np.pi * np.fft.rfftfreq(self.win)                # [0, π]
        self.wave_pred_orders = (1, 2, 4)
        self.wave_pred_responses = {
            p: torch.from_numpy(
                ((2.0 * np.sin(omega / 2.0)) ** (2 * p)).astype(np.float32)
            ).to(device)
            for p in self.wave_pred_orders
        }
        # [v4] Natural inter-frame phase advance per FFT bin = 2π * k * HOP / win
        # HOP = SR/30 (the harness framing); window length = self.win = 4096.
        HOP = SR // 30
        n_freq = self.win // 2 + 1
        k_idx = np.arange(n_freq, dtype=np.float64)
        omega_nat_per_hop = 2.0 * np.pi * k_idx * HOP / self.win        # natural phase advance per hop, per bin
        self.phase_unwind = torch.from_numpy(
            np.exp(-1j * omega_nat_per_hop).astype(np.complex64)
        ).to(device)                                                     # shape (n_freq,)
        # Phase-coherence EMA taus (frames)
        self.phase_coh_taus = (3.0, 10.0, 30.0)
        # [v5] Alpha-function IRF kernels for surprise → HG-response conversion.
        # alpha(t; tau) = (t/tau) exp(-t/tau), peaks at t=tau (frames).
        # Multiple taus span fast onsets (50 ms) → slow integration (1 s).
        self.alpha_taus = (1.5, 4.0, 9.0, 20.0, 45.0)
        self.alpha_kernels = {}
        for tau in self.alpha_taus:
            K = int(np.ceil(6 * tau)) + 1
            t = np.arange(K).astype(np.float32)
            k = (t / tau) * np.exp(-t / tau)
            k = (k / (k.sum() + 1e-12)).astype(np.float32)
            self.alpha_kernels[tau] = torch.from_numpy(k).to(device)
        # [v9] gamma-shape IRF (n=2): sharper rise, later peak — complementary
        # to alpha. Captures slower onset/sustained dynamics.
        self.gamma_taus = (2.0, 6.0, 15.0)
        self.gamma_kernels = {}
        for tau in self.gamma_taus:
            K = int(np.ceil(8 * tau)) + 1
            t = np.arange(K).astype(np.float32)
            k = (t / tau) ** 2 * np.exp(-t / tau)
            k = (k / (k.sum() + 1e-12)).astype(np.float32)
            self.gamma_kernels[tau] = torch.from_numpy(k).to(device)

    def _kern_conv(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        K = k.shape[0]
        xp = x.transpose(0, 1).unsqueeze(0)                           # (1, C, T)
        xp_p = F.pad(xp, (K - 1, 0), mode='replicate')
        C = x.shape[1]
        kk = k.flip(0).view(1, 1, -1).expand(C, 1, K).contiguous()
        y = F.conv1d(xp_p, kk, groups=C)
        return y.squeeze(0).transpose(0, 1).contiguous()

    def _alpha_conv(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        return self._kern_conv(x, self.alpha_kernels[tau])

    def _gamma_conv(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        return self._kern_conv(x, self.gamma_kernels[tau])

    def _alpha_conv_acausal(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """Time-reversed alpha-IRF — integrates FUTURE surprise."""
        return torch.flip(self._kern_conv(torch.flip(x, dims=[0]),
                                          self.alpha_kernels[tau]), dims=[0])

    def _spatial_context(self, log_e: torch.Tensor, sigma: float) -> torch.Tensor:
        """Mean of neighbors along channel axis (Gaussian, ZEROED at self) —
        each band's expected value given its spectral neighbors."""
        k = max(1, int(np.ceil(3 * sigma)))
        coords = torch.arange(-k, k + 1, device=log_e.device, dtype=log_e.dtype)
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g[k] = 0.0                              # exclude self → context-only predictor
        g = g / g.sum()
        # conv along channel axis: shape (T,C) -> (1,C,T) viewed as (1,1,C) per t?
        # simpler: treat as (T, C) -> (T,1,C), conv1d with groups=1 across C
        xp = log_e.unsqueeze(1)                 # (T, 1, C)
        ker = g.view(1, 1, -1)
        y = F.conv1d(xp, ker, padding=k)
        return y.squeeze(1)                     # (T, C)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []

        # ---- Long-window log-cochleagram (linear baseline anchor) ----
        X = complex_stft(x, self.win, self.device)                  # complex (T, F)
        p = (X.real * X.real + X.imag * X.imag).contiguous()        # (T, F) power
        e = p @ self.fb                                             # (T, C)
        log_e = torch.log(e.clamp(min=1e-10))                       # (T, C)
        feats.append(log_e)
        # ---- [v10] Short-window cochleagram (~23 ms) — better temporal resolution
        # Center-crop each long window to SHORT_WIN samples → no re-windowing/timing
        start = (self.win - self.SHORT_WIN) // 2
        xs = x[:, start:start + self.SHORT_WIN]
        p_short = power_spectrum(xs, self.SHORT_WIN, self.device)   # (T, F_short)
        e_short = p_short @ self.fb_short                            # (T, C)
        log_e_short = torch.log(e_short.clamp(min=1e-10))            # (T, C)
        feats.append(log_e_short)
        # ---- [v24] Medium-window cochleagram (~46 ms) + onset IRF ----
        start_m = (self.win - self.MED_WIN) // 2
        xm = x[:, start_m:start_m + self.MED_WIN]
        p_med = power_spectrum(xm, self.MED_WIN, self.device)
        log_e_med = torch.log((p_med @ self.fb_med).clamp(min=1e-10))
        feats.append(log_e_med)
        onset_med = (log_e_med - past_shift(log_e_med, 1)).clamp(min=0.0)
        for tau in (1.5, 4.0):
            feats.append(self._alpha_conv(onset_med, tau))               # (T, C)
            feats.append(self._alpha_conv_acausal(onset_med, tau))       # (T, C)
        # Onset/offset on the short-window cochleagram + alpha-IRF block
        d_short_k1 = log_e_short - past_shift(log_e_short, 1)
        onset_short = d_short_k1.clamp(min=0.0)
        offset_short = (-d_short_k1).clamp(min=0.0)                   # [v11]
        # [v16] dropped onset_short_k4 IRF (3*22=66) to free budget for longer-tau acausal
        for tau in (1.5, 4.0, 9.0):
            feats.append(self._alpha_conv(onset_short, tau))         # (T, C)
            feats.append(self._alpha_conv(offset_short, tau))        # [v11] (T, C)
            feats.append(self._alpha_conv(log_e_short, tau))         # (T, C)

        # [v17] dropped entire spectral phase PC block (199 features, no measured help)
        del X

        # ---- Running PRECISION (noise-scale) at multiple taus ----
        # nonlinear (abs+log) → ridge can't recover from bands alone
        log_sigmas = []
        for tau in self.PRECISION_TAUS:
            mu = causal_ema(log_e, tau)
            sig = causal_ema((log_e - mu).abs(), tau).clamp(min=1e-3)
            log_sigmas.append(torch.log(sig))
        feats.extend(log_sigmas)

        # Pre-compute the precision smoothings used for z-residual weighting
        # (computed lazily per-residual below to keep memory bounded)

        residual_bank = {}                      # for cross-horizon products

        for k in self.HORIZONS:
            xk  = past_shift(log_e, k)
            x2k = past_shift(log_e, 2 * k)

            # Predictor 1: HOLD — pred = x[t-k]
            r_hold = log_e - xk

            # Predictor 2: LINEAR EXTRAP — pred = 2*x[t-k] - x[t-2k]
            r_lin = log_e - (2.0 * xk - x2k)

            # Signed residuals at k > 3 (beyond ridge's FIR window) are NOT
            # linearly synthesizable from FIR-delayed bands. Include them.
            if k > 3:
                feats.append(r_hold)
                feats.append(r_lin)

            # NONLINEAR derivatives of each residual — ALWAYS useful:
            for r in (r_hold, r_lin):
                feats.append(r.clamp(min=0.0))                          # pos-rect (onset)
                feats.append((-r).clamp(min=0.0))                       # neg-rect (offset)
                feats.append(r * r)                                     # squared energy
                # Precision-weighted residual
                for tau in self.Z_TAUS:
                    sig = causal_ema(r.abs(), tau).clamp(min=1e-3)
                    z = r / sig
                    feats.append(z * z)
                    feats.append(z.clamp(min=0.0))
                    feats.append((-z).clamp(min=0.0))

            residual_bank[k] = r_hold

        # ---- DIVISIVE / CROSS-BAND NORMALIZATION ----
        # Each band's log-energy minus log of the EMA broadband loudness.
        # Nonlinear (exp/log composition + running EMA of nonlinear quantity).
        loud = log_e.mean(dim=1, keepdim=True)                          # (T,1) ~ log loudness
        for tau in (15.0, 60.0):                                         # [v28] dropped tau=150
            broad = causal_ema(loud, tau)
            feats.append(log_e - broad)                                  # contrast-normalized cochleagram
        # Multiplicative cross-band gain control: log of (band / running broadband energy)
        e_total = e.sum(dim=1, keepdim=True)                            # (T,1)
        ec = causal_ema(e_total, 60.0).clamp(min=1e-6)
        feats.append(log_e - torch.log(ec))                              # divisive normalization
        # [v28] drop divisive_norm tau=150 (22) to budget for modulation block

        # [v29] dropped SPATIAL (channel-axis) context residuals (132 features)
        # to budget for the modulation-filterbank block while restoring the
        # alpha-IRF on log_e. Original code retained below for reference:
        # for sigma in self.SPATIAL_SIGMAS:
        #     ctx = self._spatial_context(log_e, sigma); sp_r = log_e - ctx
        #     feats.append(sp_r.clamp(min=0.0)); feats.append((-sp_r).clamp(min=0.0))
        #     feats.append(sp_r * sp_r)

        # ---- SURPRISE SCALARS (across-band aggregates per horizon) ----
        for k in self.HORIZONS:
            r = residual_bank[k]
            feats.append(r.abs().sum(dim=1, keepdim=True))
            feats.append((r * r).sum(dim=1, keepdim=True))
            feats.append(r.clamp(min=0.0).sum(dim=1, keepdim=True))     # onset strength
            feats.append((-r).clamp(min=0.0).sum(dim=1, keepdim=True))  # offset strength
            # Spectral spread of the absolute residual (where in the spectrum
            # is the surprise concentrated)
            ar = r.abs()
            denom = ar.sum(dim=1, keepdim=True).clamp(min=1e-6)
            ch_idx = torch.arange(r.shape[1], device=r.device, dtype=r.dtype).view(1, -1)
            centroid = (ar * ch_idx).sum(dim=1, keepdim=True) / denom
            feats.append(centroid)

        # ---- CROSS-HORIZON RESIDUAL PRODUCTS ----
        for k1, k2 in [(1, 4), (2, 8), (4, 16)]:
            prod = residual_bank[k1] * residual_bank[k2]
            feats.append(prod)                                          # synergistic surprise

        # ---- [v5] ALPHA-IRF-CONVOLVED SURPRISE (HG-response-shaped onsets) ----
        # Per-band positive-residual onset signal convolved with biologically
        # plausible alpha-function IRFs at multiple timescales — exactly the
        # functional form of an HG response to an onset event.
        d_k1 = log_e - past_shift(log_e, 1)
        d_k4 = log_e - past_shift(log_e, 4)
        onset_k1, offset_k1 = d_k1.clamp(min=0.0), (-d_k1).clamp(min=0.0)
        onset_k4 = d_k4.clamp(min=0.0)
        # Cross-band surprise scalars (energy of onset / offset) at k=1
        surprise_scalar_k1 = onset_k1.sum(dim=1, keepdim=True)
        for tau in self.alpha_taus:
            feats.append(self._alpha_conv(onset_k1, tau))                # (T, C)
            # [v28] dropped alpha-IRF on onset_k4 (5*22=110) to budget modulation
            # [v30] dropped alpha-IRF on offset_k1 (5*22=110) to budget sub-frame block
            feats.append(self._alpha_conv(surprise_scalar_k1, tau))      # (T, 1)
        # [v29] RESTORED alpha-IRF on log_e (lowpass HG-smeared energy)
        for tau in self.alpha_taus:
            feats.append(self._alpha_conv(log_e, tau))                   # (T, C)
        # [v13] removed gamma-IRF (small win) to free budget for acausal expansion
        # [v12+v13] BACKWARD (acausal) alpha-IRF — integrates FUTURE surprise.
        for tau in (1.5, 4.0, 9.0):
            feats.append(self._alpha_conv_acausal(onset_k1, tau))        # (T, C)
            feats.append(self._alpha_conv_acausal(log_e, tau))           # (T, C)
        # [v13] Acausal IRF on additional base signals
        for tau in (1.5, 4.0, 9.0):
            feats.append(self._alpha_conv_acausal(offset_k1, tau))       # (T, C)
            feats.append(self._alpha_conv_acausal(onset_short, tau))     # (T, C)
        # [v16/v17] Longer-tau acausal IRF (~700 ms future) on productive signals
        for sig in (onset_k1, log_e, offset_k1, onset_short, offset_short, log_e_short):
            feats.append(self._alpha_conv_acausal(sig, 20.0))            # (T, C)
        # [v17] Very long-tau acausal (~1.5 s future) on long-window signals
        for sig in (onset_k1, log_e, offset_k1):
            feats.append(self._alpha_conv_acausal(sig, 45.0))            # (T, C)

        # ---- [v10] dropped long-horizon residuals (3*4*22=264) — kept as
        #     scalar versions only in surprise-scalar block; net no help ----

        # ---- [v3] INFORMATION-THEORETIC SPECTRAL SURPRISE ----
        # Treat each frame's cochleagram as a categorical distribution over
        # bands. Entropy, cross-entropy and KL divergences are NONLINEAR
        # scalars no linear remix of the band log-energies can synthesize.
        p_t = torch.softmax(log_e, dim=1)                                # (T, C)
        log_p_t = torch.log_softmax(log_e, dim=1)                        # (T, C)
        # Spectral entropy H = -sum p log p
        H = -(p_t * log_p_t).sum(dim=1, keepdim=True)
        feats.append(H)
        # Entropy at sharpened distribution (higher temperature exponent)
        for beta in (0.5, 2.0):
            lp = torch.log_softmax(beta * log_e, dim=1)
            pp = torch.exp(lp)
            feats.append(-(pp * lp).sum(dim=1, keepdim=True))
        # KL(p_t || p_{t-k}) — spectral distribution surprise vs k frames ago
        for k in self.KL_HORIZONS:
            log_p_k = past_shift(log_p_t, k)
            kl = (p_t * (log_p_t - log_p_k)).sum(dim=1, keepdim=True)
            feats.append(kl)
            # Reverse KL too — different sensitivity
            p_k = past_shift(p_t, k)
            kl_rev = (p_k * (log_p_k - log_p_t)).sum(dim=1, keepdim=True)
            feats.append(kl_rev)
            # Jensen-Shannon style midpoint divergence (symmetric, bounded)
            m = 0.5 * (p_t + p_k)
            log_m = torch.log(m.clamp(min=1e-10))
            js = 0.5 * (p_t * (log_p_t - log_m)).sum(dim=1, keepdim=True) + \
                 0.5 * (p_k * (log_p_k - log_m)).sum(dim=1, keepdim=True)
            feats.append(js)
        # KL(p_t || EMA(p, tau))
        for tau in self.KL_EMA_TAUS:
            p_ema = causal_ema(p_t, tau).clamp(min=1e-10)
            log_p_ema = torch.log(p_ema)
            kl_ema = (p_t * (log_p_t - log_p_ema)).sum(dim=1, keepdim=True)
            feats.append(kl_ema)
        # Surprisal of the MOST-active band (max log-prob): -log p(argmax)
        max_log_p, _ = log_p_t.max(dim=1, keepdim=True)
        feats.append(-max_log_p)
        # Effective bandwidth = exp(H) / C — fraction of bands "active"
        feats.append(torch.exp(H) / float(log_e.shape[1]))

        # [v11] dropped v2 wave-level prediction cochleagrams (no measured help)

        # ---- [v31] PITCH / PERIODICITY via per-frame waveform autocorrelation ----
        # The cochleagram is a power spectrum and DOESN'T encode voicing /
        # periodicity. STG strongly distinguishes voiced from unvoiced speech.
        # Per-frame autocorrelation via FFT (Wiener-Khinchin): ifft(|FFT(x*hann)|^2).
        # Restrict to pitch lags (~16 kHz/lag): max → periodicity strength,
        # argmax → pitch period, autocorr at several pitch lags → harmonic comb.
        # All fully nonlinear (max, ratio) w.r.t. the cochleagram bands.
        hann = torch.hann_window(self.win, periodic=True, device=self.device)
        Xw = torch.fft.rfft(x * hann, n=self.win)                         # (T, F)
        Pw = (Xw.real * Xw.real + Xw.imag * Xw.imag).contiguous()          # (T, F)
        acf = torch.fft.irfft(Pw, n=self.win, dim=1)                       # (T, win)
        acf_norm = acf / (acf[:, :1] + 1e-10)                              # normalize by AC(0)
        # pitch lags: SR=16000, range 50-500 Hz → lags 32..320 samples
        LAG_MIN, LAG_MAX = 32, 320
        pitch_zone = acf_norm[:, LAG_MIN:LAG_MAX]                           # (T, 288)
        periodicity, pitch_idx = pitch_zone.max(dim=1)                      # (T,), (T,)
        feats.append(periodicity.unsqueeze(1))                              # periodicity strength
        feats.append((pitch_idx.float() / (LAG_MAX - LAG_MIN)).unsqueeze(1))  # normalized pitch idx
        # log-HNR-style sigmoidal: log(p / (1-p+eps))
        p_clamp = periodicity.clamp(-0.999, 0.999)
        feats.append((torch.log((1 + p_clamp).clamp(min=1e-3)
                                / (1 - p_clamp).clamp(min=1e-3))
                      * 0.5).unsqueeze(1))                                  # atanh ≈ HNR
        # Autocorr at fixed pitch lags (a harmonic comb readout)
        for lag in (40, 60, 80, 110, 140, 180, 220, 260):
            feats.append(acf_norm[:, lag:lag + 1])                          # (T, 1)
        # Low-pitch vs high-pitch periodicity (male vs female / vowel quality)
        low_peak = acf_norm[:, 80:200].max(dim=1).values                    # (T,)
        high_peak = acf_norm[:, 32:80].max(dim=1).values
        feats.append((low_peak - high_peak).unsqueeze(1))
        # Periodicity averaged over pitch range — overall voicing
        feats.append(acf_norm[:, LAG_MIN:LAG_MAX].mean(dim=1, keepdim=True))
        # Second peak strength (zero-out region ±10 around first peak)
        ac_mod = pitch_zone.clone()
        # use scatter to zero ±10 around pitch_idx per row
        T_ = ac_mod.shape[0]
        row_idx = torch.arange(T_, device=self.device).unsqueeze(1)
        col_offsets = torch.arange(-10, 11, device=self.device).unsqueeze(0)  # (1, 21)
        col_idx = (pitch_idx.unsqueeze(1) + col_offsets).clamp(0, ac_mod.shape[1] - 1)
        ac_mod[row_idx, col_idx] = -10.0
        second_peak = ac_mod.max(dim=1).values
        feats.append(second_peak.unsqueeze(1))                              # (T, 1)
        feats.append((periodicity - second_peak).unsqueeze(1))              # peak prominence
        del acf, acf_norm, pitch_zone, Pw, Xw

        # ---- [v30] SUB-FRAME ENERGY DYNAMICS ----
        # Split each 4096-sample window into 4 sub-windows of 1024 samples
        # (~35 ms each). Compute a cochleagram per sub-window → exposes WITHIN-
        # frame energy dynamics that the 4096-sample (~140 ms) cochleagram
        # smears out completely. Features per band: dynamic range across the
        # 4 sub-windows (peakiness), SOFT argmax position (where in the frame
        # the energy peaks — plosive timing), and attack direction (first
        # minus last sub-window). All NONLINEAR in log_e.
        N_SUB = 4
        SUB_LEN = self.win // N_SUB                                        # 1024
        x_sub = x.view(x.shape[0], N_SUB, SUB_LEN)                         # (T, 4, 1024)
        xs_flat = x_sub.reshape(-1, SUB_LEN)
        p_sub = power_spectrum(xs_flat, SUB_LEN, self.device)              # (T*4, F_short)
        e_sub = p_sub @ self.fb_short                                       # (T*4, 22)
        log_e_sub = torch.log(e_sub.clamp(min=1e-10)).view(
            x.shape[0], N_SUB, self.N_FILTERS)                              # (T, 4, 22)
        max_sub = log_e_sub.max(dim=1).values                               # (T, 22)
        min_sub = log_e_sub.min(dim=1).values
        feats.append(max_sub - min_sub)                                     # dynamic range
        sw = torch.softmax(log_e_sub, dim=1)                                # (T, 4, 22)
        positions = torch.arange(N_SUB, device=self.device,
                                 dtype=log_e_sub.dtype).view(1, N_SUB, 1)
        peak_pos = (sw * positions).sum(dim=1)                              # (T, 22)
        feats.append(peak_pos)                                              # soft argmax position
        feats.append(log_e_sub[:, 0, :] - log_e_sub[:, -1, :])              # attack direction
        # Variance across sub-windows (within-frame "spikiness") — 4th moment
        # variant — nonlinear in log_e_sub
        feats.append(log_e_sub.var(dim=1, unbiased=False))                  # (T, 22)
        del log_e_sub, e_sub, p_sub, xs_flat, x_sub

        # ---- [v28] BANDPASS MODULATION FILTERBANK on log-cochleagram envelope ----
        # Genuinely BANDPASS (not lowpass like the alpha/gamma IRF block) at the
        # cortical modulation rates known to be encoded in STG. Per-band
        # analytic signal via FFT-Gabor → magnitude (= local AM energy at that
        # rate, nonlinear) + sin/cos of instantaneous modulation PHASE (=where
        # in the modulation cycle we are, nonlinear). Demodulation, not smoothing.
        T_frames = log_e.shape[0]
        Y_le = torch.fft.fft(log_e, dim=0)                                # (T, C) complex
        mod_freqs = torch.fft.fftfreq(T_frames, d=1.0 / 30.0).to(self.device)
        MOD_RATES = (1.0, 2.0, 4.0, 8.0, 12.0)                            # Hz
        Q_MOD = 1.5
        mag_scalars = []
        for fc in MOD_RATES:
            bw = fc / Q_MOD
            H = torch.exp(-0.5 * ((mod_freqs - fc) / bw) ** 2)
            H = torch.where(mod_freqs < 0, torch.zeros_like(H), 2.0 * H)
            yL = torch.fft.ifft(Y_le * H.unsqueeze(1), dim=0)             # (T, C) complex
            magL = yL.abs()                                                # nonlinear |·|
            # Instantaneous modulation phase — only sin (drop cos to fit budget;
            # sin alone still gives a 90°-rotated read of the modulation cycle).
            sinph = yL.imag / (magL + 1e-6)                                # = sin(phase)
            feats.append(magL)
            feats.append(sinph)
            # across-band scalar: total modulation power at this rate
            mag_scalars.append(magL.sum(dim=1, keepdim=True))              # (T, 1)
            # cross-band modulation contrast: max - mean over bands
            feats.append((magL.max(dim=1, keepdim=True).values
                          - magL.mean(dim=1, keepdim=True)))
        feats.extend(mag_scalars)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "PC-v31"
model_description = (
    "PC-v30 (test_corr=0.0702) + PITCH/PERIODICITY block via per-frame waveform "
    "autocorrelation (FFT-based Wiener-Khinchin): periodicity strength (max of "
    "normalized AC in pitch lags 32-320 samples), pitch period (argmax), atanh-"
    "HNR, AC at 8 fixed pitch lags (harmonic comb), low- vs high-pitch peak "
    "difference, mean periodicity over pitch range, second-peak strength, peak "
    "prominence. ~16 NEW features. Cochleagram is power-spectrum-only — does NOT "
    "encode voicing; STG strongly distinguishes voiced/unvoiced — a fully new axis."
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
