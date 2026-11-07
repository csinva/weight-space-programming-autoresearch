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
    N_FILTERS = 24
    HORIZONS = (1, 2, 4, 8, 16)                # short AR prediction horizons (frames)
    LONG_HORIZONS = (30, 60, 90)               # long AR horizons (1s, 2s, 3s)
    PRECISION_TAUS = (15.0, 60.0)              # for log(sigma)
    Z_TAUS = (30.0,)                           # for precision-weighting residuals
    SPATIAL_SIGMAS = (1.0, 3.0)                # channel-axis context smoothing
    KL_HORIZONS = (1, 4, 16, 60)               # for spectral-distribution KL surprise
    KL_EMA_TAUS = (30.0, 120.0)                # for KL against running spectral profile

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.N_FILTERS).to(device)
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
        # [v6] gamma-shape IRF (n=2): sharper-peaking, common HRF form.
        # gamma(t; tau, n) = (t/tau)^n exp(-t/tau)
        self.gamma_taus = (2.0, 6.0, 15.0)
        self.gamma_kernels = {}
        for tau in self.gamma_taus:
            K = int(np.ceil(8 * tau)) + 1
            t = np.arange(K).astype(np.float32)
            k = (t / tau) ** 2 * np.exp(-t / tau)
            k = (k / (k.sum() + 1e-12)).astype(np.float32)
            self.gamma_kernels[tau] = torch.from_numpy(k).to(device)

    def _kern_conv(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Convolve (T, C) signal along time with a 1-D causal kernel k."""
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

        # ---- [v4] SPECTRAL PHASE PREDICTIVE-CODING ----
        # Predicted: each FFT bin's complex coefficient simply advances its
        # phase by 2π * k * HOP / win per frame for a stationary tone.
        # Residual = wrap(arg(X[t] * conj(X[t-1])) - φ_nat(k)) ∈ [-π, π].
        # Energy-weighted, gathered into ERB bands → per-band phase surprise.
        X_prev = torch.cat([X[:1], X[:-1]], dim=0)
        # Unit-magnitude ratio (avoid amplitude leakage)
        denom = (X.abs() * X_prev.abs() + 1e-8)
        ratio = (X * X_prev.conj()) / denom                          # ≈ exp(i*Δφ) when both magnitudes are real
        # Unwind the natural inter-frame advance:
        phase_residual_complex = ratio * self.phase_unwind.unsqueeze(0)
        phase_residual_angle = phase_residual_complex.angle()        # wrapped [-π,π], (T, F)
        # Energy-weighted band L1 of |phase residual| → per-band surprise scalar
        denom_e = (p @ self.fb).clamp(min=1e-10)                      # = e
        weighted_l1 = (p * phase_residual_angle.abs()) @ self.fb     # (T, C)
        phase_l1_band = weighted_l1 / denom_e
        feats.append(phase_l1_band)
        # Surprise SCALAR: total phase-residual-energy across the spectrum
        total_phase_surprise = (p * phase_residual_angle.abs()).sum(
            dim=1, keepdim=True) / p.sum(dim=1, keepdim=True).clamp(min=1e-10)
        feats.append(total_phase_surprise)
        del X, X_prev, ratio, phase_residual_complex

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
        for tau in (15.0, 60.0, 150.0):
            broad = causal_ema(loud, tau)
            feats.append(log_e - broad)                                  # contrast-normalized cochleagram
        # Multiplicative cross-band gain control: log of (band / running broadband energy)
        e_total = e.sum(dim=1, keepdim=True)                            # (T,1)
        ec = causal_ema(e_total, 60.0).clamp(min=1e-6)
        feats.append(log_e - torch.log(ec))                              # divisive normalization

        # ---- SPATIAL CONTEXT RESIDUAL — single sigma only ----
        ctx = self._spatial_context(log_e, 2.0)
        sp_r = log_e - ctx
        feats.append(sp_r * sp_r)

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

        # ---- [v5/v6] ALPHA-IRF-CONVOLVED SURPRISE (HG-response-shaped onsets) ----
        # Per-band onset/offset/surprise signals convolved with biologically
        # plausible alpha-function IRFs at multiple timescales — exactly the
        # functional form of an HG response to an onset event.
        d_k1 = log_e - past_shift(log_e, 1)                              # signed Δlog_e at k=1
        d_k4 = log_e - past_shift(log_e, 4)
        d_k16 = log_e - past_shift(log_e, 16)
        onset_k1, offset_k1 = d_k1.clamp(min=0.0), (-d_k1).clamp(min=0.0)
        onset_k4, offset_k4 = d_k4.clamp(min=0.0), (-d_k4).clamp(min=0.0)
        abs_k16 = d_k16.abs()                                            # long-range deviance
        surprise_scalar_k1 = onset_k1.sum(dim=1, keepdim=True)
        offset_scalar_k1 = offset_k1.sum(dim=1, keepdim=True)
        log_phase_l1 = torch.log(phase_l1_band.clamp(min=1e-6))          # log of phase-surprise per band

        for tau in self.alpha_taus:
            feats.append(self._alpha_conv(onset_k1, tau))                # (T, C)
            feats.append(self._alpha_conv(offset_k1, tau))               # (T, C) [v6]
            feats.append(self._alpha_conv(onset_k4, tau))                # (T, C)
            feats.append(self._alpha_conv(offset_k4, tau))               # (T, C) [v6]
            feats.append(self._alpha_conv(abs_k16, tau))                 # (T, C) [v6] long-range
            feats.append(self._alpha_conv(surprise_scalar_k1, tau))      # (T, 1)
            feats.append(self._alpha_conv(offset_scalar_k1, tau))        # (T, 1) [v6]
        # Alpha-IRF on log-cochleagram itself ("HG-smeared" energy)
        for tau in self.alpha_taus:
            feats.append(self._alpha_conv(log_e, tau))                   # (T, C)
        # [v6] Gamma-IRF (n=2, sharper peak) on key surprise signals — different
        # temporal weighting than alpha, gives ridge a complementary basis.
        for tau in self.gamma_taus:
            feats.append(self._gamma_conv(onset_k1, tau))                # (T, C)
            feats.append(self._gamma_conv(log_e, tau))                   # (T, C)
        # [v6] IRF-convolved log of phase-residual L1 — phase surprise integrated
        for tau in self.alpha_taus[:3]:
            feats.append(self._alpha_conv(log_phase_l1, tau))            # (T, C)

        # ---- [v3] LONG-HORIZON HOLD-PREDICTOR RESIDUALS ----
        for k in self.LONG_HORIZONS:
            xk = past_shift(log_e, k)
            r = log_e - xk
            feats.append(r)                                              # signed (out of FIR reach)
            feats.append(r.clamp(min=0.0))
            feats.append((-r).clamp(min=0.0))

        # ---- [v3] INFORMATION-THEORETIC SPECTRAL SURPRISE ----
        # Treat each frame's cochleagram as a categorical distribution over
        # bands. Entropy, cross-entropy and KL divergences are NONLINEAR
        # scalars no linear remix of the band log-energies can synthesize.
        p_t = torch.softmax(log_e, dim=1)                                # (T, C)
        log_p_t = torch.log_softmax(log_e, dim=1)                        # (T, C)
        # Spectral entropy H = -sum p log p
        H = -(p_t * log_p_t).sum(dim=1, keepdim=True)
        feats.append(H)
        # KL(p_t || p_{t-k}) — spectral distribution surprise vs k frames ago
        for k in self.KL_HORIZONS:
            log_p_k = past_shift(log_p_t, k)
            kl = (p_t * (log_p_t - log_p_k)).sum(dim=1, keepdim=True)
            feats.append(kl)
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

        # [v6] dropped wave-level prediction cochleagrams (v2 noop)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "PC-v6"
model_description = (
    "PC-v5 (test_corr=0.0624, dim=12116) + EXPANDED IRF BLOCK: alpha-IRF on "
    "offsets (-Δ k=1,4), long-range deviance |Δ k=16|, and offset scalar; "
    "gamma-IRF (n=2 shape, tau=2/6/15) on onset-k1 and log_e — different "
    "temporal weighting than alpha; alpha-IRF on log(phase-residual L1) — "
    "phase surprise integrated. Raw dim ≈ 3610."
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
