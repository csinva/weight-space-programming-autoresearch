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
    return torch.cat([x[:1].expand(k, -1), x[:-k]], dim=0)


# ---------------------------------------------------------------------------
# Predictive-coding embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0
    N_FILTERS = 32
    HORIZONS = (1, 2, 4, 8, 16)                # AR prediction horizons (frames)
    PRECISION_TAUS = (15.0, 60.0)              # for log(sigma)
    Z_TAUS = (30.0,)                           # for precision-weighting residuals
    SPATIAL_SIGMAS = (1.0, 3.0)                # channel-axis context smoothing

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
        p = power_spectrum(x, self.win, self.device)               # (T, F_fft)
        e = p @ self.fb                                             # (T, C)
        log_e = torch.log(e.clamp(min=1e-10))                       # (T, C)
        feats.append(log_e)

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

        # ---- SPATIAL (CHANNEL-AXIS) CONTEXT RESIDUALS ----
        # Each band's deviance from a Gaussian-weighted mean of its spectral
        # neighbors. Rectified + squared = nonlinear.
        for sigma in self.SPATIAL_SIGMAS:
            ctx = self._spatial_context(log_e, sigma)
            sp_r = log_e - ctx
            feats.append(sp_r.clamp(min=0.0))
            feats.append((-sp_r).clamp(min=0.0))
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

        # ---- [v2] WAVEFORM-LEVEL PREDICTION-ERROR COCHLEAGRAMS ----
        # Fixed polynomial-extrapolator residuals at order p in {1,2,4} applied
        # to the raw waveform. Implemented in the frequency domain via
        # |H_p(f)|^2 multiplied into the power spectrum BEFORE the gammatone
        # filterbank — captures sub-frame transients (plosive bursts, attack
        # onsets) that the frame-rate cochleagram cannot resolve.
        for order, h2p in self.wave_pred_responses.items():
            pow_res = p * h2p.unsqueeze(0)                              # broadcast over T
            e_res = pow_res @ self.fb                                   # (T, C)
            log_e_res = torch.log(e_res.clamp(min=1e-10))
            feats.append(log_e_res)
            feats.append(log_e_res - log_e)                             # "extra surprise" per band
            feats.append(torch.log(pow_res.sum(dim=1, keepdim=True).clamp(min=1e-10)))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "PC-v2"
model_description = (
    "PC-v1 (test_corr=0.0597, dim=10596) + WAVEFORM-LEVEL prediction-error "
    "cochleagrams: fixed polynomial-extrapolator residuals at order p in "
    "{1,2,4} applied to the waveform in the FREQ domain (|2 sin(ω/2)|^(2p) "
    "multiplied into power spec) before the ERB filterbank — captures "
    "sub-frame plosive bursts and attack onsets the 33 ms frame rate smears."
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
