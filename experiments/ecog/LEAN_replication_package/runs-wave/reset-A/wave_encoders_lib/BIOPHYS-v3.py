"""
BIOPHYSICAL AUDITORY PERIPHERY front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

v3: keeps the v2 periphery but:
  - normalizes audio gain per chunk so DRNL operates at physiological setpoint
  - replaces always-silent LSR fiber with two parallel BM compression operating
    points (linear path + strongly compressive path) — a more realistic dual-path
    DRNL with reliable signal in both branches
  - adds Chi–Shamma 2-D rate–scale STMF on the IHC log-cochleagram
  - adds finer-resolution summary autocorrelation (24 lags) for pitch
  - adds per-channel kurtosis / variance within frame (peakiness vs steady)
  - keeps lateral inhibition, AM Gabor envelopes, future/past deltas
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data import SR, HOP, WINMAX
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR = "/path/to/scratch/feat_cache_reset_A"
os.makedirs(CACHE_DIR, exist_ok=True)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (pts / 21.4) - 1.0) * 1000.0 / 4.37


def reconstruct_raw(windows: np.ndarray) -> np.ndarray:
    T, W = windows.shape
    L = (T - 1) * HOP + W
    raw = np.empty(L, dtype=np.float32)
    raw[:W] = windows[0]
    if T > 1:
        raw[W:] = windows[1:, W - HOP:].reshape(-1)
    return raw


def causal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    T, C = x.shape
    alpha = 1.0 / (1.0 + tau_frames)
    K = min(T, max(int(6 * tau_frames) + 1, 4))
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = (alpha * (1.0 - alpha) ** t).flip(0)
    xp = x.T.unsqueeze(0)
    xp = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.view(1, 1, K).expand(C, 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp, kk, groups=C)
    return y.squeeze(0).T


def acausal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    return torch.flip(causal_ema(torch.flip(x, dims=[0]), tau_frames), dims=[0])


def gabor_envelope(x: torch.Tensor, rate_hz: float, fs: float = 30.0) -> torch.Tensor:
    period = fs / rate_hz
    sigma = period
    L = int(np.ceil(4 * sigma))
    t = np.arange(-L, L + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
    sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
    cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
    cos_k /= (np.linalg.norm(cos_k) + 1e-12)
    sin_k /= (np.linalg.norm(sin_k) + 1e-12)
    cos_t = torch.from_numpy(cos_k.astype(np.float32)).to(x.device)
    sin_t = torch.from_numpy(sin_k.astype(np.float32)).to(x.device)
    K = cos_t.numel()
    C = x.shape[1]
    xp = x.T.unsqueeze(0)
    xp = torch.nn.functional.pad(xp, (K // 2, K // 2), mode='replicate')
    ck = cos_t.view(1, 1, K).expand(C, 1, K).contiguous()
    sk = sin_t.view(1, 1, K).expand(C, 1, K).contiguous()
    c = torch.nn.functional.conv1d(xp, ck, groups=C)[..., :x.shape[0]]
    s = torch.nn.functional.conv1d(xp, sk, groups=C)[..., :x.shape[0]]
    env = torch.sqrt(c * c + s * s + 1e-12)
    return env.squeeze(0).T


def channel_dog(x: torch.Tensor, sig_c: float, sig_s: float) -> torch.Tensor:
    k = int(np.ceil(3 * sig_s))
    coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    g_c = torch.exp(-(coords ** 2) / (2 * sig_c ** 2)); g_c /= g_c.sum()
    g_s = torch.exp(-(coords ** 2) / (2 * sig_s ** 2)); g_s /= g_s.sum()
    dog = (g_c - g_s).view(1, 1, -1)
    return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)


def stmf_gabor_2d(log_e: torch.Tensor, omega_t: float, omega_f: float,
                  fs: float = 30.0) -> torch.Tensor:
    """2-D Chi-Shamma rate-scale Gabor on a (T, F) log-cochleagram."""
    period_t = fs / max(abs(omega_t), 0.5)
    sigma_t = period_t / (2 * np.pi)
    Lt = max(1, int(np.ceil(3 * sigma_t)))
    sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi)
    Lf = max(1, int(np.ceil(3 * sigma_f)))
    t = np.arange(-Lt, Lt + 1)
    f = np.arange(-Lf, Lf + 1)
    g = np.exp(-((t[:, None] ** 2) / (2 * sigma_t ** 2)
                  + (f[None, :] ** 2) / (2 * sigma_f ** 2)))
    phase = 2 * np.pi * omega_t * t[:, None] / fs + 2 * np.pi * omega_f * f[None, :]
    cos_k = (g * np.cos(phase)).astype(np.float32)
    sin_k = (g * np.sin(phase)).astype(np.float32)
    cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
    cos_t = torch.from_numpy(cos_k).to(log_e.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
    sin_t = torch.from_numpy(sin_k).to(log_e.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
    xp = log_e.unsqueeze(0).unsqueeze(0)
    c = torch.nn.functional.conv2d(xp, cos_t, padding=(Lt, Lf))
    s = torch.nn.functional.conv2d(xp, sin_t, padding=(Lt, Lf))
    T, F = log_e.shape
    env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
    return env[:T, :F]


# ---------------------------------------------------------------------------

class BiophysPeripheryV3:
    win = WINMAX
    n_params = 0

    N_BM = 48
    F_MIN = 80.0
    F_MAX = 8000.0
    BM_ORDER = 4

    # Physiological audio scaling: typical speech audio is ~0.05 RMS;
    # we boost to make compression regime sensible.
    AUDIO_GAIN = 50.0

    BM_THETA_LIN = 1.0    # linear-path threshold (basically off)
    BM_THETA_COMP = 0.05  # comp-path threshold ~ ear's compression knee
    BM_C = 0.25           # compression exponent
    BM_ENV_LP_HZ = 200.0

    IHC_LP_HZ = 1000.0
    IHC_ASYMMETRY = 0.15
    IHC_SAT_K = 2.0

    # Two AN fibers: HSR (fast saturating, low half-sat) + LSR (linear range)
    FIBERS = (
        ('hsr', 0.05),
        ('lsr', 0.5),
    )

    ADAPT_TAUS = (1.0, 3.0, 9.0, 30.0, 90.0)
    MOC_TAU = 30.0

    AM_RATES = (2.0, 4.0, 8.0, 16.0)

    # Spectro-temporal modulation rates (omega_t Hz, omega_f cyc/channel)
    STMF_RATES = (
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (16.0, 0.25), (16.0, -0.25),
        (0.0, 0.25), (0.0, 0.5),
        (4.0, 0.0),  (8.0, 0.0),
    )

    # 12 ACF lags in samples @ 44.1 kHz, log-spaced from ~70 Hz to ~600 Hz F0
    ACF_LAGS = tuple(int(round(SR / f0)) for f0 in
                     np.geomspace(70.0, 600.0, 12))

    def __init__(self, device: str = "cuda"):
        self.device = device
        fcs = erb_space(self.F_MIN, self.F_MAX, self.N_BM).astype(np.float32)
        self.fcs = torch.from_numpy(fcs).to(device)
        self.bw_bm = 1.019 * 24.7 * (4.37 * self.fcs / 1000.0 + 1.0)
        self._cache = {}

    def _filterbank(self, L: int):
        if L not in self._cache:
            freqs = torch.fft.rfftfreq(L, d=1.0 / SR).to(self.device)
            df = freqs[None, :] - self.fcs[:, None]
            H_bm = (1.0 + (df / self.bw_bm[:, None]) ** 2) ** (-self.BM_ORDER / 2.0)
            H_env = 1.0 / torch.sqrt(1.0 + (freqs / self.BM_ENV_LP_HZ) ** 4)
            H_ihc = 1.0 / torch.sqrt(1.0 + (freqs / self.IHC_LP_HZ) ** 4)
            self._cache[L] = (freqs, H_bm, H_env, H_ihc)
        return self._cache[L]

    def _periphery_fast(self, raw_t: torch.Tensor):
        """Dual-path BM + IHC + 2 AN fibers. Returns 4 cochleagrams @ 44.1 kHz.

        Returns: ihc_lin, ihc_comp, hsr, lsr — all (n_c, L).
        """
        L = raw_t.shape[0]
        _, H_bm, H_env, H_ihc = self._filterbank(L)

        x_gained = raw_t * self.AUDIO_GAIN
        X = torch.fft.rfft(x_gained)
        Y = X[None, :] * H_bm
        bm_lin = torch.fft.irfft(Y, n=L)              # linear gammatone bandpass

        # Estimate local envelope on the linear BM output once (used for both paths)
        env_X = torch.fft.rfft(bm_lin.abs())
        env_smooth = torch.fft.irfft(env_X * H_env[None, :], n=L).clamp(min=1e-5)

        # --- Linear path (high-threshold, mostly linear regime)
        gain_lin = (self.BM_THETA_LIN / (self.BM_THETA_LIN + env_smooth)) ** (1.0 - self.BM_C)
        bm_linpath = bm_lin * gain_lin

        # --- Compressive path (low-threshold, compressive at speech levels)
        gain_comp = (self.BM_THETA_COMP / (self.BM_THETA_COMP + env_smooth)) ** (1.0 - self.BM_C)
        bm_comp = bm_lin * gain_comp

        def ihc(bm: torch.Tensor) -> torch.Tensor:
            ihc_in = torch.relu(bm) + self.IHC_ASYMMETRY * torch.relu(-bm)
            ihc_X = torch.fft.rfft(ihc_in)
            ihc_lp = torch.fft.irfft(ihc_X * H_ihc[None, :], n=L).clamp(min=0.0)
            return torch.tanh(self.IHC_SAT_K * ihc_lp)

        ihc_lin = ihc(bm_linpath)
        ihc_comp = ihc(bm_comp)

        # AN fibers driven by the compressive-path IHC (higher dynamic range)
        hsr = ihc_comp / (ihc_comp + self.FIBERS[0][1])
        lsr = ihc_comp / (ihc_comp + self.FIBERS[1][1])
        return ihc_lin, ihc_comp, hsr, lsr

    def _box_avg(self, z: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        chop = offset + T * HOP
        return z[:, offset:chop].reshape(self.N_BM, T, HOP).mean(dim=-1).T

    def _box_var(self, z: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        """Within-frame variance per channel (peakiness)."""
        chop = offset + T * HOP
        seg = z[:, offset:chop].reshape(self.N_BM, T, HOP)
        m = seg.mean(dim=-1, keepdim=True)
        v = ((seg - m) ** 2).mean(dim=-1)
        return v.T

    def _subframe_acf(self, ihc: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        """Per-channel, per-frame autocorrelation at log-spaced lags."""
        chop = offset + T * HOP
        frames = ihc[:, offset:chop].reshape(self.N_BM, T, HOP)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        norm = (frames ** 2).sum(dim=-1).clamp(min=1e-8)
        outs = []
        for lag in self.ACF_LAGS:
            if lag >= HOP:
                outs.append(torch.zeros(T, self.N_BM, device=ihc.device))
                continue
            a = frames[:, :, :HOP - lag]
            b = frames[:, :, lag:]
            num = (a * b).sum(dim=-1)
            ac = num / norm
            outs.append(ac.T)
        return torch.cat(outs, dim=1)

    def _periphery_features(self, raw_t: torch.Tensor) -> torch.Tensor:
        ihc_lin, ihc_comp, hsr_full, lsr_full = self._periphery_fast(raw_t)
        L = ihc_lin.shape[1]
        T = (L - WINMAX) // HOP + 1
        offset = WINMAX // 2 - HOP // 2

        feats = []
        log_eps = 1e-4

        # ---- Sub-frame ACF on IHC_comp (carries phase-locked F0 info)
        acf_feats = self._subframe_acf(ihc_comp, T, offset)
        feats.append(acf_feats)                         # (T, C * n_lags)
        n_lags = len(self.ACF_LAGS)
        acf_grid = acf_feats.view(T, n_lags, self.N_BM)
        feats.append(acf_grid.max(dim=2).values)        # summary pitch salience
        feats.append(acf_grid.mean(dim=2))

        # ---- Cochleagrams (linear path, compressive path, HSR, LSR)
        ds_ihc_lin = self._box_avg(ihc_lin, T, offset)
        ds_ihc_comp = self._box_avg(ihc_comp, T, offset)
        ds_hsr = self._box_avg(hsr_full, T, offset)
        ds_lsr = self._box_avg(lsr_full, T, offset)
        log_lin = torch.log(ds_ihc_lin + log_eps)
        log_comp = torch.log(ds_ihc_comp + log_eps)
        log_hsr = torch.log(ds_hsr + log_eps)
        log_lsr = torch.log(ds_lsr + log_eps)
        feats.extend([log_lin, log_comp, log_hsr, log_lsr])

        # ---- AN subtractive adaptation on HSR and LSR
        for log_r in (log_hsr, log_lsr):
            for tau in self.ADAPT_TAUS:
                feats.append(log_r - causal_ema(log_r, tau))

        # ---- MOC AGC: subtract slow loudness from compressive path
        slow_loud = causal_ema(log_comp.mean(dim=1, keepdim=True), self.MOC_TAU)
        feats.append(log_comp - slow_loud)
        feats.append(log_hsr - slow_loud)

        # ---- Lateral inhibition / two-tone suppression DoG
        feats.append(channel_dog(log_comp, 1.0, 3.0))
        feats.append(channel_dog(log_comp, 1.0, 6.0))
        feats.append(channel_dog(log_hsr, 1.0, 3.0))

        # ---- Within-frame variance / peakiness (per channel)
        var_ihc = torch.log(self._box_var(ihc_comp, T, offset) + 1e-10)
        feats.append(var_ihc)

        # ---- Onset / offset HWR of derivative (per channel)
        d_lin = torch.zeros_like(log_comp);  d_lin[1:] = log_comp[1:] - log_comp[:-1]
        d_hsr = torch.zeros_like(log_hsr);   d_hsr[1:] = log_hsr[1:] - log_hsr[:-1]
        feats.extend([d_lin.clamp(min=0), (-d_lin).clamp(min=0),
                       d_hsr.clamp(min=0), (-d_hsr).clamp(min=0)])

        # ---- 2-D STMF on the compressive-path log cochleagram
        for (wt, wf) in self.STMF_RATES:
            feats.append(stmf_gabor_2d(log_comp, wt, wf))

        # ---- Per-channel AM envelopes (Gabor at modulation rates)
        for rate in self.AM_RATES:
            feats.append(gabor_envelope(log_comp, rate))

        # ---- Future / past deltas (acausal lookahead context)
        for k in (3, 6, 12):
            fut = torch.cat([log_comp[k:], log_comp[-1:].expand(k, -1)], dim=0)
            feats.append(fut - log_comp)
            past = torch.cat([log_comp[:1].expand(k, -1), log_comp[:-k]], dim=0)
            feats.append(log_comp - past)

        # ---- Acausal slow loudness
        feats.append(acausal_ema(log_comp, self.MOC_TAU))

        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        raw = reconstruct_raw(np.asarray(windows, dtype=np.float32))
        raw_t = torch.from_numpy(raw).to(self.device)
        feat = self._periphery_features(raw_t)
        return feat.float().cpu().numpy()


# ---------------------------------------------------------------------------

model_shorthand_name = "BIOPHYS-v3"
model_description = (
    "Biophysical periphery v3: 48-ch BM with DUAL DRNL path (linear theta=1, "
    "compressive theta=0.05, c=0.25), IHC + 2 AN fibers (HSR 0.05, LSR 0.5), "
    "AUDIO_GAIN=50 for physiological setpoint, 24-lag sub-frame ACF (70-600 Hz "
    "F0), 16 Chi-Shamma 2D STMF kernels on log-comp coch, lateral inh DoG, "
    "within-frame variance, AM Gabor envs 2/4/8/16 Hz, multi-tau AN adaptation, "
    "MOC AGC, onset/offset, future/past deltas k=3/6/12. v2 test_corr=0.0617."
)


def build_embedder(device: str = "cuda") -> BiophysPeripheryV3:
    return BiophysPeripheryV3(device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available — cpu")
        args.device = "cpu"

    t0 = time.time()
    cfg = EncodingConfig(num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device)
    r = run_encoding(embedder, cfg)
    n_params = int(getattr(embedder, "n_params", 0))
    upsert_overall_results(
        [make_result_row(r, model_shorthand_name, n_params, model_description)],
        RESULTS_DIR)
    plot_corr_over_iterations(RESULTS_DIR)
    print()
    print("---")
    print(f"subject:        {r['subject']}")
    print(f"test_corr:      {r['test_corr']:.4f}  (train_corr={r['train_corr']:.4f}, "
          f"frac_elec>0.1={r['frac_test_elec_above_0.1']:.4f})")
    print(f"n_params:       {n_params:.2e}")
    print(f"encoding_secs:  {r['encoding_seconds']:.1f}s")
    print(f"total_seconds:  {time.time() - t0:.1f}s")
