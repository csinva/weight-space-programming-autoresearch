"""
BIOPHYSICAL AUDITORY PERIPHERY front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

v2: keeps the periphery centerpiece (DRNL BM, IHC, 3 AN fiber pops, MOC AGC)
and adds the structure the ridge needs to read it out:
  - Sub-window AN envelope autocorrelation per channel (phase-locked F0 from periphery)
  - Lateral inhibition / two-tone suppression (DoG across channels of IHC + fibers)
  - Onset/offset HWR-of-derivative per channel
  - Per-channel temporal modulation envelopes (Gabor at 2/4/8/16 Hz)
  - Future + past deltas per channel (acausal lookahead context per band)
  - Multi-tau adaptation (5 timescales spanning ~33 ms to ~3 s).
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


# ---------------------------------------------------------------------------
# Small DSP helpers
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


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
    """Causal EMA along axis 0 of a (T, C) tensor."""
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
    """Gabor band-pass envelope across time on a (T, C) signal."""
    period = fs / rate_hz
    sigma = 1.0 * period
    L = int(np.ceil(4 * sigma))
    t = np.arange(-L, L + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
    sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
    cos_k = cos_k - cos_k.mean()
    sin_k = sin_k - sin_k.mean()
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
    """DoG across channel axis of (T, C). Lateral inhibition / two-tone suppression."""
    k = int(np.ceil(3 * sig_s))
    coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    g_c = torch.exp(-(coords ** 2) / (2 * sig_c ** 2)); g_c /= g_c.sum()
    g_s = torch.exp(-(coords ** 2) / (2 * sig_s ** 2)); g_s /= g_s.sum()
    dog = (g_c - g_s).view(1, 1, -1)
    return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)


# ---------------------------------------------------------------------------
# Biophysical periphery
# ---------------------------------------------------------------------------

class BiophysPeripheryV2:
    win = WINMAX
    n_params = 0

    N_BM = 48
    F_MIN = 80.0
    F_MAX = 8000.0
    BM_ORDER = 4

    BM_THETA = 1e-3
    BM_C = 0.25
    BM_ENV_LP_HZ = 200.0

    IHC_LP_HZ = 1000.0
    IHC_ASYMMETRY = 0.15
    IHC_SAT_K = 5.0

    FIBERS = (
        (1e-4, 0.05, 'hsr'),
        (3e-3, 0.20, 'msr'),
        (2e-2, 0.50, 'lsr'),
    )

    ADAPT_TAUS = (1.0, 3.0, 9.0, 30.0, 90.0)
    MOC_TAU = 30.0

    # Within-frame autocorrelation lags in samples @ 44.1 kHz
    # F0 candidates: 75, 100, 140, 180, 240, 320, 440 Hz → lags 588,441,315,245,184,138,100
    ACF_LAGS = (100, 138, 184, 245, 315, 441, 588)
    AM_RATES = (2.0, 4.0, 8.0, 16.0)

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
        """Return ihc (n_c, L), 3 fiber rates (n_c, L), bm_env (n_c, L)."""
        L = raw_t.shape[0]
        _, H_bm, H_env, H_ihc = self._filterbank(L)
        X = torch.fft.rfft(raw_t)
        Y = X[None, :] * H_bm
        bm_lin = torch.fft.irfft(Y, n=L)
        env_X = torch.fft.rfft(bm_lin.abs())
        env_smooth = torch.fft.irfft(env_X * H_env[None, :], n=L).clamp(min=1e-7)
        gain = (self.BM_THETA / (self.BM_THETA + env_smooth)) ** (1.0 - self.BM_C)
        bm = bm_lin * gain
        ihc_in = torch.relu(bm) + self.IHC_ASYMMETRY * torch.relu(-bm)
        ihc_X = torch.fft.rfft(ihc_in)
        ihc_lp = torch.fft.irfft(ihc_X * H_ihc[None, :], n=L).clamp(min=0.0)
        ihc = torch.tanh(self.IHC_SAT_K * ihc_lp)
        fiber_rates = []
        for theta, half_sat, _ in self.FIBERS:
            r = (ihc - theta).clamp(min=0.0)
            fiber_rates.append(r / (r + half_sat))
        return ihc, fiber_rates, bm.abs()

    def _box_avg(self, z: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        chop = offset + T * HOP
        seg = z[:, offset:chop].reshape(self.N_BM, T, HOP).mean(dim=-1)
        return seg.T

    def _subframe_acf(self, ihc: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        """Per-channel, per-frame autocorrelation at fixed lags.
        Returns (T, n_c * n_lags) normalized peak strengths."""
        L = ihc.shape[1]
        chop = offset + T * HOP
        # Reshape to (n_c, T, HOP) — within-frame samples
        frames = ihc[:, offset:chop].reshape(self.N_BM, T, HOP)  # (C, T, HOP)
        # Subtract mean per frame
        frames = frames - frames.mean(dim=-1, keepdim=True)
        norm = (frames ** 2).sum(dim=-1).clamp(min=1e-8).sqrt()  # (C, T)
        outs = []
        for lag in self.ACF_LAGS:
            if lag >= HOP:
                outs.append(torch.zeros(T, self.N_BM, device=ihc.device))
                continue
            a = frames[:, :, :HOP - lag]
            b = frames[:, :, lag:]
            num = (a * b).sum(dim=-1)
            # Normalize by approximate per-frame energy
            ac = num / (norm ** 2 + 1e-8)
            outs.append(ac.T)  # (T, C)
        return torch.cat(outs, dim=1)  # (T, C * n_lags)

    def _periphery_features(self, raw_t: torch.Tensor) -> torch.Tensor:
        ihc, fibers, bm_env = self._periphery_fast(raw_t)
        L = ihc.shape[1]
        T = (L - WINMAX) // HOP + 1
        offset = WINMAX // 2 - HOP // 2

        feats = []
        log_eps = 1e-4

        # ---- Sub-frame ACF for periodicity (centerpiece of phase-locked AN code)
        acf_feats = self._subframe_acf(ihc, T, offset)
        feats.append(acf_feats)                                # (T, C * n_lags)
        # Summary ACF: max over channels per lag (Meddis-Hewitt pitch)
        n_lags = len(self.ACF_LAGS)
        acf_grid = acf_feats.view(T, n_lags, self.N_BM)
        feats.append(acf_grid.max(dim=2).values)               # (T, n_lags)
        feats.append(acf_grid.mean(dim=2))                     # (T, n_lags)

        # ---- Downsampled per-channel rates (cochleagrams)
        log_ihc = torch.log(self._box_avg(ihc, T, offset) + log_eps)
        log_env = torch.log(self._box_avg(bm_env, T, offset) + log_eps)
        feats.append(log_ihc)
        feats.append(log_env)
        # Cube-root cochleagram (perceptual loudness scaling)
        feats.append(torch.pow(self._box_avg(bm_env, T, offset) + 1e-7, 1.0 / 3.0))

        # ---- AN fiber populations (log rate + multi-tau adaptation)
        log_fibers = []
        for r_full, (_, _, name) in zip(fibers, self.FIBERS):
            r_ds = self._box_avg(r_full, T, offset)
            log_r = torch.log(r_ds + log_eps)
            log_fibers.append(log_r)
            feats.append(log_r)
            for tau in self.ADAPT_TAUS:
                feats.append(log_r - causal_ema(log_r, tau))

        # ---- MOC AGC (slow loudness normalization)
        slow_loud = causal_ema(log_env.mean(dim=1, keepdim=True), self.MOC_TAU)
        feats.append(log_env - slow_loud)
        feats.append(log_ihc - slow_loud)

        # ---- Lateral inhibition / two-tone suppression (DoG across channels)
        feats.append(channel_dog(log_ihc, sig_c=1.0, sig_s=3.0))
        feats.append(channel_dog(log_ihc, sig_c=1.0, sig_s=6.0))
        feats.append(channel_dog(log_fibers[0], sig_c=1.0, sig_s=3.0))

        # ---- Onset / offset per channel (HWR of derivative)
        d_log = torch.zeros_like(log_env)
        d_log[1:] = log_env[1:] - log_env[:-1]
        feats.append(d_log.clamp(min=0))            # onset
        feats.append((-d_log).clamp(min=0))         # offset
        # Same on HSR
        d_hsr = torch.zeros_like(log_fibers[0])
        d_hsr[1:] = log_fibers[0][1:] - log_fibers[0][:-1]
        feats.append(d_hsr.clamp(min=0))
        feats.append((-d_hsr).clamp(min=0))

        # ---- Per-channel AM envelopes (Gabor at modulation rates)
        for rate in self.AM_RATES:
            feats.append(gabor_envelope(log_env, rate))
            feats.append(gabor_envelope(log_fibers[0], rate))

        # ---- Future/past deltas (per-band lookahead context)
        for k in (3, 6, 12):
            fut = torch.cat([log_env[k:], log_env[-1:].expand(k, -1)], dim=0)
            feats.append(fut - log_env)
            past = torch.cat([log_env[:1].expand(k, -1), log_env[:-k]], dim=0)
            feats.append(log_env - past)

        # ---- Acausal slow loudness (future loudness context)
        feats.append(acausal_ema(log_env, self.MOC_TAU))

        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        raw = reconstruct_raw(np.asarray(windows, dtype=np.float32))
        raw_t = torch.from_numpy(raw).to(self.device)
        feat = self._periphery_features(raw_t)
        return feat.float().cpu().numpy()


# ---------------------------------------------------------------------------

model_shorthand_name = "BIOPHYS-v2"
model_description = (
    "Biophysical periphery v2: 48-ch DRNL BM + IHC + 3 AN fibers + AN multi-tau "
    "adapt + MOC AGC, PLUS sub-frame per-channel envelope ACF at 7 F0 lags "
    "(100-588 samples), lateral-inhibition DoG (sig 1/3, 1/6), onset/offset HWR "
    "of d/dt log, per-band Gabor AM envelopes at 2/4/8/16 Hz, future+past deltas "
    "k=3/6/12, acausal slow loudness. v1 test_corr=0.0588."
)


def build_embedder(device: str = "cuda") -> BiophysPeripheryV2:
    return BiophysPeripheryV2(device=device)


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
