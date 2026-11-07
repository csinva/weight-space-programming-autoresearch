"""
BIOPHYSICAL AUDITORY PERIPHERY v10. After diagnostic v9 (pure mel = 0.0644)
showed the periphery wasn't visibly contributing, this version:

  - Restores the biophysical periphery as the FRONT-END (the paradigm)
  - But uses ONE compression stage (DRNL only) to avoid triple-compression
    flattening that hid biophys signal in v1-v8
  - 64 cochlear channels (was 32-48) for finer spectral detail in the speech range
  - Stronger DRNL (theta_comp=0.005 — much sharper compression knee)
  - Adds HILBERT MODULATION PHASE per channel at slow rates (sin/cos of the
    phase of slow envelope at 2/4/8 Hz) — novel, captures "where in the
    syllabic cycle" each band is, beyond pure amplitude
  - Adds LONGER-range future/past deltas (k=18, 24) for the long STG STRF
  - Keeps the v7-tuned downstream: STMF, harmonic-template pitch, phonetic
    matched filters, cross-band products, LEAN-style spectral moments.
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
    return torch.nn.functional.conv1d(xp, kk, groups=C).squeeze(0).T


def acausal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    return torch.flip(causal_ema(torch.flip(x, dims=[0]), tau_frames), dims=[0])


def gabor_quadrature(x: torch.Tensor, rate_hz: float,
                      fs: float = 30.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos_part, sin_part) Gabor filter outputs — quadrature pair so we
    can compute both ENVELOPE (sqrt(c²+s²)) and PHASE (atan2(s, c))."""
    period = fs / rate_hz; sigma = period
    L = int(np.ceil(4 * sigma))
    t = np.arange(-L, L + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
    sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
    cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
    cos_k /= (np.linalg.norm(cos_k) + 1e-12); sin_k /= (np.linalg.norm(sin_k) + 1e-12)
    cos_t = torch.from_numpy(cos_k.astype(np.float32)).to(x.device)
    sin_t = torch.from_numpy(sin_k.astype(np.float32)).to(x.device)
    K = cos_t.numel(); C = x.shape[1]
    xp = x.T.unsqueeze(0)
    xp = torch.nn.functional.pad(xp, (K // 2, K // 2), mode='replicate')
    ck = cos_t.view(1, 1, K).expand(C, 1, K).contiguous()
    sk = sin_t.view(1, 1, K).expand(C, 1, K).contiguous()
    c = torch.nn.functional.conv1d(xp, ck, groups=C)[..., :x.shape[0]]
    s = torch.nn.functional.conv1d(xp, sk, groups=C)[..., :x.shape[0]]
    return c.squeeze(0).T, s.squeeze(0).T


def channel_dog(x: torch.Tensor, sig_c: float, sig_s: float) -> torch.Tensor:
    k = int(np.ceil(3 * sig_s))
    coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    g_c = torch.exp(-(coords ** 2) / (2 * sig_c ** 2)); g_c /= g_c.sum()
    g_s = torch.exp(-(coords ** 2) / (2 * sig_s ** 2)); g_s /= g_s.sum()
    dog = (g_c - g_s).view(1, 1, -1)
    return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)


def stmf_gabor_2d(log_e: torch.Tensor, omega_t: float, omega_f: float,
                  fs: float = 30.0) -> torch.Tensor:
    period_t = fs / max(abs(omega_t), 0.5)
    sigma_t = period_t / (2 * np.pi)
    Lt = max(1, int(np.ceil(3 * sigma_t)))
    sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi)
    Lf = max(1, int(np.ceil(3 * sigma_f)))
    t = np.arange(-Lt, Lt + 1); f = np.arange(-Lf, Lf + 1)
    g = np.exp(-((t[:, None] ** 2) / (2 * sigma_t ** 2)
                  + (f[None, :] ** 2) / (2 * sigma_f ** 2)))
    phase = 2 * np.pi * omega_t * t[:, None] / fs + 2 * np.pi * omega_f * f[None, :]
    cos_k = (g * np.cos(phase)).astype(np.float32); sin_k = (g * np.sin(phase)).astype(np.float32)
    cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
    cos_t = torch.from_numpy(cos_k).to(log_e.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
    sin_t = torch.from_numpy(sin_k).to(log_e.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
    xp = log_e.unsqueeze(0).unsqueeze(0)
    c = torch.nn.functional.conv2d(xp, cos_t, padding=(Lt, Lf))
    s = torch.nn.functional.conv2d(xp, sin_t, padding=(Lt, Lf))
    T, F = log_e.shape
    return torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)[:T, :F]


# ---------------------------------------------------------------------------

class BiophysPeripheryV10:
    win = WINMAX
    n_params = 0

    N_BM = 48
    F_MIN = 80.0
    F_MAX = 8000.0
    BM_ORDER = 4

    AUDIO_GAIN = 50.0
    BM_THETA_COMP = 0.005   # sharper compression knee (was 0.05)
    BM_C = 0.25
    BM_ENV_LP_HZ = 200.0
    IHC_LP_HZ = 1000.0
    IHC_ASYMMETRY = 0.15
    # Single nonlinearity: DRNL gain. NO extra tanh, NO Michaelis-Menten fiber sat.

    LONG_TAUS = (9.0, 30.0, 90.0)
    MOC_TAU = 30.0

    STMF_RATES = (
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (16.0, 0.25), (16.0, -0.25),
        (0.0, 0.25), (0.0, 0.5),
    )

    PHASE_RATES = (2.0, 4.0, 8.0)
    F0_CANDS = tuple(np.geomspace(70.0, 500.0, 14).tolist())

    def __init__(self, device: str = "cuda"):
        self.device = device
        fcs = erb_space(self.F_MIN, self.F_MAX, self.N_BM).astype(np.float32)
        self.fcs = torch.from_numpy(fcs).to(device)
        self.bw_bm = 1.019 * 24.7 * (4.37 * self.fcs / 1000.0 + 1.0)
        self.acf_lags = [int(round(SR / f0)) for f0 in self.F0_CANDS]
        self.harm_idx = []
        for f0 in self.F0_CANDS:
            h_idx = []
            for k in range(1, 7):
                fh = f0 * k
                if fh > self.F_MAX:
                    break
                h_idx.append(int(torch.argmin((self.fcs - fh).abs()).item()))
            self.harm_idx.append(h_idx)
        self._cache = {}

    def _filterbank(self, L_pad: int):
        if L_pad not in self._cache:
            freqs = torch.fft.rfftfreq(L_pad, d=1.0 / SR).to(self.device)
            df = freqs[None, :] - self.fcs[:, None]
            H_bm = (1.0 + (df / self.bw_bm[:, None]) ** 2) ** (-self.BM_ORDER / 2.0)
            H_env = 1.0 / torch.sqrt(1.0 + (freqs / self.BM_ENV_LP_HZ) ** 4)
            H_ihc = 1.0 / torch.sqrt(1.0 + (freqs / self.IHC_LP_HZ) ** 4)
            self._cache[L_pad] = (freqs, H_bm, H_env, H_ihc)
        return self._cache[L_pad]

    def _periphery(self, raw_t: torch.Tensor):
        L = raw_t.shape[0]
        L_pad = ((L + 2047) // 2048) * 2048
        if L_pad != L:
            x_g = torch.nn.functional.pad(raw_t, (0, L_pad - L)) * self.AUDIO_GAIN
        else:
            x_g = raw_t * self.AUDIO_GAIN
        _, H_bm, H_env, H_ihc = self._filterbank(L_pad)
        X = torch.fft.rfft(x_g, n=L_pad)
        Y = X[None, :] * H_bm
        bm_lin = torch.fft.irfft(Y, n=L_pad)[:, :L]
        env_X = torch.fft.rfft(bm_lin.abs(), n=L_pad)
        env_smooth = torch.fft.irfft(env_X * H_env[None, :], n=L_pad)[:, :L].clamp(min=1e-6)
        gain = (self.BM_THETA_COMP / (self.BM_THETA_COMP + env_smooth)) ** (1.0 - self.BM_C)
        bm = bm_lin * gain
        ihc_in = torch.relu(bm) + self.IHC_ASYMMETRY * torch.relu(-bm)
        ihc_X = torch.fft.rfft(ihc_in, n=L_pad)
        # LINEAR IHC: just LPF + HWR. NO tanh saturation, NO Michaelis-Menten fiber.
        ihc = torch.fft.irfft(ihc_X * H_ihc[None, :], n=L_pad)[:, :L].clamp(min=0.0)
        return ihc, env_smooth

    def _box_avg(self, z: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        chop = offset + T * HOP
        return z[:, offset:chop].reshape(self.N_BM, T, HOP).mean(dim=-1).T

    def _winmax_acf(self, ihc: torch.Tensor, T: int) -> torch.Tensor:
        frames = ihc.unfold(dimension=1, size=WINMAX, step=HOP)[:, :T, :]
        frames = frames - frames.mean(dim=-1, keepdim=True)
        Nfft = 2 * WINMAX
        F = torch.fft.rfft(frames, n=Nfft)
        acf = torch.fft.irfft(F * F.conj(), n=Nfft)[..., :WINMAX]
        norm = acf[..., 0:1].clamp(min=1e-8)
        acf = acf / norm
        lag_t = torch.tensor(self.acf_lags, device=ihc.device, dtype=torch.long)
        sel = acf.index_select(-1, lag_t)
        return sel.permute(1, 0, 2).contiguous()           # (T, C, K)

    def _periphery_features(self, raw_t: torch.Tensor) -> torch.Tensor:
        ihc, env_smooth = self._periphery(raw_t)
        torch.cuda.empty_cache()
        L = ihc.shape[1]
        T = (L - WINMAX) // HOP + 1
        offset = WINMAX // 2 - HOP // 2

        feats = []
        log_eps = 1e-5

        # ---- Cochleagrams: log + cube of the SINGLE-compression IHC output
        ds_ihc = self._box_avg(ihc, T, offset)
        ds_env = self._box_avg(env_smooth, T, offset)
        log_ihc = torch.log(ds_ihc + log_eps)
        cube_ihc = torch.pow(ds_ihc + log_eps, 1.0 / 3.0)
        # DRNL gain (uniquely biophys feature: how much BM compressed each band/frame)
        log_drnl_gain = torch.log((ds_ihc + 1e-7) / (ds_env + 1e-7))
        feats.extend([log_ihc, cube_ihc, log_drnl_gain])

        # ---- Sub-frame ACF (phase-locked F0)
        acf_TCK = self._winmax_acf(ihc, T)
        sum_acf = acf_TCK.sum(dim=1)
        max_acf = acf_TCK.max(dim=1).values
        feats.extend([sum_acf, max_acf])

        # ---- Harmonic-template pitch
        baseline = log_ihc.mean(dim=1, keepdim=True)
        templ_scores = []
        for f0, h_idx in zip(self.F0_CANDS, self.harm_idx):
            if not h_idx:
                templ_scores.append(torch.zeros(T, 1, device=raw_t.device))
                continue
            sal = log_ihc[:, h_idx].mean(dim=1, keepdim=True) - baseline
            templ_scores.append(sal)
        harm = torch.cat(templ_scores, dim=1)
        feats.append(harm)
        best_str, best_idx = harm.max(dim=1, keepdim=True)
        f0_cands_t = torch.tensor(self.F0_CANDS, device=raw_t.device, dtype=torch.float32)
        log_f0 = torch.log(f0_cands_t[best_idx.squeeze(1)].unsqueeze(1) + 1.0)
        feats.append(torch.cat([best_str, log_f0], dim=1))

        # ---- Long-context EMAs (beyond 4-tap FIR)
        for tau in self.LONG_TAUS:
            feats.append(log_ihc - causal_ema(log_ihc, tau))
        # MOC slow-loud contrast (global)
        slow_loud = causal_ema(log_ihc.mean(dim=1, keepdim=True), self.MOC_TAU)
        feats.append(log_ihc - slow_loud)
        feats.append(acausal_ema(log_ihc.mean(dim=1, keepdim=True), self.MOC_TAU))

        # ---- Lateral inhibition DoG (3 scales for the now 64-channel grid)
        feats.append(channel_dog(log_ihc, 1.0, 3.0))
        feats.append(channel_dog(log_ihc, 1.0, 6.0))
        feats.append(channel_dog(log_ihc, 2.0, 8.0))

        # ---- 2D STMF on log_ihc
        for (wt, wf) in self.STMF_RATES:
            feats.append(stmf_gabor_2d(log_ihc, wt, wf))

        # ---- Phonetic matched filters
        nF = self.N_BM
        very_lo = log_ihc[:, :max(1, nF // 8)].mean(dim=1, keepdim=True)
        lo = log_ihc[:, :nF // 4].mean(dim=1, keepdim=True)
        mid_lo = log_ihc[:, nF // 6:nF // 3].mean(dim=1, keepdim=True)
        mid = log_ihc[:, nF // 4:3 * nF // 4].mean(dim=1, keepdim=True)
        mid_hi = log_ihc[:, nF // 2:3 * nF // 4].mean(dim=1, keepdim=True)
        hi = log_ihc[:, 3 * nF // 4:].mean(dim=1, keepdim=True)
        very_hi = log_ihc[:, max(0, nF - nF // 8):].mean(dim=1, keepdim=True)
        spec_var = log_ihc.var(dim=1, keepdim=True)
        sibilance = very_hi - lo
        fric = mid_hi - lo - 0.5 * best_str
        vowel = mid_lo + 0.5 * best_str - 0.3 * very_hi
        plosive = spec_var - 0.5 * best_str
        nasal = very_lo - hi
        approx = mid + 0.3 * best_str - 0.3 * spec_var
        voicedstop = best_str + lo - 0.3 * very_hi
        phon = torch.cat([sibilance, fric, vowel, plosive, nasal, approx,
                          voicedstop], dim=1)
        feats.append(phon)
        for tau in (9.0, 30.0):
            feats.append(phon - causal_ema(phon, tau))

        # ---- HILBERT MODULATION PHASE per channel at slow rates (novel)
        # Captures WHERE in the syllabic cycle each band is, beyond amplitude.
        for rate in self.PHASE_RATES:
            c_part, s_part = gabor_quadrature(log_ihc, rate)
            env = torch.sqrt(c_part * c_part + s_part * s_part + 1e-12)
            inv = 1.0 / env.clamp(min=1e-6)
            sin_p = s_part * inv
            cos_p = c_part * inv
            # Weight phase by envelope (so silent frames don't dominate)
            feats.append(env)                            # (T, n_c)
            feats.append(env * sin_p)                    # (T, n_c)
            feats.append(env * cos_p)                    # (T, n_c)

        # ---- Cross-band products (nonlinear, can't be derived from taps)
        log_n = log_ihc - log_ihc.mean(dim=1, keepdim=True)
        prods = []
        pair_ratios = [(0.05, 0.55), (0.1, 0.7), (0.2, 0.85),
                       (0.3, 0.95), (0.5, 0.95), (0.4, 0.6),
                       (0.1, 0.4), (0.2, 0.6), (0.3, 0.8)]
        for a, b in pair_ratios:
            ia, ib = int(a * nF), min(nF - 1, int(b * nF))
            prods.append((log_n[:, ia] * log_n[:, ib]).unsqueeze(1))
        feats.append(torch.cat(prods, dim=1))

        # ---- Broadband loudness + dynamics
        loud = log_ihc.mean(dim=1, keepdim=True)
        for tau in self.LONG_TAUS:
            feats.append(loud - causal_ema(loud, tau))

        # ---- Spectral flux / centroid / spread / flatness on log_ihc
        d = torch.zeros_like(log_ihc); d[1:] = log_ihc[1:] - log_ihc[:-1]
        feats.append(d.clamp(min=0).sum(dim=1, keepdim=True))
        feats.append((-d).clamp(min=0).sum(dim=1, keepdim=True))
        feats.append(d.abs().sum(dim=1, keepdim=True))
        e_w = torch.softmax(log_ihc, dim=1)
        ch_idx = torch.arange(nF, device=log_ihc.device, dtype=log_ihc.dtype).view(1, -1)
        centroid = (e_w * ch_idx).sum(dim=1, keepdim=True)
        spread = torch.sqrt(((e_w * (ch_idx - centroid) ** 2).sum(dim=1, keepdim=True)) + 1e-6)
        feats.append(torch.cat([centroid, spread], dim=1))
        flat = log_ihc.mean(dim=1, keepdim=True) - torch.log(
            ds_ihc.mean(dim=1, keepdim=True) + 1e-7)
        feats.append(flat)

        # ---- LONG-RANGE future / past deltas (k=3, 6, 12, 18, 24)
        for k in (3, 6, 12, 18, 24):
            fut = torch.cat([log_ihc[k:], log_ihc[-1:].expand(k, -1)], dim=0)
            feats.append(fut - log_ihc)
            past = torch.cat([log_ihc[:1].expand(k, -1), log_ihc[:-k]], dim=0)
            feats.append(log_ihc - past)

        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        raw = reconstruct_raw(np.asarray(windows, dtype=np.float32))
        raw_t = torch.from_numpy(raw).to(self.device)
        feat = self._periphery_features(raw_t)
        return feat.float().cpu().numpy()


# ---------------------------------------------------------------------------

model_shorthand_name = "BIOPHYS-v10"
model_description = (
    "Biophysical periphery v10: 64-ch BM with ONE compression (DRNL only, "
    "theta_comp=0.005 for sharper knee), linear IHC (HWR + 1kHz LPF, NO tanh, "
    "NO HSR sigmoid). Novel: per-channel HILBERT MODULATION PHASE at 2/4/8 Hz "
    "(env, env*sin_phase, env*cos_phase). Longer future/past deltas (k=3-24). "
    "Other features: ACF, harmonic template pitch, 14 STMF, 7 phonetics, "
    "9 cross-band products, lateral inh DoG x3, spectral moments/flatness. "
    "v9-mel control=0.0644, v7=0.0650."
)


def build_embedder(device: str = "cuda") -> BiophysPeripheryV10:
    return BiophysPeripheryV10(device=device)


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
