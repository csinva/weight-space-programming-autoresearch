"""
BIOPHYSICAL AUDITORY PERIPHERY v14: MINIMAL feature set, OPTIMIZED for
cross-session generalization. Test set is a held-out 450a36e4-prefix session
that was excluded from training; v7-v13 all converged on test ~0.064 with
rapidly inflating train scores (overfit). v14 strips to ~1000 features per
frame — exclusively features that should be SESSION-INVARIANT:

  - Periphery cochleagram (log + cube, the basic auditory rate code)
  - Cepstral coefficients of log_comp (DCT abstracts away spectral envelope)
  - Phonetic matched filters (high-level abstract speech features)
  - Harmonic-template pitch + RELATIVE log_F0 (median-subtracted per chunk)
  - Sub-frame ACF summary (Meddis-Hewitt) — F0 strength, session-invariant
  - 2-D STMF on log_comp (Chi-Shamma rate-scale) — physiological STRF
  - DRNL gain log (the BIOPHYS-unique signal)
  - LATERAL inh DoG (sharpens spectral edges)

The audio is RMS-normalized per chunk (the v12 bug fix) so the DRNL operates
at its designed compression knee.
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


class BiophysPeripheryV14:
    win = WINMAX
    n_params = 0

    N_BM = 32
    F_MIN = 80.0
    F_MAX = 8000.0
    BM_ORDER = 4

    AUDIO_TARGET_RMS = 0.1
    # PARALLEL DRNL operating points: strong / medium / weak compression
    # — emulates fiber populations with different SPL thresholds.
    BM_THETA_LIST = (0.002, 0.02, 0.2)
    BM_C = 0.25
    BM_ENV_LP_HZ = 200.0

    IHC_LP_HZ = 1000.0
    IHC_ASYMMETRY = 0.15
    IHC_SAT_K = 2.0
    HSR_HALF_SAT = 0.05

    LONG_TAUS = (9.0, 30.0, 90.0)
    MOC_TAU = 30.0

    STMF_RATES = (
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (16.0, 0.25), (16.0, -0.25),
    )

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
        # DCT-II basis for cepstral features
        n_ceps = 16
        k = np.arange(n_ceps)
        n = np.arange(self.N_BM)
        dct = np.cos(np.pi / self.N_BM * (n[:, None] + 0.5) * k[None, :])
        dct[:, 0] *= 1.0 / np.sqrt(2)
        dct = dct * np.sqrt(2.0 / self.N_BM)
        self.dct_basis = torch.from_numpy(dct.astype(np.float32)).to(device)
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
        rms = torch.sqrt((raw_t.float() ** 2).mean()).clamp(min=1e-6)
        scale = self.AUDIO_TARGET_RMS / rms
        L_pad = ((L + 2047) // 2048) * 2048
        if L_pad != L:
            x_g = torch.nn.functional.pad(raw_t, (0, L_pad - L)) * scale
        else:
            x_g = raw_t * scale
        _, H_bm, H_env, H_ihc = self._filterbank(L_pad)
        X = torch.fft.rfft(x_g, n=L_pad)
        Y = X[None, :] * H_bm
        bm_lin = torch.fft.irfft(Y, n=L_pad)[:, :L]
        env_X = torch.fft.rfft(bm_lin.abs(), n=L_pad)
        env_smooth = torch.fft.irfft(env_X * H_env[None, :], n=L_pad)[:, :L].clamp(min=1e-5)
        # PARALLEL DRNL: three operating points (strong/med/weak compression)
        ihcs = []
        for theta in self.BM_THETA_LIST:
            gain = (theta / (theta + env_smooth)) ** (1.0 - self.BM_C)
            bm = bm_lin * gain
            ihc_in = torch.relu(bm) + self.IHC_ASYMMETRY * torch.relu(-bm)
            ihc_X = torch.fft.rfft(ihc_in, n=L_pad)
            ihc_lp = torch.fft.irfft(ihc_X * H_ihc[None, :], n=L_pad)[:, :L].clamp(min=0.0)
            ihcs.append(torch.tanh(self.IHC_SAT_K * ihc_lp))
        return ihcs, env_smooth

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
        ihcs, env_smooth = self._periphery(raw_t)            # list of 3 (n_c, L)
        L = ihcs[0].shape[1]
        T = (L - WINMAX) // HOP + 1
        offset = WINMAX // 2 - HOP // 2

        feats = []
        log_eps = 1e-4

        # ---- 3 PARALLEL DRNL cochleagrams (LSR/MSR/HSR-like by SPL threshold)
        log_comps = []
        for ihc_ in ihcs:
            ds = self._box_avg(ihc_, T, offset)
            lc = torch.log(ds + log_eps)
            log_comps.append(lc)
            feats.append(lc)
        # Use MEDIUM-compression cochleagram as the primary log_comp downstream
        log_comp = log_comps[1]
        ds_ihc = self._box_avg(ihcs[1], T, offset)
        ds_env = self._box_avg(env_smooth, T, offset)
        cube_comp = torch.pow(ds_ihc + 1e-7, 1.0 / 3.0)
        log_drnl_gain = torch.log((ds_ihc + 1e-7) / (ds_env + 1e-7))
        feats.extend([cube_comp, log_drnl_gain])
        # Cochleagram for ACF: use the medium one
        ihc = ihcs[1]

        # ---- Sub-frame ACF summary (Meddis-Hewitt pitch detector)
        acf_TCK = self._winmax_acf(ihc, T)
        sum_acf = acf_TCK.sum(dim=1)                        # (T, K)
        max_acf = acf_TCK.max(dim=1).values
        feats.extend([sum_acf, max_acf])

        # ---- Harmonic-template pitch (cross-channel matched filter)
        baseline = log_comp.mean(dim=1, keepdim=True)
        templ_scores = []
        for f0, h_idx in zip(self.F0_CANDS, self.harm_idx):
            if not h_idx:
                templ_scores.append(torch.zeros(T, 1, device=raw_t.device))
                continue
            sal = log_comp[:, h_idx].mean(dim=1, keepdim=True) - baseline
            templ_scores.append(sal)
        harm = torch.cat(templ_scores, dim=1)
        feats.append(harm)
        best_str, best_idx = harm.max(dim=1, keepdim=True)
        f0_cands_t = torch.tensor(self.F0_CANDS, device=raw_t.device, dtype=torch.float32)
        log_f0 = torch.log(f0_cands_t[best_idx.squeeze(1)].unsqueeze(1) + 1.0)
        # RELATIVE F0 (median-subtracted, session-invariant)
        rel_log_f0 = log_f0 - log_f0.median()
        feats.append(torch.cat([best_str, log_f0, rel_log_f0], dim=1))

        # ---- LONG-CONTEXT EMA on log_comp (beyond FIR taps)
        for tau in self.LONG_TAUS:
            feats.append(log_comp - causal_ema(log_comp, tau))

        # ---- MOC global slow loudness contrast (session-invariant via median-subtraction)
        loud = log_comp.mean(dim=1, keepdim=True)
        rel_loud = loud - loud.median()
        slow_loud = causal_ema(loud, self.MOC_TAU)
        feats.append(loud - slow_loud)
        feats.append(rel_loud)

        # ---- Lateral inhibition DoG (single scale)
        feats.append(channel_dog(log_comp, 1.0, 3.0))

        # ---- 2-D STMF on log_comp (Chi-Shamma rate-scale)
        for (wt, wf) in self.STMF_RATES:
            feats.append(stmf_gabor_2d(log_comp, wt, wf))

        # ---- CEPSTRAL coefficients (DCT, abstract from spectral envelope)
        ceps = log_comp @ self.dct_basis
        feats.append(ceps)
        d_c = torch.zeros_like(ceps); d_c[1:] = ceps[1:] - ceps[:-1]
        dd_c = torch.zeros_like(ceps); dd_c[1:] = d_c[1:] - d_c[:-1]
        feats.extend([d_c, dd_c])
        # Long-tau cepstra dynamics
        for tau in self.LONG_TAUS:
            feats.append(ceps - causal_ema(ceps, tau))

        # ---- Phonetic matched filters
        nF = self.N_BM
        very_lo = log_comp[:, :max(1, nF // 8)].mean(dim=1, keepdim=True)
        lo = log_comp[:, :nF // 4].mean(dim=1, keepdim=True)
        mid_lo = log_comp[:, nF // 6:nF // 3].mean(dim=1, keepdim=True)
        mid = log_comp[:, nF // 4:3 * nF // 4].mean(dim=1, keepdim=True)
        mid_hi = log_comp[:, nF // 2:3 * nF // 4].mean(dim=1, keepdim=True)
        hi = log_comp[:, 3 * nF // 4:].mean(dim=1, keepdim=True)
        very_hi = log_comp[:, max(0, nF - nF // 8):].mean(dim=1, keepdim=True)
        spec_var = log_comp.var(dim=1, keepdim=True)
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

        # ---- Spectral flux, centroid, spread, flatness
        d = torch.zeros_like(log_comp); d[1:] = log_comp[1:] - log_comp[:-1]
        feats.append(torch.cat([d.clamp(min=0).sum(dim=1, keepdim=True),
                                 (-d).clamp(min=0).sum(dim=1, keepdim=True),
                                 d.abs().sum(dim=1, keepdim=True)], dim=1))
        e_w = torch.softmax(log_comp, dim=1)
        ch_idx = torch.arange(nF, device=log_comp.device, dtype=log_comp.dtype).view(1, -1)
        centroid = (e_w * ch_idx).sum(dim=1, keepdim=True)
        spread = torch.sqrt(((e_w * (ch_idx - centroid) ** 2).sum(dim=1, keepdim=True)) + 1e-6)
        flat = log_comp.mean(dim=1, keepdim=True) - torch.log(
            ds_ihc.mean(dim=1, keepdim=True) + 1e-7)
        feats.append(torch.cat([centroid, spread, flat], dim=1))

        # ---- Future deltas on log_comp (3, 6, 12) — STG STRF lookahead
        for k in (3, 6, 12):
            fut = torch.cat([log_comp[k:], log_comp[-1:].expand(k, -1)], dim=0)
            feats.append(fut - log_comp)

        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        raw = reconstruct_raw(np.asarray(windows, dtype=np.float32))
        raw_t = torch.from_numpy(raw).to(self.device)
        feat = self._periphery_features(raw_t)
        return feat.float().cpu().numpy()


model_shorthand_name = "BIOPHYS-v15"
model_description = (
    "BIOPHYS v15: v14 + PARALLEL DRNL (3 operating points: theta=0.002/"
    "0.02/0.2). Three log-cochleagrams from the same audio at different "
    "compression knees — emulates fiber populations with different SPL "
    "thresholds. Now that RMS-normalization (v12 fix) puts speech RMS at "
    "the medium knee, each cochleagram captures a different part of the "
    "dynamic range and combining all 3 gives the ridge access to the "
    "true biophysical multi-fiber AN code. v14=0.0632 (3496 feats)."
)


def build_embedder(device: str = "cuda") -> BiophysPeripheryV14:
    return BiophysPeripheryV14(device=device)


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
