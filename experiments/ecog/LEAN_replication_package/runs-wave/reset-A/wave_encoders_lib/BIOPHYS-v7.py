"""
BIOPHYSICAL AUDITORY PERIPHERY front-end (v5).

The core periphery is unchanged: dual-path DRNL → IHC → AN, with MOC AGC.
v5 enlarges the per-frame analysis window for ACF from HOP (33 ms) to WINMAX
(93 ms) using Wiener-Khinchin so that low-F0 periodicities are detected reliably,
adds a Meddis-Hewitt summary autocorrelogram across channels (robust pitch),
adds PER-CHANNEL slow MOC gain normalization (not just global), and adds an
explicit cochlear "channel-pair coincidence" feature for harmonic synchrony.
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


def gabor_envelope(x: torch.Tensor, rate_hz: float, fs: float = 30.0) -> torch.Tensor:
    period = fs / rate_hz
    sigma = period
    L = int(np.ceil(4 * sigma))
    t = np.arange(-L, L + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs); sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
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

class BiophysPeripheryV5:
    win = WINMAX
    n_params = 0

    N_BM = 48
    F_MIN = 80.0
    F_MAX = 8000.0
    BM_ORDER = 4

    AUDIO_GAIN = 50.0
    BM_THETA_LIN = 1.0
    BM_THETA_COMP = 0.05
    BM_C = 0.25
    BM_ENV_LP_HZ = 200.0

    IHC_LP_HZ = 1000.0
    IHC_ASYMMETRY = 0.15
    IHC_SAT_K = 2.0

    HSR_HALF_SAT = 0.05

    ADAPT_TAUS = (1.0, 3.0, 9.0, 30.0, 90.0)
    MOC_TAU = 30.0  # ~1 s

    STMF_RATES = (
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (16.0, 0.25), (16.0, -0.25),
        (0.0, 0.25), (0.0, 0.5),
    )

    # 16 F0 candidates spanning 70-500 Hz
    F0_CANDS = tuple(np.geomspace(70.0, 500.0, 16).tolist())

    def __init__(self, device: str = "cuda"):
        self.device = device
        fcs = erb_space(self.F_MIN, self.F_MAX, self.N_BM).astype(np.float32)
        self.fcs = torch.from_numpy(fcs).to(device)
        self.bw_bm = 1.019 * 24.7 * (4.37 * self.fcs / 1000.0 + 1.0)
        # ACF lags in samples at 44.1 kHz, for F0 candidates
        self.acf_lags = [int(round(SR / f0)) for f0 in self.F0_CANDS]
        # Predicted harmonic channel indices for each F0 candidate
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

    def _periphery_fast(self, raw_t: torch.Tensor):
        L = raw_t.shape[0]
        # Pad to a smooth length for cuFFT (next mult of 2^11 = 2048)
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
        env_smooth = torch.fft.irfft(env_X * H_env[None, :], n=L_pad)[:, :L].clamp(min=1e-5)
        gain_lin = (self.BM_THETA_LIN / (self.BM_THETA_LIN + env_smooth)) ** (1.0 - self.BM_C)
        gain_comp = (self.BM_THETA_COMP / (self.BM_THETA_COMP + env_smooth)) ** (1.0 - self.BM_C)
        bm_linpath = bm_lin * gain_lin
        bm_comp = bm_lin * gain_comp

        def ihc(bm):
            ihc_in = torch.relu(bm) + self.IHC_ASYMMETRY * torch.relu(-bm)
            ihc_X = torch.fft.rfft(ihc_in, n=L_pad)
            ihc_lp = torch.fft.irfft(ihc_X * H_ihc[None, :], n=L_pad)[:, :L].clamp(min=0.0)
            return torch.tanh(self.IHC_SAT_K * ihc_lp)

        ihc_comp = ihc(bm_comp)
        ihc_lin = ihc(bm_linpath)
        hsr = ihc_comp / (ihc_comp + self.HSR_HALF_SAT)
        return ihc_lin, ihc_comp, hsr

    def _box_avg(self, z: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        chop = offset + T * HOP
        return z[:, offset:chop].reshape(self.N_BM, T, HOP).mean(dim=-1).T

    def _winmax_acf(self, ihc: torch.Tensor, T: int) -> torch.Tensor:
        """Per-channel per-frame autocorrelation using a WINMAX-wide window
        centered on each frame's center sample. Returns (T, n_c, n_lags)."""
        # Frame i window: ihc[:, i*HOP : i*HOP + WINMAX]
        # Use as_strided / unfold for efficiency
        frames = ihc.unfold(dimension=1, size=WINMAX, step=HOP)[:, :T, :]   # (C, T, W)
        # Subtract mean
        frames = frames - frames.mean(dim=-1, keepdim=True)
        # ACF via FFT
        Nfft = 2 * WINMAX
        F = torch.fft.rfft(frames, n=Nfft)
        acf_full = torch.fft.irfft(F * F.conj(), n=Nfft)[..., :WINMAX]
        norm = acf_full[..., 0:1].clamp(min=1e-8)
        acf_norm = acf_full / norm                                  # (C, T, WINMAX)
        # Extract values at chosen lags
        lag_t = torch.tensor(self.acf_lags, device=ihc.device, dtype=torch.long)
        sel = acf_norm.index_select(-1, lag_t)                       # (C, T, n_lags)
        return sel.permute(1, 0, 2).contiguous()                     # (T, C, n_lags)

    def _multiwin_cochleagrams(self, windows_t: torch.Tensor) -> dict:
        """Hann-windowed FFT cochleagrams on the harness-aligned windows at 3
        analysis window sizes. Provides finer-time-res complement to the
        periphery's chunk-wide FFT. Returns (log_e, cube_e) per window size
        plus per-window pitch+moments."""
        T, W = windows_t.shape
        results = {}
        for w in (1024, 2048, 4096):
            start = (W - w) // 2
            xw = windows_t[:, start:start + w]
            hann = torch.hann_window(w, periodic=True, device=xw.device)
            P = torch.fft.rfft(xw * hann, n=w).abs() ** 2          # (T, w//2+1)
            freqs = torch.fft.rfftfreq(w, d=1.0 / SR).to(xw.device)
            df = freqs[None, :] - self.fcs[:, None]
            bw = self.bw_bm * (1.0 + 800.0 / w)  # widen filters for short win
            Hw = (1.0 + (df / bw[:, None]) ** 2) ** (-self.BM_ORDER / 2.0)
            e = P @ Hw.T                                            # (T, C)
            log_e = torch.log(e + 1e-7)
            cube_e = torch.pow(e + 1e-7, 1.0 / 3.0)
            results[w] = (log_e, cube_e, freqs, P)
        return results

    def _periphery_features(self, raw_t: torch.Tensor,
                              windows_t: torch.Tensor) -> torch.Tensor:
        ihc_lin, ihc_comp, hsr_full = self._periphery_fast(raw_t)
        L = ihc_comp.shape[1]
        T = (L - WINMAX) // HOP + 1
        offset = WINMAX // 2 - HOP // 2

        feats = []
        log_eps = 1e-4

        # ---- Sub-frame ACF on IHC_comp with WINMAX-wide window
        acf_TCK = self._winmax_acf(ihc_comp, T)            # (T, C, K)
        n_lags = len(self.acf_lags)
        # Summary across channels (Meddis-Hewitt pitch)
        sum_acf = acf_TCK.sum(dim=1)                        # (T, K)
        max_acf = acf_TCK.max(dim=1).values                 # (T, K)
        feats.extend([sum_acf, max_acf])

        # Per-channel periodicity at ONE consolidated lag (the strongest F0 per frame)
        # Keep small: max ACF per channel across lags (= "periodicity strength")
        per_ch_periodicity = acf_TCK.max(dim=2).values      # (T, C) — strongest F0 strength per channel
        feats.append(per_ch_periodicity)

        # Pitch summary scalars
        best_k_str, best_k_idx = sum_acf.max(dim=1, keepdim=True)
        f0_cands_t = torch.tensor(self.F0_CANDS, device=raw_t.device, dtype=torch.float32)
        log_f0 = torch.log(f0_cands_t[best_k_idx.squeeze(1)].unsqueeze(1) + 1.0)
        feats.append(torch.cat([best_k_str, log_f0], dim=1))

        # ---- Cochleagrams
        ds_comp = self._box_avg(ihc_comp, T, offset)
        ds_lin = self._box_avg(ihc_lin, T, offset)
        ds_hsr = self._box_avg(hsr_full, T, offset)
        log_comp = torch.log(ds_comp + log_eps)
        log_lin = torch.log(ds_lin + log_eps)
        log_hsr = torch.log(ds_hsr + log_eps)
        feats.extend([log_comp, log_lin, log_hsr])

        # ---- AN multi-tau subtractive adaptation on log_hsr
        for tau in self.ADAPT_TAUS:
            feats.append(log_hsr - causal_ema(log_hsr, tau))

        # ---- PER-CHANNEL MOC AGC: slow per-band gain normalization
        slow_per_ch = causal_ema(log_comp, self.MOC_TAU)
        feats.append(log_comp - slow_per_ch)
        # Global slow loudness contrast
        slow_global = causal_ema(log_comp.mean(dim=1, keepdim=True), self.MOC_TAU)
        feats.append(log_comp - slow_global)

        # ---- Lateral inhibition DoG (two scales)
        feats.append(channel_dog(log_comp, 1.0, 3.0))
        feats.append(channel_dog(log_hsr, 1.0, 3.0))

        # ---- Onset / offset HWR of d/dt log_comp
        d = torch.zeros_like(log_comp); d[1:] = log_comp[1:] - log_comp[:-1]
        feats.extend([d.clamp(min=0), (-d).clamp(min=0)])

        # ---- 2-D STMF on the comp log cochleagram
        for (wt, wf) in self.STMF_RATES:
            feats.append(stmf_gabor_2d(log_comp, wt, wf))

        # ---- Phonetic matched filters (small but high-level)
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
        fric = mid_hi - lo - 0.5 * best_k_str
        vowel = mid_lo + 0.5 * best_k_str - 0.3 * very_hi
        plosive = spec_var - 0.5 * best_k_str
        nasal = very_lo - hi
        approx = mid + 0.3 * best_k_str - 0.3 * spec_var
        voicedstop = best_k_str + lo - 0.3 * very_hi
        phon = torch.cat([sibilance, fric, vowel, plosive, nasal, approx,
                          voicedstop], dim=1)
        feats.append(phon)
        for tau in (3.0, 9.0, 30.0):
            feats.append(phon - causal_ema(phon, tau))

        # ---- Per-band Gabor AM envelopes (3 rates)
        for rate in (4.0, 8.0, 16.0):
            feats.append(gabor_envelope(log_comp, rate))

        # ---- Future / past deltas on log_comp
        for k in (3, 6, 12):
            fut = torch.cat([log_comp[k:], log_comp[-1:].expand(k, -1)], dim=0)
            feats.append(fut - log_comp)
            past = torch.cat([log_comp[:1].expand(k, -1), log_comp[:-k]], dim=0)
            feats.append(log_comp - past)

        # ---- Acausal slow loudness (future context)
        feats.append(acausal_ema(log_comp, self.MOC_TAU))

        # ---- BROADBAND ENVELOPE features — central STG predictor.
        # Loudness in 3 compressive forms over multiple smoothing taus.
        loud_log = log_comp.mean(dim=1, keepdim=True)
        loud_cube = torch.pow(ds_comp + 1e-7, 1.0 / 3.0).mean(dim=1, keepdim=True)
        loud_lin = ds_comp.mean(dim=1, keepdim=True)
        feats.extend([loud_log, loud_cube, loud_lin])
        # Multi-tau smoothing of loudness — captures slow loudness contour
        for tau in (1.0, 3.0, 9.0, 30.0):
            feats.append(causal_ema(loud_log, tau))
            feats.append(loud_log - causal_ema(loud_log, tau))
        # Loudness derivative (envelope onset/offset)
        dL = torch.zeros_like(loud_log); dL[1:] = loud_log[1:] - loud_log[:-1]
        feats.extend([dL, dL.clamp(min=0), (-dL).clamp(min=0)])
        # Loudness 2nd derivative (envelope onset acceleration)
        d2L = torch.zeros_like(loud_log); d2L[1:] = dL[1:] - dL[:-1]
        feats.extend([d2L.clamp(min=0), (-d2L).clamp(min=0)])
        # Loudness modulation envelopes at speech-relevant rates
        for rate in (2.0, 4.0, 8.0, 16.0):
            feats.append(gabor_envelope(loud_log, rate))

        # ---- HARMONIC TEMPLATE PITCH (cross-channel matched filter)
        # For each F0 candidate, sum log_comp values at predicted harmonic channels
        # minus the cochleagram baseline. Robust pitch.
        baseline = log_comp.mean(dim=1, keepdim=True)
        templ_scores = []
        for f0, h_idx in zip(self.F0_CANDS, self.harm_idx):
            if not h_idx:
                templ_scores.append(torch.zeros(T, 1, device=raw_t.device))
                continue
            sal = log_comp[:, h_idx].mean(dim=1, keepdim=True) - baseline
            templ_scores.append(sal)
        harm_template = torch.cat(templ_scores, dim=1)  # (T, 16)
        feats.append(harm_template)
        best_t_str, best_t_idx = harm_template.max(dim=1, keepdim=True)
        log_f0_t = torch.log(f0_cands_t[best_t_idx.squeeze(1)].unsqueeze(1) + 1.0)
        feats.append(torch.cat([best_t_str, log_f0_t], dim=1))
        # F0 derivatives + multi-tau
        for tau in (3.0, 9.0, 30.0):
            feats.append(log_f0_t - causal_ema(log_f0_t, tau))
            feats.append(best_t_str - causal_ema(best_t_str, tau))

        # ---- PER-CHANNEL MODULATION-RATE features (cochlear modulation spectrum)
        # For each channel, compute variance of log_comp in 3 modulation bands
        # via Gabor band-pass envelopes — captures HOW MUCH each channel
        # modulates at 2 / 6 / 12 Hz.
        for rate in (3.0, 6.0, 12.0):
            mod = gabor_envelope(log_comp, rate)
            feats.append(mod)

        # ---- SUB-FRAME envelope statistics within each HOP
        # Captures within-frame onsets / energy distribution per channel.
        # Reshape ihc_comp (n_c, L) into (n_c, T, HOP), get 4-bin energy.
        chop = offset + T * HOP
        n_sub = 5  # HOP=1470 = 5 * 294
        sub_w = HOP // n_sub
        sub = ihc_comp[:, offset:chop].reshape(self.N_BM, T, n_sub, sub_w).mean(dim=-1)
        sub = sub.permute(1, 0, 2).contiguous()  # (T, C, n_sub)
        early = sub[:, :, 0]
        late = sub[:, :, -1]
        e_l_diff = (early - late)  # (T, C)
        feats.append(e_l_diff)
        # Within-frame max - min per channel (peak/valley range)
        sub_max = sub.max(dim=2).values
        sub_min = sub.min(dim=2).values
        feats.append(torch.log(sub_max + 1e-7) - torch.log(sub_min + 1e-7))

        # ---- MULTI-WINDOW Hann-FFT COCHLEAGRAMS (1024 / 2048 / 4096 sample windows)
        # Provides fine, medium, and coarse time-frequency resolution complements
        # to the chunk-wide periphery FFT.
        mw = self._multiwin_cochleagrams(windows_t)
        for w in (1024, 2048, 4096):
            log_e, cube_e, freqs_w, P = mw[w]
            feats.extend([log_e, cube_e])
            # Per-window multi-tau dynamics
            for tau in (3.0, 9.0, 30.0):
                feats.append(log_e - causal_ema(log_e, tau))
            # Per-window spectral flux (broadband)
            d = torch.zeros_like(log_e); d[1:] = log_e[1:] - log_e[:-1]
            feats.append(torch.cat([d.clamp(min=0).sum(dim=1, keepdim=True),
                                     (-d).clamp(min=0).sum(dim=1, keepdim=True),
                                     d.abs().sum(dim=1, keepdim=True)], dim=1))
            # Per-window spectral moments
            P_norm = P / (P.sum(dim=1, keepdim=True) + 1e-12)
            mu = (P_norm * freqs_w).sum(dim=1, keepdim=True)
            std = torch.sqrt(((P_norm * (freqs_w - mu) ** 2).sum(dim=1, keepdim=True)) + 1e-6)
            feats.append(torch.cat([torch.log(mu.clamp(min=1.0)),
                                     torch.log(std.clamp(min=1.0))], dim=1))
            # Spectral flatness (log of geo/arith mean)
            log_P = torch.log(P.clamp(min=1e-10))
            flat = log_P.mean(dim=1, keepdim=True) - torch.log(P.mean(dim=1, keepdim=True) + 1e-10)
            feats.append(flat)
            # Spectral rolloff at 85% energy
            cs = torch.cumsum(P, dim=1)
            total = cs[:, -1:] + 1e-12
            roll_idx = (cs >= 0.85 * total).float().argmax(dim=1)
            feats.append(torch.log(freqs_w[roll_idx].clamp(min=1.0)).unsqueeze(1))

        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        raw = reconstruct_raw(np.asarray(windows, dtype=np.float32))
        raw_t = torch.from_numpy(raw).to(self.device)
        windows_t = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feat = self._periphery_features(raw_t, windows_t)
        return feat.float().cpu().numpy()


# ---------------------------------------------------------------------------

model_shorthand_name = "BIOPHYS-v7"
model_description = (
    "Biophysical periphery v7: v6 base + MULTI-WINDOW HANN-FFT COCHLEAGRAMS "
    "at 1024/2048/4096 sample windows (fine/medium/coarse time-freq tradeoffs), "
    "each with log+cube, 3-tau dynamics, broadband flux, spectral moments "
    "(centroid/spread), flatness, 85% rolloff. The periphery's chunk-wide "
    "FFT covers long-time structure; the harness-aligned short-window FFTs "
    "cover fast onsets/transients. v6 test_corr=0.0652."
)


def build_embedder(device: str = "cuda") -> BiophysPeripheryV5:
    return BiophysPeripheryV5(device=device)


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
