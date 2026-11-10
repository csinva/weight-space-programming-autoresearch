from __future__ import annotations
SR = 44100
import argparse
import hashlib
import os
import sys
import time
import numpy as np
import torch

def morlet_bank_freq(n_fft: int, sr: float, fcs: np.ndarray, Q: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    sigmas = np.maximum(fcs / max(Q, 0.5), 1.0)
    df = freqs[None, :] - fcs[:, None]
    psi = np.exp(-0.5 * (df / sigmas[:, None]) ** 2).astype(np.float32)
    psi[:, 0] = 0.0
    return psi

def log_space(f_min: float, f_max: float, Q: float) -> np.ndarray:
    n_oct = np.log2(f_max / f_min)
    n = int(np.round(n_oct * Q)) + 1
    return (f_min * 2.0 ** (np.arange(n) / Q)).astype(np.float32)

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)

def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37

def gammatone_powerbank(win, n_filters=50, f_min=50.0, f_max=8000.0, order=4):
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))

def power_spectrum(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2

def leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()

def acausal_leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    return torch.flip(leaky_lowpass(torch.flip(x, dims=[0]), tau_frames), dims=[0])

def harmonic_comb_salience(power: torch.Tensor, win: int, f0_candidates: torch.Tensor, n_harmonics: int=8) -> torch.Tensor:
    n_freqs = power.shape[1]
    freq_res = SR / win
    log_p = torch.log(torch.clamp(power, min=1e-10))
    total = log_p.mean(dim=1, keepdim=True)
    out = []
    for f0 in f0_candidates:
        ks = torch.arange(1, n_harmonics + 1, device=power.device, dtype=torch.float32)
        bins = (f0 * ks / freq_res).long().clamp(max=n_freqs - 1)
        sal = log_p[:, bins].mean(dim=1, keepdim=True) - total
        out.append(sal)
    return torch.cat(out, dim=1)

def spectral_moments(power: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    p = power / (power.sum(dim=1, keepdim=True) + 1e-12)
    mu = (p * freqs).sum(dim=1, keepdim=True)
    df = freqs - mu
    var = (p * df ** 2).sum(dim=1, keepdim=True)
    std = torch.sqrt(var + 1e-12)
    skew = (p * df ** 3).sum(dim=1, keepdim=True) / (std ** 3 + 1e-12)
    kurt = (p * df ** 4).sum(dim=1, keepdim=True) / (var ** 2 + 1e-12)
    return torch.cat([torch.log(mu.clamp(min=1.0)), torch.log(std.clamp(min=1.0)), skew.clamp(-10, 10), kurt.clamp(0, 50)], dim=1)

def spectral_flatness(power: torch.Tensor) -> torch.Tensor:
    log_p = torch.log(torch.clamp(power, min=1e-10))
    geo = log_p.mean(dim=1, keepdim=True)
    arith = torch.log(power.mean(dim=1, keepdim=True) + 1e-10)
    return geo - arith

def spectral_rolloff(power: torch.Tensor, freqs: torch.Tensor, thresholds=(0.85, 0.95)) -> torch.Tensor:
    cs = torch.cumsum(power, dim=1)
    total = cs[:, -1:] + 1e-12
    out = []
    for thr in thresholds:
        mask = cs >= thr * total
        idx = mask.float().argmax(dim=1)
        out.append(torch.log(freqs[idx].clamp(min=1.0)).unsqueeze(1))
    return torch.cat(out, dim=1)

def hnr_from_acf(power: torch.Tensor, win: int, f_min: float=75.0, f_max: float=400.0) -> torch.Tensor:
    full = torch.fft.irfft(power, n=win)
    acf0 = full[:, :1] + 1e-12
    acf_n = full / acf0
    lag_min = max(1, int(SR / f_max))
    lag_max = min(win - 1, int(SR / f_min))
    region = acf_n[:, lag_min:lag_max + 1]
    (peak_val, _) = region.max(dim=1, keepdim=True)
    peak_val = peak_val.clamp(min=0.0, max=0.999)
    hnr_db = 10.0 * torch.log10(peak_val.clamp(min=0.001) / (1.0 - peak_val).clamp(min=0.001))
    return torch.cat([peak_val, hnr_db], dim=1)

def phonetic_templates(log_e_long: torch.Tensor, n_filters: int) -> torch.Tensor:
    hi = log_e_long[:, max(0, n_filters - 6):].mean(dim=1, keepdim=True)
    lo = log_e_long[:, :max(1, n_filters // 3)].mean(dim=1, keepdim=True)
    mid = log_e_long[:, n_filters // 4:3 * n_filters // 4].mean(dim=1, keepdim=True)
    very_hi = log_e_long[:, max(0, n_filters - 3):].mean(dim=1, keepdim=True)
    narrow_lo = log_e_long[:, 1:6].mean(dim=1, keepdim=True)
    notch = log_e_long[:, 8:14].mean(dim=1, keepdim=True) if n_filters >= 14 else mid
    flatness = log_e_long.std(dim=1, keepdim=True)
    sibilant = hi - lo
    vowel = mid - flatness
    nasal = narrow_lo - notch
    plosive = flatness - log_e_long.mean(dim=1, keepdim=True)
    fricative = very_hi - mid
    return torch.cat([sibilant, vowel, nasal, plosive, fricative], dim=1)

def cross_band_comodulation(log_e: torch.Tensor) -> torch.Tensor:
    nF = log_e.shape[1]
    band_idx = np.linspace(0, nF - 1, 7).astype(int)
    bands = []
    for i in range(6):
        bands.append(log_e[:, band_idx[i]:band_idx[i + 1] + 1].mean(dim=1, keepdim=True))
    bands_t = torch.cat(bands, dim=1)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (2, 5)]
    out = []
    for (a, b) in pairs:
        out.append((bands_t[:, a] * bands_t[:, b]).unsqueeze(1))
    return torch.cat(out, dim=1)

class WaveEmbedder:
    win = 4096
    n_params = 0
    WIN_LIST = (1024, 2048, 4096)
    TAUS = (1.5, 4.5, 15.0, 60.0, 150.0)
    SUB_WIN = 1024
    F0_CANDIDATES = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0, 230.0, 270.0, 320.0, 380.0, 450.0)
    HL_TAUS = (15.0, 60.0, 150.0)
    STMF_RATES = ((2.0, 0.25), (2.0, -0.25), (4.0, 0.25), (4.0, -0.25), (4.0, 0.5), (4.0, -0.5), (8.0, 0.25), (8.0, -0.25), (8.0, 0.5), (8.0, -0.5), (12.0, 0.25), (12.0, -0.25), (4.0, 0.0), (8.0, 0.0), (12.0, 0.0), (0.0, 0.25), (0.0, 0.5))
    STMF_RATES_ADAPT = ((4.0, 0.25), (4.0, -0.25), (8.0, 0.25), (8.0, -0.25))
    STMF_RATES_FAST = ((12.0, 0.25), (12.0, -0.25), (16.0, 0.25), (16.0, -0.25))
    STMF_RATES_MID = ((4.0, 0.25), (4.0, -0.25), (8.0, 0.25), (8.0, -0.25), (8.0, 0.5), (8.0, -0.5))

    def __init__(self, device: str='cuda', n_filters: int=20):
        self.device = device
        self.n_filters = n_filters
        self.fbs = [gammatone_powerbank(w, n_filters=n_filters).to(device) for w in self.WIN_LIST]
        self.fb_sub = gammatone_powerbank(self.SUB_WIN, n_filters=n_filters).to(device)
        self.fb_fine = gammatone_powerbank(4096, n_filters=40).to(device)
        self.fb_fine_med = gammatone_powerbank(2048, n_filters=40).to(device)
        self.channel_idx = torch.arange(n_filters, dtype=torch.float32, device=device).view(1, -1)
        self.f0_cands = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32, device=device)
        self.long_freqs = torch.from_numpy(np.fft.rfftfreq(4096, d=1.0 / SR).astype(np.float32)).to(device)
        self._gabor_cache = {}
        self._stmf_cache = {}
        self._cond_logged = False
        self.scat_n_fft = 4096
        Q1 = 4.0
        Q2 = 1.0
        self.scat_fcs1 = log_space(50.0, 12800.0, Q1)
        psi1_np = morlet_bank_freq(self.scat_n_fft, SR, self.scat_fcs1, Q1)
        self.scat_psi1 = torch.from_numpy(psi1_np).to(device)
        self.scat_j1_sub = np.arange(0, len(self.scat_fcs1), 4)
        self.scat_fcs2 = log_space(0.5, 4096.0, Q2)
        psi2_np = morlet_bank_freq(self.scat_n_fft, SR, self.scat_fcs2, Q2)
        self.scat_psi2 = torch.from_numpy(psi2_np).to(device)
        self.scat_n_bins = 1
        self._scat_logged = False
        k = np.arange(13)
        n = np.arange(n_filters)
        dct = np.cos(np.pi / n_filters * (n[:, None] + 0.5) * k[None, :])
        dct[:, 0] *= 1.0 / np.sqrt(2)
        dct = dct * np.sqrt(2.0 / n_filters)
        self.dct_basis = torch.from_numpy(dct.astype(np.float32)).to(device)

    def _stmf_quadrature(self, log_e: torch.Tensor, omega_t: float, omega_f: float, fs: float=30.0):
        key = (round(omega_t, 3), round(omega_f, 3))
        if key not in self._stmf_cache:
            period_t = fs / max(abs(omega_t), 0.5)
            sigma_t = 1.0 * period_t / (2 * np.pi)
            Lt = max(1, int(np.ceil(3 * sigma_t)))
            sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi) * 1.0
            Lf = max(1, int(np.ceil(3 * sigma_f)))
            t = np.arange(-Lt, Lt + 1)
            f = np.arange(-Lf, Lf + 1)
            g = np.exp(-(t[:, None] ** 2 / (2 * sigma_t ** 2) + f[None, :] ** 2 / (2 * sigma_f ** 2)))
            phase = 2 * np.pi * omega_t * t[:, None] / fs + 2 * np.pi * omega_f * f[None, :]
            cos_k = (g * np.cos(phase)).astype(np.float32)
            sin_k = (g * np.sin(phase)).astype(np.float32)
            cos_k -= cos_k.mean()
            sin_k -= sin_k.mean()
            self._stmf_cache[key] = (torch.from_numpy(cos_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1), torch.from_numpy(sin_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1), Lt, Lf)
        (cos_k, sin_k, Lt, Lf) = self._stmf_cache[key]
        xp = log_e.unsqueeze(0).unsqueeze(0)
        c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
        s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
        (T, F) = log_e.shape
        return (c.squeeze(0).squeeze(0)[:T, :F], s.squeeze(0).squeeze(0)[:T, :F])

    def _stmf_envelope(self, log_e: torch.Tensor, omega_t: float, omega_f: float, fs: float=30.0) -> torch.Tensor:
        key = (round(omega_t, 3), round(omega_f, 3))
        if key not in self._stmf_cache:
            period_t = fs / max(abs(omega_t), 0.5)
            sigma_t = 1.0 * period_t / (2 * np.pi)
            Lt = max(1, int(np.ceil(3 * sigma_t)))
            sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi) * 1.0
            Lf = max(1, int(np.ceil(3 * sigma_f)))
            t = np.arange(-Lt, Lt + 1)
            f = np.arange(-Lf, Lf + 1)
            g = np.exp(-(t[:, None] ** 2 / (2 * sigma_t ** 2) + f[None, :] ** 2 / (2 * sigma_f ** 2)))
            phase = 2 * np.pi * omega_t * t[:, None] / fs + 2 * np.pi * omega_f * f[None, :]
            cos_k = (g * np.cos(phase)).astype(np.float32)
            sin_k = (g * np.sin(phase)).astype(np.float32)
            cos_k -= cos_k.mean()
            sin_k -= sin_k.mean()
            self._stmf_cache[key] = (torch.from_numpy(cos_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1), torch.from_numpy(sin_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1), Lt, Lf)
        (cos_k, sin_k, Lt, Lf) = self._stmf_cache[key]
        xp = log_e.unsqueeze(0).unsqueeze(0)
        c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
        s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
        (T, F) = log_e.shape
        env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
        return env[:T, :F]

    def _dog_channels(self, x: torch.Tensor, sigma_c: float, sigma_s: float) -> torch.Tensor:
        k = int(np.ceil(3 * sigma_s))
        coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
        g_c = torch.exp(-coords ** 2 / (2 * sigma_c ** 2))
        g_c = g_c / g_c.sum()
        g_s = torch.exp(-coords ** 2 / (2 * sigma_s ** 2))
        g_s = g_s / g_s.sum()
        dog = (g_c - g_s).view(1, 1, -1)
        return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)

    def _gabor_envelope(self, x: torch.Tensor, rate_hz: float, fs: float=30.0) -> torch.Tensor:
        if rate_hz not in self._gabor_cache:
            period = fs / rate_hz
            sigma = 1.0 * period
            L = int(np.ceil(4 * sigma))
            t = np.arange(-L, L + 1)
            g = np.exp(-t ** 2 / (2 * sigma ** 2))
            cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
            sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
            cos_k = cos_k - cos_k.mean()
            sin_k = sin_k - sin_k.mean()
            cos_k = cos_k / (np.linalg.norm(cos_k) + 1e-12)
            sin_k = sin_k / (np.linalg.norm(sin_k) + 1e-12)
            self._gabor_cache[rate_hz] = (torch.from_numpy(cos_k.astype(np.float32)).to(self.device), torch.from_numpy(sin_k.astype(np.float32)).to(self.device))
        (cos_k, sin_k) = self._gabor_cache[rate_hz]
        L = cos_k.numel()
        pad = L // 2
        C = x.shape[1]
        xp = x.transpose(0, 1).unsqueeze(0)
        ck = cos_k.view(1, 1, -1).expand(C, 1, L).contiguous()
        sk = sin_k.view(1, 1, -1).expand(C, 1, L).contiguous()
        xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
        c = torch.nn.functional.conv1d(xp_p, ck, groups=C)
        s = torch.nn.functional.conv1d(xp_p, sk, groups=C)
        c = c[..., :x.shape[0]]
        s = s[..., :x.shape[0]]
        env = torch.sqrt(c * c + s * s + 1e-12)
        return env.squeeze(0).transpose(0, 1).contiguous()
    cond_mode: str = 'none'
    interact_set: str = 'none'
    _COND_PRESETS = {'rep_target60_k8': dict(target=60, k_max=8), 'rep_target120_k16': dict(target=120, k_max=16), 'rep_target30_k4': dict(target=30, k_max=4)}

    def _replication_factor(self, name: str, d: int) -> int:
        if self.cond_mode in self._COND_PRESETS:
            p = self._COND_PRESETS[self.cond_mode]
            return max(1, min(p['k_max'], int(round(p['target'] / max(d, 1)))))
        if self.cond_mode == 'rep_sqrt60':
            return max(1, min(8, int(round(float(np.sqrt(60.0 / max(d, 1)))))))
        if self.cond_mode == 'rep_small_only':
            return 16 if d <= 4 else 1
        return 1

    @torch.no_grad()
    def _wavelet_scattering(self, sig: torch.Tensor) -> torch.Tensor:
        (T, N) = sig.shape
        NB = self.scat_n_bins
        bin_w = N // NB
        X = torch.fft.rfft(sig, n=N, dim=1)
        psi1 = self.scat_psi1
        psi2 = self.scat_psi2
        (n1, n2) = (psi1.shape[0], psi2.shape[0])
        s1_all = []
        s1_chunk = 16
        for s in range(0, n1, s1_chunk):
            psi_blk = psi1[s:s + s1_chunk]
            Xpsi = X.unsqueeze(1) * psi_blk.unsqueeze(0)
            yr = torch.fft.irfft(Xpsi, n=N, dim=2)
            yi = torch.fft.irfft(-1j * Xpsi, n=N, dim=2)
            env = torch.sqrt(yr * yr + yi * yi + 1e-20)
            s1_all.append(env.mean(dim=2))
            del Xpsi, yr, yi, env
        S1_all = torch.cat(s1_all, dim=1)
        S1_all_log = torch.pow(S1_all.clamp(min=0.0) + 1e-08, 1.0 / 3.0)
        sub_set = sorted((int(j) for j in self.scat_j1_sub))
        sub_envs = {}
        sub_if_mean = {}
        sub_coh = {}
        sub_if_slope = {}
        chunk = 8
        for s in range(0, len(sub_set), chunk):
            js = sub_set[s:s + chunk]
            psi_blk = psi1[js]
            Xpsi = X.unsqueeze(1) * psi_blk.unsqueeze(0)
            yr = torch.fft.irfft(Xpsi, n=N, dim=2)
            yi = torch.fft.irfft(-1j * Xpsi, n=N, dim=2)
            u = torch.sqrt(yr * yr + yi * yi + 1e-20)
            yc_re = yr[..., 1:] * yr[..., :-1] + yi[..., 1:] * yi[..., :-1]
            yc_im = yi[..., 1:] * yr[..., :-1] - yr[..., 1:] * yi[..., :-1]
            yc_re_m = yc_re.mean(dim=2)
            yc_im_m = yc_im.mean(dim=2)
            yc_mag = torch.sqrt(yc_re_m * yc_re_m + yc_im_m * yc_im_m + 1e-20)
            u2_mean = (u * u).mean(dim=2)
            coherence = yc_mag / u2_mean.clamp(min=1e-12)
            if_norm = torch.atan2(yc_im_m, yc_re_m) * (SR / (2 * np.pi))
            if_per_sample = torch.atan2(yc_im, yc_re) * (SR / (2 * np.pi))
            if_jitter_local = if_per_sample.std(dim=2)
            for (k_local, j1) in enumerate(js):
                sub_envs[j1] = u[:, k_local, :].contiguous()
                sub_if_mean[j1] = if_norm[:, k_local:k_local + 1]
                sub_coh[j1] = coherence[:, k_local:k_local + 1]
                sub_if_slope[j1] = if_jitter_local[:, k_local:k_local + 1]
            del Xpsi, yr, yi, u, yc_re, yc_im, yc_re_m, yc_im_m, yc_mag, u2_mean, coherence, if_norm, if_per_sample, if_jitter_local
        s1_means = []
        for j1 in sub_set:
            e = sub_envs[j1].view(T, NB, bin_w).mean(dim=2)
            s1_means.append(e)
        S1m = torch.stack(s1_means, dim=1)
        S1m_log = torch.log(S1m.clamp(min=1e-08))
        joint = (S1m[:, 1:, :] - S1m[:, :-1, :]).abs()
        s1_local = 0.5 * (S1m[:, 1:, :] + S1m[:, :-1, :]).clamp(min=1e-06)
        joint_log = torch.log1p(joint / s1_local)
        s2_means = []
        s2_stds = []
        s2_peak_per_j1_list = []
        fcs1 = self.scat_fcs1
        fcs2 = self.scat_fcs2
        s1_per_j1 = {}
        for j1 in sub_set:
            e = sub_envs[j1]
            s1_per_j1[j1] = e.view(T, NB, bin_w).mean(dim=2)
        for j1 in sub_set:
            keep = [j2 for j2 in range(n2) if fcs2[j2] < fcs1[j1] / 2.0]
            if not keep:
                continue
            e = sub_envs[j1]
            E = torch.fft.rfft(e, n=N, dim=1)
            psi2_blk = psi2[keep]
            Epsi = E.unsqueeze(1) * psi2_blk.unsqueeze(0)
            yr = torch.fft.irfft(Epsi, n=N, dim=2)
            yi = torch.fft.irfft(-1j * Epsi, n=N, dim=2)
            u2 = torch.sqrt(yr * yr + yi * yi + 1e-20)
            u2_b = u2.view(T, len(keep), NB, bin_w)
            m = u2_b.mean(dim=3)
            sd = u2_b.std(dim=3)
            pk = u2_b.amax(dim=3)
            s1_norm = s1_per_j1[j1].unsqueeze(1).clamp(min=1e-06)
            m_norm = m / s1_norm
            s2_means.append(m_norm.reshape(T, len(keep) * NB))
            s2_stds.append((sd / s1_norm).reshape(T, len(keep) * NB))
            pk_per_j1 = pk.amax(dim=1)
            s2_peak_per_j1_list.append(pk_per_j1)
            del E, Epsi, yr, yi, u2, u2_b, pk
        S2m = torch.cat(s2_means, dim=1) if s2_means else torch.zeros(T, 0, device=sig.device)
        S2s = torch.cat(s2_stds, dim=1) if s2_stds else torch.zeros(T, 0, device=sig.device)
        S2peak = torch.cat(s2_peak_per_j1_list, dim=1) if s2_peak_per_j1_list else torch.zeros(T, 0, device=sig.device)
        S2peak_log = torch.pow(S2peak.clamp(min=0.0) + 1e-08, 1.0 / 3.0)
        S2m_log = torch.log1p(S2m.clamp(min=0.0))
        S2s_log = torch.log1p(S2s.clamp(min=0.0))
        contour_blocks = []
        for j1 in sub_set:
            e = sub_envs[j1]
            q = N // 4
            ef = e[:, :q].mean(dim=1, keepdim=True).clamp(min=1e-08)
            el = e[:, -q:].mean(dim=1, keepdim=True).clamp(min=1e-08)
            contour_blocks.append(torch.log(el / ef))
        contour = torch.cat(contour_blocks, dim=1)
        s1_sub_log = torch.log(S1m.squeeze(2).clamp(min=1e-08))
        s1_peak_blocks = []
        for j1 in sub_set:
            s1_peak_blocks.append(sub_envs[j1].amax(dim=1, keepdim=True).clamp(min=1e-08))
        s1_peak_log = torch.log(torch.cat(s1_peak_blocks, dim=1))
        n_idx = torch.arange(N, device=sig.device, dtype=sig.dtype).view(1, -1)
        s1_centroid_blocks = []
        for j1 in sub_set:
            e = sub_envs[j1]
            num = (n_idx * e).sum(dim=1, keepdim=True)
            den = e.sum(dim=1, keepdim=True).clamp(min=1e-08)
            s1_centroid_blocks.append(num / den / max(N - 1, 1))
        s1_centroid = torch.cat(s1_centroid_blocks, dim=1)
        if_blocks = []
        coh_blocks = []
        for j1 in sub_set:
            fc = float(self.scat_fcs1[j1])
            ratio = sub_if_mean[j1] / fc
            if_blocks.append(ratio)
            coh_blocks.append(sub_coh[j1])
        if_feat = torch.cat(if_blocks, dim=1).clamp(-3.0, 3.0)
        coh_feat = torch.cat(coh_blocks, dim=1).clamp(0.0, 0.9999)
        coh_feat = torch.atanh(coh_feat)
        slope_blocks = []
        for j1 in sub_set:
            fc = float(self.scat_fcs1[j1])
            slope_blocks.append(sub_if_slope[j1] / fc)
        if_jitter = torch.cat(slope_blocks, dim=1).clamp(0.0, 3.0)
        peak_blocks = []
        for j1 in sub_set:
            e = sub_envs[j1]
            argmx = e.argmax(dim=1).float() / max(N - 1, 1)
            peak_blocks.append(argmx.unsqueeze(1))
        peak_pos = torch.cat(peak_blocks, dim=1)
        s2_entropy_blocks = []
        s2_off = 0
        for j1 in sub_set:
            keep = [j2 for j2 in range(self.scat_psi2.shape[0]) if self.scat_fcs2[j2] < self.scat_fcs1[j1] / 2.0]
            if not keep:
                s2_entropy_blocks.append(torch.zeros(T, 1, device=sig.device))
                continue
            slice_size = len(keep) * NB
            s2_block = S2m[:, s2_off:s2_off + slice_size]
            p = s2_block / s2_block.sum(dim=1, keepdim=True).clamp(min=1e-08)
            ent = -(p * torch.log(p.clamp(min=1e-10))).sum(dim=1, keepdim=True)
            s2_entropy_blocks.append(ent)
            s2_off += slice_size
        s2_entropy = torch.cat(s2_entropy_blocks, dim=1)
        s1_sub_dyn = []
        for tau in self.HL_TAUS:
            s1_sub_dyn.append(s1_sub_log - leaky_lowpass(s1_sub_log, tau_frames=tau))
        s1_sub_dyn = torch.cat(s1_sub_dyn, dim=1) if s1_sub_dyn else s1_sub_log[:, :0]
        out = torch.cat([S2m_log, S2s_log, S2peak_log, joint_log.reshape(T, -1), contour, s1_sub_log, s1_sub_dyn, s1_peak_log, if_feat, coh_feat, if_jitter, s2_entropy], dim=1)
        if not self._scat_logged:
            print(f'  scattering: K={len(sub_set)} NB={NB} dim={out.shape[1]} (S1m {S1m_log.shape[1]}, joint {joint_log.shape[1]}, S2m {S2m_log.shape[1]}, S2s {S2s_log.shape[1]})')
            self._scat_logged = True
        return out

    def _condition(self, groups: list) -> torch.Tensor:
        out = []
        rep_log = []
        for (name, t) in groups:
            d = t.shape[1]
            K = self._replication_factor(name, d)
            if K > 1:
                t = t.repeat(1, K)
                rep_log.append(f'{name}×{K}({d}→{d * K})')
            out.append(t)
        if rep_log and self.cond_mode != 'none' and (not self._cond_logged):
            print('  group conditioning:', ', '.join(rep_log))
            self._cond_logged = True
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        groups: list[tuple[str, torch.Tensor]] = []

        def add(name: str, t: torch.Tensor):
            groups.append((name, t))
        log_es = []
        powers = []
        cochlea_blocks = []
        adapt_blocks = []
        for (w, fb) in zip(self.WIN_LIST, self.fbs):
            if w == 4096:
                xw = x
            else:
                start = (4096 - w) // 2
                xw = x[:, start:start + w]
            p = power_spectrum(xw, w, self.device)
            powers.append(p)
            log_e = torch.log(torch.clamp(p @ fb, min=1e-10))
            log_es.append(log_e)
            cochlea_blocks.append(log_e)
            for tau in self.TAUS:
                adapt_blocks.append(log_e - leaky_lowpass(log_e, tau_frames=tau))
        add('cochleagram', torch.cat(cochlea_blocks, dim=1))
        add('adapt_delta', torch.cat(adapt_blocks, dim=1))
        power_long = powers[-1]
        log_e_long = log_es[-1]
        ps = harmonic_comb_salience(power_long, 4096, self.f0_cands, n_harmonics=8)
        add('pitch_salience', ps)
        (best_strength, best_idx) = ps.max(dim=1, keepdim=True)
        best_f0 = self.f0_cands[best_idx.squeeze(1)].unsqueeze(1)
        log_f0 = torch.log(best_f0.clamp(min=20.0))
        add('pitch_summary', torch.cat([best_strength, log_f0], dim=1))
        pitch_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            pitch_dyn_blocks.append(log_f0 - leaky_lowpass(log_f0, tau_frames=tau))
            pitch_dyn_blocks.append(best_strength - leaky_lowpass(best_strength, tau_frames=tau))
        add('pitch_dyn', torch.cat(pitch_dyn_blocks, dim=1))
        mom = spectral_moments(power_long, self.long_freqs)
        add('spectral_moments', mom)
        flat = spectral_flatness(power_long)
        add('spectral_flatness', flat)
        roll = spectral_rolloff(power_long, self.long_freqs)
        add('spectral_rolloff', roll)
        hnr = hnr_from_acf(power_long, 4096)
        add('hnr', hnr)
        phon = phonetic_templates(log_e_long, self.n_filters)
        add('phonetic', phon)
        comod = cross_band_comodulation(log_e_long)
        flux_feats = []
        for log_e_w in log_es:
            d = torch.zeros_like(log_e_w)
            d[1:] = log_e_w[1:] - log_e_w[:-1]
            flux_pos = d.clamp(min=0.0).sum(dim=1, keepdim=True)
            flux_neg = (-d).clamp(min=0.0).sum(dim=1, keepdim=True)
            flux_abs = d.abs().sum(dim=1, keepdim=True)
            flux_feats.append(torch.cat([flux_pos, flux_neg, flux_abs], dim=1))
        flux_all = torch.cat(flux_feats, dim=1)
        flux_dyn = [flux_all]
        for tau in self.HL_TAUS:
            flux_dyn.append(flux_all - leaky_lowpass(flux_all, tau_frames=tau))
        add('flux', torch.cat(flux_dyn, dim=1))
        long_loud = log_es[-1].mean(dim=1, keepdim=True)
        e_w = torch.softmax(log_es[-1], dim=1)
        long_centroid = (e_w * self.channel_idx).sum(dim=1, keepdim=True)
        mod_signals = [long_loud, long_centroid, hnr, best_strength, log_f0, flux_all, phon, mom, flat, roll, comod]
        mod_blocks = []
        for rate in (2.0, 4.0, 6.0, 10.0):
            for sig in mod_signals:
                mod_blocks.append(self._gabor_envelope(sig, rate))
        add('modulation', torch.cat(mod_blocks, dim=1))
        stmf_means = []
        stmf_blocks = []
        for (omega_t, omega_f) in self.STMF_RATES:
            stmf_e = self._stmf_envelope(log_e_long, omega_t, omega_f)
            stmf_blocks.append(stmf_e)
            stmf_means.append(stmf_e.mean(dim=1, keepdim=True))
        add('stmf', torch.cat(stmf_blocks, dim=1))
        stmf_mean_all = torch.cat(stmf_means, dim=1)
        add('stmf_mean_gabor', self._gabor_envelope(stmf_mean_all, 2.0))
        nF = self.n_filters
        f1_band = log_e_long[:, :nF // 2]
        f2_band = log_e_long[:, nF // 2:]
        w1 = torch.softmax(f1_band * 2.0, dim=1)
        w2 = torch.softmax(f2_band * 2.0, dim=1)
        ch1 = torch.arange(nF // 2, device=log_e_long.device, dtype=log_e_long.dtype).view(1, -1)
        ch2 = torch.arange(nF // 2, device=log_e_long.device, dtype=log_e_long.dtype).view(1, -1) + nF // 2
        f1_loc = (w1 * ch1).sum(dim=1, keepdim=True)
        f2_loc = (w2 * ch2).sum(dim=1, keepdim=True)
        f1_amp = (w1 * f1_band).sum(dim=1, keepdim=True)
        f2_amp = (w2 * f2_band).sum(dim=1, keepdim=True)
        f12_diff = f2_loc - f1_loc
        formants = torch.cat([f1_loc, f2_loc, f12_diff, f1_amp, f2_amp, f2_amp - f1_amp], dim=1)
        formants_blocks = [formants]
        for tau in (4.5, 15.0, 60.0):
            formants_blocks.append(formants - leaky_lowpass(formants, tau_frames=tau))
        add('formants', torch.cat(formants_blocks, dim=1))
        fmt_loc = torch.cat([f1_loc, f2_loc, f12_diff], dim=1)
        fmt_d = torch.zeros_like(fmt_loc)
        fmt_d[1:] = fmt_loc[1:] - fmt_loc[:-1]
        add('formant_dyn', torch.cat([fmt_d, fmt_d.clamp(min=0), (-fmt_d).clamp(min=0)], dim=1))
        cube_long_var = torch.pow(power_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)
        mdepth_blocks = []
        for (hp_tau, lp_tau) in ((4.5, 15.0), (1.5, 15.0)):
            hp = log_e_long - leaky_lowpass(log_e_long, tau_frames=hp_tau)
            mdepth_blocks.append(leaky_lowpass(hp.abs(), tau_frames=lp_tau))
        add('mod_depth', torch.cat(mdepth_blocks, dim=1))
        tonic_blocks = []
        for (i, (w, fb)) in enumerate(zip(self.WIN_LIST, self.fbs)):
            le_w = log_es[i]
            cube_w = torch.pow(powers[i] @ fb + 1e-10, 1.0 / 3.0)
            for tau in (60.0, 150.0):
                tonic_blocks.append(leaky_lowpass(le_w, tau_frames=tau))
                tonic_blocks.append(leaky_lowpass(cube_w, tau_frames=tau))
            tonic_blocks.append(acausal_leaky_lowpass(le_w, tau_frames=60.0))
        add('tonic', torch.cat(tonic_blocks, dim=1))
        fp_blocks = []
        for k in (3, 6, 12):
            fut_k = torch.cat([log_e_long[k:], log_e_long[-1:].expand(k, -1)], dim=0)
            future_delta = fut_k - log_e_long
            fp_blocks.append(future_delta)
            fp_blocks.append(future_delta.clamp(min=0.0))
            fp_blocks.append((-future_delta).clamp(min=0.0))
            past_k = torch.cat([log_e_long[:1].expand(k, -1), log_e_long[:-k]], dim=0)
            past_delta = log_e_long - past_k
            fp_blocks.append(past_delta)
        fut_c = torch.cat([cube_long_var[6:], cube_long_var[-1:].expand(6, -1)], dim=0)
        fp_blocks.append(fut_c - cube_long_var)
        add('future_past', torch.cat(fp_blocks, dim=1))
        f0_d1 = torch.zeros_like(log_f0)
        f0_d1[1:] = log_f0[1:] - log_f0[:-1]
        f0_d2 = torch.zeros_like(log_f0)
        f0_d2[1:] = f0_d1[1:] - f0_d1[:-1]
        add('f0_dyn', torch.cat([f0_d1, f0_d2, f0_d1.clamp(min=0), (-f0_d1).clamp(min=0)], dim=1))
        hnr_db = hnr[:, 1:2]
        hnr_d = torch.zeros_like(hnr_db)
        hnr_d[1:] = hnr_db[1:] - hnr_db[:-1]
        add('hnr_dyn', torch.cat([hnr_d, hnr_d.clamp(min=0), (-hnr_d).clamp(min=0)], dim=1))
        if self.interact_set != 'none':
            hnr_peak = hnr[:, 0:1]
            hnr_dvox = hnr[:, 1:2]
            long_loud_1d = long_loud
            flux_long = flux_feats[-1]
            flux_abs_long = flux_long[:, 2:3]
            flux_pos_long = flux_long[:, 0:1]
            f12_disp = f2_loc - f1_loc
            f12_amp = f1_amp + f2_amp
            inter = []
            if self.interact_set == 'v3e':
                key_sigs10 = [long_loud_1d, best_strength, log_f0, flux_abs_long, hnr_peak, long_centroid, flat, hnr_dvox, f12_disp, f12_amp]
                for sig in key_sigs10:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-06)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                for sig in key_sigs10:
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-06)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for j in range(phon.shape[1]):
                    sig = phon[:, j:j + 1]
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-06)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for sig in (f1_loc, f2_loc, f1_amp, f2_amp):
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-06)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)

                def _lag(sig, k):
                    return torch.cat([sig[:k], sig[:-k]], dim=0)
                inter.append(_lag(flux_abs_long, 2) * hnr_d)
                inter.append(_lag(hnr_d, 2) * flux_abs_long)
                inter.append(_lag(flux_abs_long, 2) * long_loud_1d)
                fwd_f0d1 = torch.cat([f0_d1[2:], f0_d1[-1:].expand(2, -1)], dim=0)
                inter.append(best_strength * fwd_f0d1)
                fwd_flux = torch.cat([flux_abs_long[2:], flux_abs_long[-1:].expand(2, -1)], dim=0)
                inter.append(long_loud_1d * fwd_flux)
                fwd_unv = torch.cat([(1.0 - hnr_peak)[2:], (1.0 - hnr_peak)[-1:].expand(2, -1)], dim=0)
                inter.append(hnr_peak * fwd_unv)
                phon_factors = [phon[:, j:j + 1] for j in range(phon.shape[1])]
                sig_factors = [best_strength, long_loud_1d, flux_abs_long, hnr_d]
                for pf in phon_factors:
                    for sf in sig_factors:
                        p = pf * sf
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-06)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                formant_factors = [f1_loc, f2_loc, f1_amp, f2_amp]
                for pf in phon_factors:
                    for ff in formant_factors:
                        p = pf * ff
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-06)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for sf in sig_factors:
                    for ff in formant_factors:
                        p = sf * ff
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-06)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                flux_dev = flux_abs_long - leaky_lowpass(flux_abs_long, tau_frames=15.0)
                fm = flux_dev.mean(dim=0, keepdim=True)
                fs = flux_dev.std(dim=0, keepdim=True).clamp(min=1e-06)
                fz = (flux_dev - fm) / fs
                onset_gate = fz.clamp(min=0.0)
                offset_gate = (-fz).clamp(min=0.0)
                for pf in phon_factors:
                    for gate in (onset_gate, offset_gate):
                        p = pf * gate
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-06)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
            if len(inter) > 0:
                add('interact_' + self.interact_set, torch.cat(inter, dim=1))
        log_pl = torch.log(torch.clamp(power_long, min=1e-10))
        cep = torch.fft.irfft(log_pl, n=4096)
        q_min = int(SR / 400.0)
        q_max = int(SR / 60.0)
        cep_pitch = cep[:, q_min:q_max + 1]
        (peak_val_cep, _) = cep_pitch.max(dim=1, keepdim=True)
        L_ = cep_pitch.shape[1]
        idx = torch.arange(L_, device=cep.device, dtype=cep.dtype).view(1, -1)
        mu_i = idx.mean(dim=1, keepdim=True)
        mu_y = cep_pitch.mean(dim=1, keepdim=True)
        di = idx - mu_i
        dy = cep_pitch - mu_y
        slope = (di * dy).sum(dim=1, keepdim=True) / ((di * di).sum(dim=1, keepdim=True) + 1e-12)
        intercept = mu_y - slope * mu_i
        argmax = cep_pitch.argmax(dim=1, keepdim=True).float()
        baseline = intercept + slope * argmax
        cpp = peak_val_cep - baseline
        cpp_blocks = [cpp]
        for tau in (4.5, 15.0, 60.0):
            cpp_blocks.append(cpp - leaky_lowpass(cpp, tau_frames=tau))
        add('cpp', torch.cat(cpp_blocks, dim=1))
        freqs_long = self.long_freqs
        hi_mask = (freqs_long >= 4000.0) & (freqs_long <= 8000.0)
        lo_mask = (freqs_long >= 60.0) & (freqs_long <= 500.0)
        p_hi = power_long[:, hi_mask]
        p_lo = power_long[:, lo_mask]
        flat_hi = torch.log(p_hi.clamp(min=1e-10)).mean(dim=1, keepdim=True) - torch.log(p_hi.mean(dim=1, keepdim=True) + 1e-10)
        flat_lo = torch.log(p_lo.clamp(min=1e-10)).mean(dim=1, keepdim=True) - torch.log(p_lo.mean(dim=1, keepdim=True) + 1e-10)
        region_flat = torch.cat([flat_hi, flat_lo], dim=1)
        rf_blocks = [region_flat]
        for tau in (4.5, 15.0, 60.0):
            rf_blocks.append(region_flat - leaky_lowpass(region_flat, tau_frames=tau))
        add('region_flat', torch.cat(rf_blocks, dim=1))
        contrast_list = []
        for (f_lo, f_hi) in ((62.0, 125.0), (125.0, 250.0), (250.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)):
            mk_o = (freqs_long >= f_lo) & (freqs_long <= f_hi)
            p_o = power_long[:, mk_o]
            if p_o.shape[1] < 4:
                contrast_list.append(torch.zeros(p_o.shape[0], 1, device=p_o.device))
                continue
            lp_o = torch.log(p_o.clamp(min=1e-10))
            n_o = lp_o.shape[1]
            k_top = max(1, n_o // 4)
            (top_v, _) = lp_o.topk(k_top, dim=1, largest=True)
            (bot_v, _) = lp_o.topk(k_top, dim=1, largest=False)
            cont_o = top_v.mean(dim=1, keepdim=True) - bot_v.mean(dim=1, keepdim=True)
            contrast_list.append(cont_o)
        contrast = torch.cat(contrast_list, dim=1)
        ct_blocks = [contrast]
        for tau in (4.5, 15.0, 60.0):
            ct_blocks.append(contrast - leaky_lowpass(contrast, tau_frames=tau))
        add('spectral_contrast', torch.cat(ct_blocks, dim=1))
        power_s = powers[0]
        freqs_short_np = np.fft.rfftfreq(1024, d=1.0 / SR).astype(np.float32)
        freqs_short = torch.from_numpy(freqs_short_np).to(self.device)
        contrast_short_list = []
        for (f_lo, f_hi) in ((62.0, 125.0), (125.0, 250.0), (250.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)):
            mk_so = (freqs_short >= f_lo) & (freqs_short <= f_hi)
            p_so = power_s[:, mk_so]
            if p_so.shape[1] < 2:
                contrast_short_list.append(torch.zeros(p_so.shape[0], 1, device=p_so.device))
                continue
            lp_so = torch.log(p_so.clamp(min=1e-10))
            n_so = lp_so.shape[1]
            k_top_s = max(1, n_so // 3)
            (top_v_s, _) = lp_so.topk(k_top_s, dim=1, largest=True)
            (bot_v_s, _) = lp_so.topk(k_top_s, dim=1, largest=False)
            contrast_short_list.append(top_v_s.mean(dim=1, keepdim=True) - bot_v_s.mean(dim=1, keepdim=True))
        contrast_short = torch.cat(contrast_short_list, dim=1)
        cts_blocks = [contrast_short]
        for tau in (4.5, 15.0, 60.0):
            cts_blocks.append(contrast_short - leaky_lowpass(contrast_short, tau_frames=tau))
        add('contrast_short', torch.cat(cts_blocks, dim=1))
        power_m = powers[1]
        freqs_med_np = np.fft.rfftfreq(2048, d=1.0 / SR).astype(np.float32)
        freqs_med = torch.from_numpy(freqs_med_np).to(self.device)
        contrast_med_list = []
        for (f_lo, f_hi) in ((62.0, 125.0), (125.0, 250.0), (250.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)):
            mk_m = (freqs_med >= f_lo) & (freqs_med <= f_hi)
            p_m = power_m[:, mk_m]
            if p_m.shape[1] < 2:
                contrast_med_list.append(torch.zeros(p_m.shape[0], 1, device=p_m.device))
                continue
            lp_m = torch.log(p_m.clamp(min=1e-10))
            n_m = lp_m.shape[1]
            k_top_m = max(1, n_m // 4)
            (top_v_m, _) = lp_m.topk(k_top_m, dim=1, largest=True)
            (bot_v_m, _) = lp_m.topk(k_top_m, dim=1, largest=False)
            contrast_med_list.append(top_v_m.mean(dim=1, keepdim=True) - bot_v_m.mean(dim=1, keepdim=True))
        contrast_med = torch.cat(contrast_med_list, dim=1)
        ctm_blocks = [contrast_med]
        for tau in (4.5, 15.0, 60.0):
            ctm_blocks.append(contrast_med - leaky_lowpass(contrast_med, tau_frames=tau))
        add('contrast_med', torch.cat(ctm_blocks, dim=1))
        cochlea_groups = []
        for gi in range(5):
            band = log_e_long[:, gi * 4:(gi + 1) * 4]
            (top_v_c, _) = band.topk(2, dim=1, largest=True)
            (bot_v_c, _) = band.topk(2, dim=1, largest=False)
            cochlea_groups.append(top_v_c.mean(dim=1, keepdim=True) - bot_v_c.mean(dim=1, keepdim=True))
        cochlea_contrast = torch.cat(cochlea_groups, dim=1)
        cc_blocks = [cochlea_contrast]
        for tau in (4.5, 15.0, 60.0):
            cc_blocks.append(cochlea_contrast - leaky_lowpass(cochlea_contrast, tau_frames=tau))
        add('cochlea_contrast', torch.cat(cc_blocks, dim=1))
        log_e_med = log_es[1]
        med_groups = []
        for gi in range(5):
            band = log_e_med[:, gi * 4:(gi + 1) * 4]
            (top_v_m, _) = band.topk(2, dim=1, largest=True)
            (bot_v_m, _) = band.topk(2, dim=1, largest=False)
            med_groups.append(top_v_m.mean(dim=1, keepdim=True) - bot_v_m.mean(dim=1, keepdim=True))
        med_cochlea_contrast = torch.cat(med_groups, dim=1)
        mcc_blocks = [med_cochlea_contrast]
        for tau in (4.5, 15.0, 60.0):
            mcc_blocks.append(med_cochlea_contrast - leaky_lowpass(med_cochlea_contrast, tau_frames=tau))
        add('med_cochlea_contrast', torch.cat(mcc_blocks, dim=1))
        log_e_short = log_es[0]
        short_groups = []
        for gi in range(5):
            band = log_e_short[:, gi * 4:(gi + 1) * 4]
            (top_v_s2, _) = band.topk(2, dim=1, largest=True)
            (bot_v_s2, _) = band.topk(2, dim=1, largest=False)
            short_groups.append(top_v_s2.mean(dim=1, keepdim=True) - bot_v_s2.mean(dim=1, keepdim=True))
        short_cochlea_contrast = torch.cat(short_groups, dim=1)
        scc_blocks = [short_cochlea_contrast]
        for tau in (4.5, 15.0, 60.0):
            scc_blocks.append(short_cochlea_contrast - leaky_lowpass(short_cochlea_contrast, tau_frames=tau))
        add('short_cochlea_contrast', torch.cat(scc_blocks, dim=1))
        mt_contrast_blocks = []
        for tau in (1.5, 15.0):
            le_adapt = log_e_long - leaky_lowpass(log_e_long, tau_frames=tau)
            for gi in range(5):
                band = le_adapt[:, gi * 4:(gi + 1) * 4]
                (top_v_mt, _) = band.topk(2, dim=1, largest=True)
                (bot_v_mt, _) = band.topk(2, dim=1, largest=False)
                mt_contrast_blocks.append(top_v_mt.mean(dim=1, keepdim=True) - bot_v_mt.mean(dim=1, keepdim=True))
        mtc = torch.cat(mt_contrast_blocks, dim=1)
        add('multitau_contrast', mtc)
        log_e_med2 = log_es[1]
        le_med_adapt = log_e_med2 - leaky_lowpass(log_e_med2, tau_frames=4.5)
        med_adapt_contrast = []
        for gi in range(5):
            band = le_med_adapt[:, gi * 4:(gi + 1) * 4]
            (top_v_ma, _) = band.topk(2, dim=1, largest=True)
            (bot_v_ma, _) = band.topk(2, dim=1, largest=False)
            med_adapt_contrast.append(top_v_ma.mean(dim=1, keepdim=True) - bot_v_ma.mean(dim=1, keepdim=True))
        mac = torch.cat(med_adapt_contrast, dim=1)
        add('med_adapt_contrast', mac)
        topbot_lst = []
        for gi in range(5):
            band = log_e_long[:, gi * 4:(gi + 1) * 4]
            (top_v_tb, _) = band.topk(2, dim=1, largest=True)
            (bot_v_tb, _) = band.topk(2, dim=1, largest=False)
            topbot_lst.append(top_v_tb.mean(dim=1, keepdim=True))
            topbot_lst.append(bot_v_tb.mean(dim=1, keepdim=True))
        topbot = torch.cat(topbot_lst, dim=1)
        add('topbot', topbot)
        log_e_short84 = log_es[0]
        topbot_short_lst = []
        for gi in range(5):
            band = log_e_short84[:, gi * 4:(gi + 1) * 4]
            (tv, _) = band.topk(2, dim=1, largest=True)
            (bv, _) = band.topk(2, dim=1, largest=False)
            topbot_short_lst.append(tv.mean(dim=1, keepdim=True))
            topbot_short_lst.append(bv.mean(dim=1, keepdim=True))
        topbot_short = torch.cat(topbot_short_lst, dim=1)
        add('topbot_short', topbot_short)
        k_fut = 3
        log_e_fut = torch.cat([log_e_long[k_fut:], log_e_long[-1:].expand(k_fut, -1)], dim=0)
        fut_contrast = []
        for gi in range(5):
            band = log_e_fut[:, gi * 4:(gi + 1) * 4]
            (tv, _) = band.topk(2, dim=1, largest=True)
            (bv, _) = band.topk(2, dim=1, largest=False)
            fut_contrast.append(tv.mean(dim=1, keepdim=True) - bv.mean(dim=1, keepdim=True))
        fc = torch.cat(fut_contrast, dim=1)
        add('fut_cochlea_contrast', fc)
        log_e_med92 = log_es[1]
        k_futm = 3
        log_e_fut_m = torch.cat([log_e_med92[k_futm:], log_e_med92[-1:].expand(k_futm, -1)], dim=0)
        fut_contrast_m = []
        for gi in range(5):
            band = log_e_fut_m[:, gi * 4:(gi + 1) * 4]
            (tv, _) = band.topk(2, dim=1, largest=True)
            (bv, _) = band.topk(2, dim=1, largest=False)
            fut_contrast_m.append(tv.mean(dim=1, keepdim=True) - bv.mean(dim=1, keepdim=True))
        fcm_ = torch.cat(fut_contrast_m, dim=1)
        add('fut_cochlea_contrast_med', fcm_)
        log_e_med96 = log_es[1]
        log_e_med_fut6 = torch.cat([log_e_med96[6:], log_e_med96[-1:].expand(6, -1)], dim=0)
        fut_med6 = []
        for gi in range(5):
            band = log_e_med_fut6[:, gi * 4:(gi + 1) * 4]
            (tv, _) = band.topk(2, dim=1, largest=True)
            (bv, _) = band.topk(2, dim=1, largest=False)
            fut_med6.append(tv.mean(dim=1, keepdim=True) - bv.mean(dim=1, keepdim=True))
        fm6 = torch.cat(fut_med6, dim=1)
        add('fut_med_contrast_6', fm6)
        cur_contrast_lst = []
        for gi in range(5):
            band = log_e_long[:, gi * 4:(gi + 1) * 4]
            (tv, _) = band.topk(2, dim=1, largest=True)
            (bv, _) = band.topk(2, dim=1, largest=False)
            cur_contrast_lst.append(tv.mean(dim=1, keepdim=True) - bv.mean(dim=1, keepdim=True))
        cur_contrast = torch.cat(cur_contrast_lst, dim=1)
        fmc_minus_cur = fm6 - cur_contrast
        add('fut_minus_cur_contrast', fmc_minus_cur)
        cube_long = torch.pow(power_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)
        cube_long_fut = torch.cat([cube_long[3:], cube_long[-1:].expand(3, -1)], dim=0)
        cube_fut_contrast = []
        for gi in range(5):
            band = cube_long_fut[:, gi * 4:(gi + 1) * 4]
            (tv, _) = band.topk(2, dim=1, largest=True)
            (bv, _) = band.topk(2, dim=1, largest=False)
            cube_fut_contrast.append(tv.mean(dim=1, keepdim=True) - bv.mean(dim=1, keepdim=True))
        cfc = torch.cat(cube_fut_contrast, dim=1)
        add('cube_fut_contrast', cfc)
        log_e_fine_long = torch.log(torch.clamp(power_long @ self.fb_fine, min=1e-10))
        QUAD_RATES = (2.0, 4.0, 8.0, 12.0)
        QUAD_SCALES = (0.0, 0.25, -0.25, 0.5, -0.5)
        nF40 = log_e_fine_long.shape[1]
        bin_edges = [(0, nF40 // 3), (nF40 // 3, 2 * nF40 // 3), (2 * nF40 // 3, nF40)]
        quad_blocks = []
        for wt in QUAD_RATES:
            for wf in QUAD_SCALES:
                (c, s) = self._stmf_quadrature(log_e_fine_long, wt, wf)
                for (lo, hi) in bin_edges:
                    cb = c[:, lo:hi].mean(dim=1, keepdim=True)
                    sb = s[:, lo:hi].mean(dim=1, keepdim=True)
                    mb = torch.sqrt(c[:, lo:hi] ** 2 + s[:, lo:hi] ** 2 + 1e-12).mean(dim=1, keepdim=True)
                    quad_blocks.append(cb)
                    quad_blocks.append(sb)
                    quad_blocks.append(mb)
        quad_raw = torch.cat(quad_blocks, dim=1)
        add('quad_modbank', quad_raw)
        quad_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            quad_dyn_blocks.append(quad_raw - leaky_lowpass(quad_raw, tau_frames=tau))
        add('quad_modbank_dyn', torch.cat(quad_dyn_blocks, dim=1))
        acausal_blocks = []
        for tau in (60.0,):
            acausal_blocks.append(quad_raw - acausal_leaky_lowpass(quad_raw, tau_frames=tau))
        add('quad_modbank_acausal', torch.cat(acausal_blocks, dim=1))
        n_rates_q17 = len(QUAD_RATES)
        n_scales_q17 = len(QUAD_SCALES)
        n_bins_q17 = len(bin_edges)
        qr = quad_raw.view(quad_raw.shape[0], n_rates_q17, n_scales_q17, n_bins_q17, 3)
        mag_only = qr[..., 2]
        energy_per_rate_bin = mag_only.sum(dim=2)
        xrate_pairs = [(0, 2), (0, 3), (1, 3), (1, 2), (2, 3), (0, 1)]
        xrate_blocks = []
        for (i, j) in xrate_pairs:
            xrate_blocks.append(energy_per_rate_bin[:, i, :] * energy_per_rate_bin[:, j, :])
        xrate_raw = torch.cat(xrate_blocks, dim=1)
        xrate_log = torch.log1p(xrate_raw.clamp(min=0.0))
        add('quad_xrate_energy', xrate_log)
        xrate_dyn = []
        for tau in (4.5, 15.0):
            xrate_dyn.append(xrate_log - leaky_lowpass(xrate_log, tau_frames=tau))
        add('quad_xrate_energy_dyn', torch.cat(xrate_dyn, dim=1))
        gate = torch.sigmoid((best_strength - 0.3) * 8.0)
        gated_cochlea = log_e_long * gate
        unvoiced_cochlea = log_e_long * (1.0 - gate)
        add('cochlea_voiced_gated', torch.cat([gated_cochlea, unvoiced_cochlea], dim=1))
        gated_dyn = gated_cochlea - leaky_lowpass(gated_cochlea, tau_frames=15.0)
        add('cochlea_voiced_gated_dyn', gated_dyn)
        v_th = (0.15, 0.3, 0.6)
        f_th = (0.2, 0.4, 0.6)
        gate_signals = []
        for th in v_th:
            gate_signals.append(torch.sigmoid((best_strength - th) * 8.0))
        for th in f_th:
            gate_signals.append(torch.sigmoid((flat - th) * 10.0))
        gate_cat = torch.cat(gate_signals, dim=1)
        add('gate_signals', gate_cat)
        gate_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            gate_dyn_blocks.append(gate_cat - leaky_lowpass(gate_cat, tau_frames=tau))
        add('gate_signals_dyn', torch.cat(gate_dyn_blocks, dim=1))
        SLOW_RATES = (0.5, 1.0, 1.5, 2.0)
        SLOW_SCALES = (0.0, 0.25, -0.25, 0.5, -0.5)
        slow_blocks = []
        for wt in SLOW_RATES:
            for wf in SLOW_SCALES:
                (c, s) = self._stmf_quadrature(log_e_fine_long, wt, wf)
                for (lo, hi) in bin_edges:
                    cb = c[:, lo:hi].mean(dim=1, keepdim=True)
                    sb = s[:, lo:hi].mean(dim=1, keepdim=True)
                    mb = torch.sqrt(c[:, lo:hi] ** 2 + s[:, lo:hi] ** 2 + 1e-12).mean(dim=1, keepdim=True)
                    slow_blocks.append(cb)
                    slow_blocks.append(sb)
                    slow_blocks.append(mb)
        slow_raw = torch.cat(slow_blocks, dim=1)
        add('slow_quad_modbank', slow_raw)
        slow_dyn = slow_raw - leaky_lowpass(slow_raw, tau_frames=30.0)
        add('slow_quad_modbank_dyn', slow_dyn)
        feat = self._condition(groups)
        if getattr(self, 'scat_enabled', True):
            scatter = self._wavelet_scattering(x)
            feat = torch.cat([feat, scatter], dim=1)
        return feat.float().cpu().numpy()

def build_embedder(device: str='cuda', cond_mode: str='none', interact_set: str='none') -> WaveEmbedder:
    emb = WaveEmbedder(device=device)
    emb.cond_mode = cond_mode
    emb.interact_set = interact_set
    return emb
