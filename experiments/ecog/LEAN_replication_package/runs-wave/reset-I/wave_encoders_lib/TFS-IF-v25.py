"""TFS-IF v3: strong envelope substrate + analytic-signal TFS additions.

Paradigm: temporal fine structure / FM (phase, instantaneous frequency).

Architecture:
  BASE: multi-window log + cube-root cochleagrams (windows 1024, 2048, 4096) +
        multi-tau dynamics. Gives the ridge a clean envelope substrate.
  TFS:  analytic signal on the 4096 window via the Hilbert transform ->
        instantaneous frequency (IF), IF deviation from band fc, instantaneous
        bandwidth (std of IF), FM slope (up/down), envelope-IF coupling,
        inter-band phase locking (PLV, gap 1 and 2), and ENVELOPE-MODULATED
        TFS products (genuinely nonlinear in the bands).

Closed-form. Linear ridge is the only fit.
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


# ---------------------------------------------------------------------------
# Filterbank helpers
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_power(win, n_filters=32, f_min=80.0, f_max=6000.0, device="cuda"):
    """Real-valued gammatone-magnitude power bank for a given window length."""
    freqs = torch.fft.rfftfreq(win, d=1.0 / SR).to(device)
    fcs_np = erb_space(f_min, f_max, n_filters)
    fcs = torch.from_numpy(fcs_np.astype(np.float32)).to(device)
    bw = torch.from_numpy((1.019 * erb(fcs_np)).astype(np.float32)).to(device)
    H = (1.0 + ((freqs[:, None] - fcs[None, :]) / bw[None, :]) ** 2) ** (-4)
    H = H / H.max(dim=0, keepdim=True).values.clamp(min=1e-12)
    return H, fcs, bw


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


def gabor_envelope_time(x: torch.Tensor, rate_hz: float, fs: float = 30.0) -> torch.Tensor:
    """Magnitude of complex Gabor band-pass on a (T, C) signal along T.
    Captures modulation power at rate_hz on each channel of x."""
    period = fs / rate_hz
    sigma = 1.0 * period
    L = int(np.ceil(4 * sigma))
    t = torch.arange(-L, L + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * torch.cos(2 * np.pi * rate_hz * t / fs)
    sin_k = g * torch.sin(2 * np.pi * rate_hz * t / fs)
    cos_k = cos_k - cos_k.mean()
    sin_k = sin_k - sin_k.mean()
    cos_k = cos_k / (cos_k.norm() + 1e-12)
    sin_k = sin_k / (sin_k.norm() + 1e-12)
    C = x.shape[1]
    Llen = cos_k.numel()
    pad = Llen // 2
    xp = x.transpose(0, 1).unsqueeze(0)
    ck = cos_k.view(1, 1, -1).expand(C, 1, Llen).contiguous()
    sk = sin_k.view(1, 1, -1).expand(C, 1, Llen).contiguous()
    xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    c = torch.nn.functional.conv1d(xp_p, ck, groups=C)[..., :x.shape[0]]
    s = torch.nn.functional.conv1d(xp_p, sk, groups=C)[..., :x.shape[0]]
    env = torch.sqrt(c * c + s * s + 1e-12)
    return env.squeeze(0).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0

    N_BANDS = 32
    N_BANDS_ENV = 48
    F_MIN = 80.0
    F_MAX = 8000.0  # v25: extended from 6000 to 8000 Hz to cover sibilants
    WIN_LIST = (2048, 4096)  # 1024 window dropped in v12 to free budget for onsets
    PHASE_LOCK_HZ = 1500.0
    INNER = 2048
    TAUS = (15.0, 60.0)
    F0_CANDS = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0, 235.0, 275.0, 320.0, 380.0, 450.0, 540.0)
    N_HARMONICS = 6

    def __init__(self, device: str = "cuda"):
        self.device = device
        # --- envelope filterbanks per window (richer 48-band) ---
        self.fbs = []
        for w in self.WIN_LIST:
            H, fcs, bw = gammatone_power(w, n_filters=self.N_BANDS_ENV,
                                          f_min=self.F_MIN, f_max=self.F_MAX,
                                          device=device)
            self.fbs.append(H)
        # --- analytic-signal filterbank for 4096 window ---
        H4, fcs4, bw4 = gammatone_power(self.win, n_filters=self.N_BANDS,
                                          f_min=self.F_MIN, f_max=self.F_MAX,
                                          device=device)
        n_freqs = self.win // 2 + 1
        factor = torch.full((n_freqs,), 2.0, device=device)
        factor[0] = 1.0
        factor[-1] = 1.0
        self.H_analytic = H4 * factor[:, None]  # (n_freqs, n_bands)
        self.fcs = fcs4
        self.bw_norm = bw4
        self.log_fcs = torch.log(fcs4)
        self.n_freqs = n_freqs
        self.hann_w = {w: torch.hann_window(w, periodic=True, device=device) for w in self.WIN_LIST}
        # Pre-mask phase-locking range
        self.pl_mask = (fcs4 <= self.PHASE_LOCK_HZ).float()
        # Sample-index time vector for FM-slope regression (zero-mean)
        ti = torch.arange(self.INNER - 1, device=device, dtype=torch.float32)
        ti = ti - ti.mean()
        self.ti = ti
        self.ti_sq_sum = float((ti ** 2).sum().item())
        # F0 candidates and per-band harmonic-coincidence target table
        f0_cands = torch.tensor(self.F0_CANDS, dtype=torch.float32, device=device)
        self.f0_cands = f0_cands
        # For each (f0, band fc), compute the closest harmonic ratio and the deviation
        # in cents from that harmonic. We use this every call after IF is known.
        # Tolerance per band scales with the band's bandwidth (in Hz).
        self.harm_tol_hz = self.bw_norm.clone()  # (B,)

    @torch.no_grad()
    def _envelope_block(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-window log + cube-root cochleagrams + multi-tau, on (T, 4096) audio."""
        # x: (T, 4096) real
        blocks = []
        for w, fb in zip(self.WIN_LIST, self.fbs):
            if w == self.win:
                xw = x
            else:
                start = (self.win - w) // 2
                xw = x[:, start:start + w]
            hann = self.hann_w[w]
            Xf = torch.fft.rfft(xw * hann, n=w, dim=1)
            P = (Xf.real ** 2 + Xf.imag ** 2)  # (T, n_freqs)
            E = P @ fb  # (T, n_bands)
            logE = torch.log(E.clamp(min=1e-10))
            cubeE = torch.pow(E.clamp(min=1e-10), 1.0 / 3.0)
            blocks.append(logE)
            blocks.append(cubeE)
        envelope = torch.cat(blocks, dim=1)  # (T, n_windows * 2 * n_bands = 192)
        # Multi-tau deltas
        feats = [envelope]
        for tau in self.TAUS:
            feats.append(envelope - leaky_lowpass(envelope, tau_frames=tau))
        # Acausal (future) for long-window log only
        long_logE = blocks[-2]  # the 4096 log
        feats.append(acausal_leaky_lowpass(long_logE, tau_frames=60.0) - long_logE)
        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def _analytic_signal_batch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (Tb, 4096). Returns (Tb, n_bands, 4096) complex64."""
        hann = self.hann_w[self.win]
        Xf = torch.fft.rfft(x * hann, n=self.win, dim=1)  # (Tb, n_freqs)
        Yp = Xf.unsqueeze(1) * self.H_analytic.transpose(0, 1).unsqueeze(0)  # (Tb, n_bands, n_freqs)
        Z_full = torch.zeros((Yp.shape[0], self.N_BANDS, self.win),
                              dtype=torch.complex64, device=self.device)
        Z_full[..., :self.n_freqs] = Yp
        z = torch.fft.ifft(Z_full, n=self.win, dim=2)
        return z

    @torch.no_grad()
    def _tfs_block(self, x: torch.Tensor) -> torch.Tensor:
        """Compute TFS features per frame. x: (T, win). Returns (T, F_tfs)."""
        T = x.shape[0]
        Tb = 128
        out_list = []
        i0 = (self.win - self.INNER) // 2
        pl = self.pl_mask.unsqueeze(0)
        for s in range(0, T, Tb):
            e = min(s + Tb, T)
            z = self._analytic_signal_batch(x[s:e])  # (Tb, B, win) complex
            zc = z[..., i0:i0 + self.INNER]  # (Tb, B, INNER)
            amp = zc.abs()  # (Tb, B, INNER)
            E = amp.mean(dim=2)  # (Tb, B)

            # IF via phase-difference
            zr = zc[..., 1:] * zc[..., :-1].conj()
            ifreq = torch.angle(zr) * (SR / (2.0 * np.pi))  # (Tb, B, INNER-1) Hz
            w = amp[..., 1:]
            wsum = w.sum(dim=2).clamp(min=1e-10)
            if_mean = (w * ifreq).sum(dim=2) / wsum
            df = ifreq - if_mean.unsqueeze(2)
            if_var = (w * df ** 2).sum(dim=2) / wsum
            if_std = torch.sqrt(if_var.clamp(min=0.0) + 1e-6)

            # IF deviation from filter center, normalized
            if_dev = if_mean - self.fcs.unsqueeze(0)
            if_dev_norm = if_dev / self.bw_norm.unsqueeze(0)
            log_if = torch.log(if_mean.clamp(min=20.0))

            # FM slope: weighted regression of ifreq on ti
            num = (w * df * self.ti).sum(dim=2)
            denom = (w * self.ti ** 2).sum(dim=2).clamp(min=1e-6)
            fm_slope = num / denom  # Hz/sample
            fm_hz_s = fm_slope * SR
            fm_feat = torch.sign(fm_hz_s) * torch.log1p(fm_hz_s.abs())
            fm_up = fm_feat.clamp(min=0.0)
            fm_dn = (-fm_feat).clamp(min=0.0)

            # Envelope-IF coupling (within-window correlation)
            env_mid = amp[..., :-1]
            env_dm = env_mid - env_mid.mean(dim=2, keepdim=True)
            if_dm = df
            cov_env_if = (env_dm * if_dm).mean(dim=2)
            env_std = env_mid.std(dim=2).clamp(min=1e-3)
            corr_env_if = (cov_env_if / (env_std * if_std + 1e-3)).clamp(-1.0, 1.0)

            # Inter-band PLV (gap 1 and gap 2)
            unit = zc / zc.abs().clamp(min=1e-8)
            plv1 = (unit[:, :-1, :] * unit[:, 1:, :].conj()).mean(dim=2).abs()
            plv2 = (unit[:, :-2, :] * unit[:, 2:, :].conj()).mean(dim=2).abs()
            plv1 = torch.cat([plv1, plv1[:, -1:]], dim=1)
            plv2 = torch.cat([plv2, plv2[:, -2:]], dim=1)

            # Apply phase-locking mask
            log_if_pl = log_if * pl
            if_dev_norm_pl = if_dev_norm * pl
            if_std_pl = if_std * pl
            fm_up_pl = fm_up * pl
            fm_dn_pl = fm_dn * pl
            corr_env_if_pl = corr_env_if * pl
            plv1_pl = plv1 * pl
            plv2_pl = plv2 * pl

            # === FAST AM POWER per band at specific rates ===
            # Compute envelope FFT inside inner window: |sum env(t) e^{-j 2pi r t/SR}|.
            # Rates 32, 64, 128, 256 Hz are ABOVE what ridge-FIR (30Hz, 4 taps) can reach,
            # so they are genuinely nonlinear-in-the-bands modulation features.
            INNER = self.INNER
            env_centered = amp - amp.mean(dim=2, keepdim=True)  # (Tb, B, INNER)
            tt = torch.arange(INNER, device=self.device, dtype=torch.float32) / SR  # seconds
            am_rates = torch.tensor([32.0, 64.0, 128.0, 256.0], device=self.device)
            # phases: (R, INNER)
            phases = 2 * np.pi * am_rates.unsqueeze(1) * tt.unsqueeze(0)
            cos_b = torch.cos(phases)
            sin_b = torch.sin(phases)
            # (Tb, B, INNER) x (R, INNER)^T  ->  (Tb, B, R)
            am_cos = env_centered @ cos_b.T
            am_sin = env_centered @ sin_b.T
            am_pwr = torch.log(am_cos ** 2 + am_sin ** 2 + 1e-10)  # (Tb, B, R)
            am_pwr_flat = am_pwr.reshape(am_pwr.shape[0], -1)  # (Tb, B*R = 128)

            # === envelope shape moments per band ===
            amp_centered = amp - amp.mean(dim=2, keepdim=True)
            amp_std_in = amp.std(dim=2).clamp(min=1e-6)
            env_skew = (amp_centered ** 3).mean(dim=2) / (amp_std_in ** 3 + 1e-6)
            env_kurt = (amp_centered ** 4).mean(dim=2) / (amp_std_in ** 4 + 1e-6)
            env_skew = env_skew.clamp(-10, 10)
            env_kurt = env_kurt.clamp(0, 50)

            # === phase-amplitude coupling (PAC): low-band phase x high-band envelope ===
            # 8 low-freq bands (0..7) phases ⊗ 8 mid/high bands envelopes (16..23, etc.).
            low_idx = torch.arange(0, 8, device=self.device)
            hi_idx = torch.arange(16, 24, device=self.device)
            phi_low = zc[:, low_idx, :] / zc[:, low_idx, :].abs().clamp(min=1e-8)  # (Tb, 8, INNER)
            env_hi = amp[:, hi_idx, :]  # (Tb, 8, INNER)
            # PAC[l, h] = | sum_t env_hi[h, t] * phi_low[l, t] |  /  sum_t env_hi[h, t]
            # Implement via batched matmul-like reduction
            env_hi_norm = env_hi.sum(dim=2).clamp(min=1e-8)  # (Tb, 8)
            # outer over (l, h): (Tb, 8, 8) complex
            num_complex = torch.einsum('tli,thi->tlh', phi_low, env_hi.to(torch.complex64))
            pac = num_complex.abs() / env_hi_norm.unsqueeze(1)  # (Tb, 8, 8)
            pac_flat = pac.reshape(pac.shape[0], -1)  # (Tb, 64)

            # === F0 track from IF using harmonic coincidence ===
            # Per band b and per F0 candidate f0, harmonic distance from nearest
            # multiple of f0 to IF_b. Sum across phase-locked bands weighted by E.
            f0_cands = self.f0_cands.view(1, 1, -1)
            ifm = if_mean.unsqueeze(2)
            ratios = ifm / f0_cands
            nearest_k = torch.round(ratios).clamp(min=1.0, max=20.0)
            harm_predict = nearest_k * f0_cands
            dev_hz = (ifm - harm_predict).abs()
            tol = self.harm_tol_hz.view(1, -1, 1)
            coincidence = torch.exp(-(dev_hz / tol) ** 2)  # (Tb, B, F0)
            env_w = (E * self.pl_mask.unsqueeze(0)).unsqueeze(2)
            harm_pitch = (coincidence * env_w).sum(dim=1) / (env_w.sum(dim=1) + 1e-10)
            # F0 SCALAR: argmax F0 (soft, weighted)
            sm_p = torch.softmax(harm_pitch * 8.0, dim=1)  # (Tb, F0)
            f0_est = (sm_p * self.f0_cands.view(1, -1)).sum(dim=1, keepdim=True)
            log_f0 = torch.log(f0_est.clamp(min=20.0))
            f0_strength = harm_pitch.max(dim=1, keepdim=True).values

            # === Voicing score ===
            # Mean PLV across phase-locked adjacent pairs weighted by envelope.
            voicing = (plv1_pl * E * self.pl_mask.unsqueeze(0)).sum(dim=1, keepdim=True) / \
                       ((E * self.pl_mask.unsqueeze(0)).sum(dim=1, keepdim=True) + 1e-10)

            block = torch.cat([
                log_if_pl,        # 32
                if_dev_norm_pl,   # 32
                if_std_pl,        # 32
                fm_up_pl,         # 32
                fm_dn_pl,         # 32
                plv1_pl,          # 32
                plv2_pl,          # 32
                env_skew,         # 32  envelope shape skew per band
                env_kurt,         # 32  envelope shape kurt per band
                am_pwr_flat,      # 128 fast AM power per band at 32/64/128/256Hz
                pac_flat,         # 64  PAC: low-band phase x high-band envelope
                log_f0,           # 1   F0 estimate (log)
                f0_strength,      # 1   pitch salience
                harm_pitch,       # 12  F0 candidate distribution
                voicing,          # 1   voicing index
            ], dim=1)  # 9*32 + 128 + 64 + 15 = 495
            out_list.append(block.cpu())
            del z, zc, amp, zr, ifreq, w, env_mid, unit, plv1, plv2
        tfs = torch.cat(out_list, dim=0).to(self.device)
        # Multi-tau on TFS
        feats = [tfs]
        for tau in self.TAUS:
            feats.append(tfs - leaky_lowpass(tfs, tau_frames=tau))
        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        env = self._envelope_block(x)   # base substrate (192 base + multi-tau + acausal)
        tfs = self._tfs_block(x)        # TFS additions

        # === Future/past delta on the LONG-window logE (48 bands) ===
        # Provide explicit per-band lookahead and lookback for the ridge — taus
        # alone smear over time; explicit lag shifts give pointed temporal context.
        # _envelope_block packs long-window log first, then cube, then per-window.
        # Long-window log is at offset N_BANDS_ENV*2 from the start (we packed
        # (logE, cubeE) per window in WIN_LIST order; the 4096 window is last).
        # The slice indices: long_log starts at 2 * 2 * N_BANDS_ENV (= 192)? Actually
        # blocks order: (1024 log, 1024 cube, 2048 log, 2048 cube, 4096 log, 4096 cube).
        # So 4096 log starts at 4 * N_BANDS_ENV.
        # WIN_LIST order: (2048 log, 2048 cube, 4096 log, 4096 cube). 4096 log
        # starts at offset 2 * N_BANDS_ENV.
        long_logE = env[:, 2 * self.N_BANDS_ENV:3 * self.N_BANDS_ENV]  # (T, 48)
        deltas = []
        for k in (3, 6, 12):
            fut = torch.cat([long_logE[k:], long_logE[-1:].expand(k, -1)], dim=0)
            past = torch.cat([long_logE[:1].expand(k, -1), long_logE[:-k]], dim=0)
            deltas.append(fut - long_logE)
            deltas.append(long_logE - past)
        ctx = torch.cat(deltas, dim=1)  # (T, 6 * 48 = 288)

        # === Inter-frame modulation filterbank on long-window log envelope ===
        # Gabor band-pass envelopes at syllabic/phonetic rates.
        mod_feats = []
        for rate in (2.0, 4.0, 6.0, 10.0):
            mod_feats.append(gabor_envelope_time(long_logE, rate))
        mod_filt = torch.cat(mod_feats, dim=1)  # (T, 4 * 48 = 192)

        # === Formant tracking on long-window log cochleagram ===
        # Soft-argmax position of energy peak in 3 frequency ranges (F1/F2/F3).
        nF = self.N_BANDS_ENV
        f_ranges = [(0, nF // 3), (nF // 3, 2 * nF // 3), (2 * nF // 3, nF)]
        ch_idx = torch.arange(nF, device=self.device, dtype=torch.float32)
        formant_feats = []
        for lo, hi in f_ranges:
            band = long_logE[:, lo:hi]
            w = torch.softmax(band * 2.0, dim=1)
            loc = (w * ch_idx[lo:hi]).sum(dim=1, keepdim=True)
            amp_b = (w * band).sum(dim=1, keepdim=True)
            formant_feats.append(loc)
            formant_feats.append(amp_b)
        formants = torch.cat(formant_feats, dim=1)  # (T, 6) F1/F2/F3 loc + amp
        formant_d = torch.zeros_like(formants)
        formant_d[1:] = formants[1:] - formants[:-1]
        # Multi-tau on formants for prosodic context
        formant_taus = []
        for tau in (15.0, 60.0):
            formant_taus.append(formants - leaky_lowpass(formants, tau_frames=tau))
        formant_taus_t = torch.cat(formant_taus, dim=1)  # (T, 12)

        # Scattering level 2 dropped in v8b to keep ridge tractable.
        scat2 = long_logE[:, :0]  # empty

        # === Lateral inhibition (DoG across channel axis), rectified ===
        # Sharpens spectral peaks. The rectification is the nonlinear step that
        # makes this not just a linear remix of bands.
        def dog_chan(x, sigma_c=1.0, sigma_s=4.0):
            k = int(np.ceil(3 * sigma_s))
            coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
            g_c = torch.exp(-coords ** 2 / (2 * sigma_c ** 2)); g_c = g_c / g_c.sum()
            g_s = torch.exp(-coords ** 2 / (2 * sigma_s ** 2)); g_s = g_s / g_s.sum()
            dog = (g_c - g_s).view(1, 1, -1)
            return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)
        li = dog_chan(long_logE, sigma_c=1.0, sigma_s=4.0)
        li_rect_pos = li.clamp(min=0.0)
        li_rect_neg = (-li).clamp(min=0.0)  # negative DoG (anti-formant) — useful for nulls
        # 2 × 48 = 96

        # === Cross-channel envelope statistics per frame ===
        # std and skew across channels — captures spectral spread vs peakiness.
        lE_mean = long_logE.mean(dim=1, keepdim=True)
        lE_std = long_logE.std(dim=1, keepdim=True)
        lE_centered = long_logE - lE_mean
        lE_skew = (lE_centered ** 3).mean(dim=1, keepdim=True) / (lE_std ** 3 + 1e-6)
        lE_kurt = (lE_centered ** 4).mean(dim=1, keepdim=True) / (lE_std ** 4 + 1e-6)
        spec_shape = torch.cat([lE_std, lE_skew.clamp(-10, 10), lE_kurt.clamp(0, 50)],
                                dim=1)  # (T, 3)

        # === Per-band ONSET / OFFSET RECTIFIERS (NEW in v12) ===
        # The biological correlate of STG onset responses: half-wave rectified
        # temporal derivative of log envelope, per band. ONSET = HWR(+d/dt logE),
        # OFFSET = HWR(-d/dt logE). The half-wave rectification is genuinely
        # nonlinear and is NOT a linear function of the band envelopes the ridge
        # already has — sign-selective rectification can't be synthesized by any
        # linear combination of log envelopes.
        #
        # Two temporal scales:
        #   k=1  (33 ms): fast transients (stops, plosives, voice onsets)
        #   k=3  (100 ms): syllable-rate envelope flux
        # On the 4096 log envelope (48 bands).
        onset_feats = []
        for k in (1, 3):
            past_k = torch.cat([long_logE[:1].expand(k, -1), long_logE[:-k]], dim=0)
            d = long_logE - past_k  # (T, 48), positive = onset, negative = offset
            onset_feats.append(d.clamp(min=0.0))   # ONSET
            onset_feats.append((-d).clamp(min=0.0))  # OFFSET
        onsets = torch.cat(onset_feats, dim=1)  # (T, 4*48=192)

        # === Cross-band SPECTRAL FLUX (NEW in v12) ===
        # Sum of HWR(+) onsets across bands per frame — a single SCALAR per scale
        # that fires when broadband energy rises. Captures global onset events
        # (sentence onsets, breath bursts) that depend on coactivation across
        # multiple bands — not derivable from any single band's onset linearly.
        # Use the same 2 scales; sum gives 2 features, max gives 2 more.
        flux_feats = []
        for k in (1, 3):
            past_k = torch.cat([long_logE[:1].expand(k, -1), long_logE[:-k]], dim=0)
            d = long_logE - past_k
            flux_feats.append(d.clamp(min=0.0).sum(dim=1, keepdim=True))
            flux_feats.append(d.clamp(min=0.0).max(dim=1, keepdim=True).values)
        flux = torch.cat(flux_feats, dim=1)  # (T, 4)

        # === Pairwise PRODUCTS of macro-band log envelopes (NEW in v14) ===
        # The ridge sees log envelopes per band already. log(env_a)*log(env_b) is
        # genuinely QUADRATIC in the log envelopes (not a linear function), so
        # the ridge cannot synthesize these from the existing per-band log feats.
        # Captures co-activation gating: high only when BOTH macro-bands are
        # simultaneously high (harmonic structure, multi-formant resonance,
        # broadband sibilance). 8 macro-bands of 6 fine bands each → 28 pairs.
        nMacro = 8
        per_macro = self.N_BANDS_ENV // nMacro  # 6
        macro = long_logE.view(long_logE.shape[0], nMacro, per_macro).mean(dim=2)  # (T, 8)
        # All-pairs (i, j), i < j
        iu = torch.triu_indices(nMacro, nMacro, offset=1, device=long_logE.device)
        prod_pairs = macro[:, iu[0]] * macro[:, iu[1]]  # (T, 28)
        # One multi-tau (60) for slower context on these products.
        prod_tau = prod_pairs - leaky_lowpass(prod_pairs, tau_frames=60.0)
        prod_block = torch.cat([prod_pairs, prod_tau], dim=1)  # (T, 56)

        feat = torch.cat([env, tfs, ctx, mod_filt, formants, formant_d,
                           formant_taus_t,
                           scat2, li_rect_pos, li_rect_neg, spec_shape,
                           onsets, flux, prod_block],
                          dim=1).float().cpu().numpy()
        return feat


model_shorthand_name = "TFS-IF-v25"
model_description = (
    "v25 = v14 with F_MAX extended from 6000 to 8000 Hz. Same number of bands "
    "(48 envelope, 32 TFS); spectrum re-stretched to cover sibilant frequencies "
    "(/s/, /sh/, /f/ have energy 4-8 kHz). No feature-count change; pure "
    "filterbank re-tuning. v14=0.0642 (best), v24=0.0638 (crest factor regressed)."
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
        print("cuda not available -- falling back to cpu")
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
