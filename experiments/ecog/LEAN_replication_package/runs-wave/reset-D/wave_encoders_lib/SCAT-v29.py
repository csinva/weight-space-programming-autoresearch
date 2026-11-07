"""
Wavelet scattering transform front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

Iter-3 design: richer scattering with multi-window phi, transient/onset features,
and extended modulation rate coverage.

  * 1st-order log-scattering at 4 averaging windows (512/1024/2048/4096):
    multi-scale phi (lowpass averaging at different time scales). 4 * 84 ch.
  * Within-window 2nd-order AM scattering at long window: analytic envelope ->
    AM spectrum (6..1024 Hz, Q=1). 84 * n_lambda2 ch.
  * Across-frame slow AM scattering on long-window S1: 0.75..14 Hz at 30 Hz frame
    rate -> 84 * 6 ch.
  * TRANSIENT/ONSET features on long-window S1: HWR(+) and HWR(-) of (S1[t+k] -
    S1[t]) at multiple lags k. Captures abrupt envelope changes that smooth
    scattering averages out.
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_D"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Morlet wavelet filterbanks (freq domain, constant-Q Gaussian magnitude)
# ---------------------------------------------------------------------------

def morlet_freq_bank(n_freqs: int, freq_step: float, q: float, n_octaves: int,
                     f_max: float, q_floor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    n_filters = int(round(q * n_octaves))
    js = np.arange(n_filters, dtype=np.float64)
    xi = f_max * 2.0 ** (-js / q)
    sigma_f = np.maximum(xi / q, q_floor)
    freqs = np.arange(n_freqs, dtype=np.float64) * freq_step
    H = np.exp(-((freqs[None, :] - xi[:, None]) ** 2) /
               (2.0 * sigma_f[:, None] ** 2))
    return H.astype(np.float32), xi.astype(np.float32)


def time_morlet_kernels(rates_hz, fs=30.0, widths_periods=1.0):
    kernels = []
    for f in rates_hz:
        period = fs / f
        sigma = widths_periods * period
        L = int(np.ceil(4 * sigma))
        t = np.arange(-L, L + 1)
        env = np.exp(-(t ** 2) / (2 * sigma ** 2))
        cos_k = env * np.cos(2 * np.pi * f * t / fs)
        sin_k = env * np.sin(2 * np.pi * f * t / fs)
        cos_k -= cos_k.mean()
        sin_k -= sin_k.mean()
        n = np.sqrt((cos_k ** 2 + sin_k ** 2).sum()) + 1e-12
        kernels.append((cos_k.astype(np.float32) / n, sin_k.astype(np.float32) / n))
    return kernels


def apply_time_modulus_bank(x: torch.Tensor, kernels) -> torch.Tensor:
    T, C = x.shape
    device = x.device
    outs = []
    xp = x.transpose(0, 1).unsqueeze(0)
    for cos_k, sin_k in kernels:
        ck = torch.from_numpy(cos_k).to(device).view(1, 1, -1).expand(C, 1, -1).contiguous()
        sk = torch.from_numpy(sin_k).to(device).view(1, 1, -1).expand(C, 1, -1).contiguous()
        L = ck.shape[-1]
        pad = L // 2
        xpp = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
        c = torch.nn.functional.conv1d(xpp, ck, groups=C)[..., :T]
        s = torch.nn.functional.conv1d(xpp, sk, groups=C)[..., :T]
        env = torch.sqrt(c * c + s * s + 1e-12)
        outs.append(env.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0

    WIN_LIST = (1024, 4096)  # v17: drop 2048 (mostly redundant with 1024+4096)
    S2_WITHIN_GROUPS = 24  # collapse lambda1 axis of within-window 2nd order

    Q1 = 12
    N_OCT1 = 7
    F_MAX = 8000.0  # revert: 12000 lost low-freq pitch coverage

    Q2 = 2.0
    AM_F_MAX = 1024.0
    AM_F_MIN_TARGET = 6.0

    TIME_MOD_RATES = (1.5, 3.0, 6.0, 10.0, 14.0)  # v29: drop 0.75 Hz to make room for acausal env_cum

    # Transient / onset lags (frames at 30 Hz)
    ONSET_LAGS = (2, 4, 8)  # v17: 3 lags (drop k=1, mostly noise)

    # Joint-TF 2D Morlet kernels (rate_hz at 30 Hz frame, scale cyc/channel)
    JOINT_TF_KERNELS = (
        (2.0, 0.0), (4.0, 0.0), (8.0, 0.0), (12.0, 0.0),
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (8.0, 0.25), (8.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.5),  (8.0, -0.5),
        (0.0, 0.25), (0.0, 0.5),
    )

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.W = self.win
        freq_step_long = SR / self.W

        # 1st-order Morlet bank per window
        self.H1_sq_per_win = {}
        for w in self.WIN_LIST:
            n_f = w // 2 + 1
            step = SR / w
            H, xi = morlet_freq_bank(n_f, step, self.Q1, self.N_OCT1, self.F_MAX,
                                     q_floor=max(8.0, step))
            self.H1_sq_per_win[w] = torch.from_numpy(H * H).to(device)
            self.n_lambda1 = H.shape[0]
            self.xi1 = xi
        self.hanns = {w: torch.hann_window(w, periodic=True, device=device) for w in self.WIN_LIST}

        # Analytic Morlet (long window) for within-window 2nd order
        n_f = self.W // 2 + 1
        step = freq_step_long
        H1_long, _ = morlet_freq_bank(n_f, step, self.Q1, self.N_OCT1, self.F_MAX, q_floor=8.0)
        Hf = np.zeros((self.n_lambda1, self.W), dtype=np.complex64)
        Hf[:, :n_f] = 2.0 * H1_long
        Hf[:, 0] = H1_long[:, 0]
        if self.W % 2 == 0:
            Hf[:, n_f - 1] = H1_long[:, n_f - 1]
        self.Hf = torch.from_numpy(Hf).to(device)
        self.n_freqs_long = n_f

        # 2nd-order AM bank on env's rfft
        n_oct2 = max(1, int(np.floor(np.log2(self.AM_F_MAX / self.AM_F_MIN_TARGET))))
        H2, _ = morlet_freq_bank(n_f, step, self.Q2, n_oct2, self.AM_F_MAX, q_floor=4.0)
        self.n_lambda2 = H2.shape[0]
        self.H2_sq = torch.from_numpy(H2 * H2).to(device)

        # across-frame Morlet
        self.time_kernels = time_morlet_kernels(self.TIME_MOD_RATES, fs=30.0,
                                                widths_periods=1.0)
        # 3rd-order across-frame Morlet (fewer, broader rates)
        self.s3_kernels = time_morlet_kernels((1.5, 4.0, 8.0), fs=30.0,
                                              widths_periods=1.0)

        # F0 candidates for harmonic stacking
        self._f0_cands = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0,
                          230.0, 270.0, 320.0, 380.0, 450.0)
        self._f0_cand_t = torch.tensor(self._f0_cands, dtype=torch.float32, device=device)

        # Long-window rfft frequency grid (Hz) for spectral moments etc.
        self._long_freqs_t = torch.from_numpy(
            np.fft.rfftfreq(self.W, d=1.0 / SR).astype(np.float32)).to(device)

        # Short-window analytic Morlet (W_short=1024) for fast within-window 2nd order
        self.W_short = 1024
        n_f_s = self.W_short // 2 + 1
        step_s = SR / self.W_short
        H1_short, _ = morlet_freq_bank(n_f_s, step_s, self.Q1, self.N_OCT1, self.F_MAX, q_floor=step_s)
        Hf_s = np.zeros((self.n_lambda1, self.W_short), dtype=np.complex64)
        Hf_s[:, :n_f_s] = 2.0 * H1_short
        Hf_s[:, 0] = H1_short[:, 0]
        if self.W_short % 2 == 0:
            Hf_s[:, n_f_s - 1] = H1_short[:, n_f_s - 1]
        self.Hf_short = torch.from_numpy(Hf_s).to(device)
        # 2nd-order AM bank on short envelope rfft. Step ≈ 43 Hz. Cover up to 2048Hz.
        AM_short_max = 2048.0
        n_oct2_s = max(1, int(np.floor(np.log2(AM_short_max / 60.0))))
        H2_short, _ = morlet_freq_bank(n_f_s, step_s, q=2.0, n_octaves=n_oct2_s,
                                       f_max=AM_short_max, q_floor=step_s)
        self.n_lambda2_short = H2_short.shape[0]
        self.H2_short_sq = torch.from_numpy(H2_short * H2_short).to(device)
        self.n_freqs_short = n_f_s

        # Multi-Q (Q=4) 1st-order bank at long window
        H1_Q4, _ = morlet_freq_bank(n_f, step, q=4.0, n_octaves=7,
                                    f_max=self.F_MAX, q_floor=24.0)
        self.H1_Q4_sq = torch.from_numpy(H1_Q4 * H1_Q4).to(device)
        self.n_lambda1_Q4 = H1_Q4.shape[0]

        # High-Q (Q=24) 1st-order bank at long window
        H1_Q24, _ = morlet_freq_bank(n_f, step, q=24.0, n_octaves=6,
                                     f_max=self.F_MAX, q_floor=4.0)
        self.H1_Q24_sq = torch.from_numpy(H1_Q24 * H1_Q24).to(device)
        self.n_lambda1_Q24 = H1_Q24.shape[0]

    def _harmonic_stack(self, power: torch.Tensor, n_harmonics: int = 10) -> torch.Tensor:
        """power: (T, n_freqs). Long-window rfft power. Return (T, n_f0) where each
        column is mean log-power at first n_harmonics multiples of f0."""
        n_freqs = power.shape[1]
        freq_res = SR / self.W
        log_p = torch.log(power.clamp(min=1e-10))
        total = log_p.mean(dim=1, keepdim=True)
        out = []
        for f0 in self._f0_cands:
            ks = torch.arange(1, n_harmonics + 1, device=power.device, dtype=torch.float32)
            bins = (f0 * ks / freq_res).long().clamp(max=n_freqs - 1)
            sal = log_p[:, bins].mean(dim=1, keepdim=True) - total
            out.append(sal)
        return torch.cat(out, dim=1)

    def _dog_bands(self, x: torch.Tensor, sigma_c: float, sigma_s: float) -> torch.Tensor:
        """DoG (lateral inhibition) along band axis of (T, C)."""
        device = x.device
        k = int(np.ceil(3 * sigma_s))
        coords = torch.arange(-k, k + 1, device=device, dtype=x.dtype)
        g_c = torch.exp(-(coords ** 2) / (2 * sigma_c ** 2)); g_c = g_c / g_c.sum()
        g_s = torch.exp(-(coords ** 2) / (2 * sigma_s ** 2)); g_s = g_s / g_s.sum()
        dog = (g_c - g_s).view(1, 1, -1)
        return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)

    def _acausal_lowpass(self, x: torch.Tensor, tau_frames: float) -> torch.Tensor:
        return torch.flip(self._causal_lowpass(torch.flip(x, dims=[0]), tau_frames),
                          dims=[0])

    def _causal_lowpass(self, x: torch.Tensor, tau_frames: float) -> torch.Tensor:
        """Causal leaky-integrator lowpass via conv1d."""
        alpha = 1.0 / (1.0 + tau_frames)
        K = max(int(6 * tau_frames) + 1, 4)
        t = torch.arange(K, device=x.device, dtype=x.dtype)
        kernel = alpha * (1.0 - alpha) ** t
        xp = x.transpose(0, 1).unsqueeze(0)
        xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
        kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
        y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
        return y.squeeze(0).transpose(0, 1).contiguous()

    def _build_joint_kernel(self, rate_hz: float, scale_cpc: float,
                            fs: float = 30.0, sigma_periods: float = 1.0):
        period_t = fs / max(abs(rate_hz), 0.5)
        sigma_t = sigma_periods * period_t
        Lt = max(2, int(np.ceil(3 * sigma_t)))
        period_f = 1.0 / max(abs(scale_cpc), 0.05)
        sigma_f = sigma_periods * period_f
        Lf = max(2, int(np.ceil(3 * sigma_f)))
        t = np.arange(-Lt, Lt + 1)
        f = np.arange(-Lf, Lf + 1)
        env = np.exp(-((t[:, None] ** 2) / (2 * sigma_t ** 2) +
                       (f[None, :] ** 2) / (2 * sigma_f ** 2)))
        phase = 2 * np.pi * rate_hz * t[:, None] / fs + 2 * np.pi * scale_cpc * f[None, :]
        cos_k = env * np.cos(phase)
        sin_k = env * np.sin(phase)
        cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
        n = np.sqrt((cos_k ** 2 + sin_k ** 2).sum()) + 1e-12
        ck = torch.from_numpy((cos_k / n).astype(np.float32)).to(self.device).view(1, 1, -1, 2 * Lf + 1)
        sk = torch.from_numpy((sin_k / n).astype(np.float32)).to(self.device).view(1, 1, -1, 2 * Lf + 1)
        return ck, sk, Lt, Lf

    def _S1(self, x: torch.Tensor, w: int) -> torch.Tensor:
        """Hann-windowed 1st-order log-scattering at window size `w`."""
        if w == self.W:
            xw = x
        else:
            start = (self.W - w) // 2
            xw = x[:, start:start + w]
        Xh = torch.fft.rfft(xw * self.hanns[w].unsqueeze(0), n=w)
        power = (Xh.conj() * Xh).real
        E = power @ self.H1_sq_per_win[w].T
        return torch.log(E + 1e-8), Xh  # also return rfft for re-use if w==W

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        T = x.shape[0]
        W = self.W
        feats = []

        # ---------- 1st-order at multiple averaging windows ----------
        S1_long = None
        Xh_long = None
        for w in self.WIN_LIST:
            S1, Xh = self._S1(x, w)
            feats.append(S1)
            if w == W:
                S1_long = S1
                Xh_long = Xh

        # ---------- WITHIN-WINDOW 2nd-order AM + PHASE-DERIVATIVE (IF) SCATTERING ----------
        # NEW IN v17: For each band, in addition to the amplitude-envelope 2nd-order
        # scattering, compute the INSTANTANEOUS FREQUENCY (IF) signal:
        #     IF[tau] = imag( log( y[tau] * conj(y[tau-1]) ) )
        # weighted by envelope (so only voiced/loud regions contribute). The
        # envelope-weighted mean / variance of IF is a closed-form pitch/formant
        # tracker per band. The 2nd-order Morlet bank applied to (env * IF) gives
        # the FM-RATE SPECTRUM per band — a representation of how the carrier
        # within a band wobbles (vibrato, pitch motion, formant glides) that
        # magnitude-only scattering destroys at first order.
        X_pad = torch.zeros((T, W), dtype=torch.complex64, device=self.device)
        X_pad[:, :self.n_freqs_long] = Xh_long
        BATCH = 8  # smaller batch -- we now keep env-weighted IF stats per band
        S2_within_chunks = []
        if_mean_per_band = []
        if_std_per_band = []
        if_fm_rate_chunks = []
        env_skew_per_band = []
        env_kurt_per_band = []
        log_env_skew_per_band = []
        log_env_kurt_per_band = []
        for j0 in range(0, self.n_lambda1, BATCH):
            j1 = min(j0 + BATCH, self.n_lambda1)
            Hf_b = self.Hf[j0:j1].unsqueeze(0)
            Y = X_pad.unsqueeze(1) * Hf_b
            analytic = torch.fft.ifft(Y, n=W, dim=2)                # (T, B, W) complex
            env = analytic.abs()                                     # (T, B, W)
            ENV = torch.fft.rfft(env, n=W, dim=2)
            pE = (ENV.conj() * ENV).real
            E2 = pE @ self.H2_sq.T
            S2_within_chunks.append(torch.log(E2 + 1e-8))

            # v25: per-band ENVELOPE CUMULANTS (skewness, kurtosis) within
            # window. Captures BURSTINESS vs sustained character of each band's
            # envelope — sharp plosives have high skew/kurt; sustained vowels
            # have low. Magnitude scattering (mean of |env|^p with lowpass)
            # collapses this signal; raw cumulants preserve it.
            env_m = env.mean(dim=2, keepdim=True)
            ec = env - env_m
            ev = (ec ** 2).mean(dim=2)
            es = torch.sqrt(ev + 1e-12)
            esk = ((ec ** 3).mean(dim=2) / (es ** 3 + 1e-12)).clamp(-10, 10)
            ek = ((ec ** 4).mean(dim=2) / (ev ** 2 + 1e-12)).clamp(0, 50)
            env_skew_per_band.append(esk)
            env_kurt_per_band.append(ek)
            # v26: LOG-envelope cumulants (perceptual compression).
            log_env = torch.log(env + 1e-6)
            le_m = log_env.mean(dim=2, keepdim=True)
            lec = log_env - le_m
            lev = (lec ** 2).mean(dim=2)
            les = torch.sqrt(lev + 1e-12)
            lesk = ((lec ** 3).mean(dim=2) / (les ** 3 + 1e-12)).clamp(-10, 10)
            lek = ((lec ** 4).mean(dim=2) / (lev ** 2 + 1e-12)).clamp(0, 50)
            log_env_skew_per_band.append(lesk)
            log_env_kurt_per_band.append(lek)

            # Instantaneous-frequency signal per band (radians/sample), via
            # phase derivative of the analytic signal. Use the analytic ratio
            # trick — no atan / unwrap needed: phi'[n] = imag(log(y[n] conj(y[n-1])))
            #   = imag(log(y[n])) - imag(log(y[n-1])), modulo 2pi
            # We compute it as the imag of a normalized product so it stays in
            # the principal branch.
            y_curr = analytic[:, :, 1:]
            y_prev = analytic[:, :, :-1]
            cross = y_curr * y_prev.conj()
            # angle of cross is phi'[n]; we want a stable nonlinear measure
            # weighted by local envelope amplitude squared (to suppress noise
            # in silent regions). Use the imag(cross)/(|y_curr|*|y_prev|+eps)
            # which is sin(phi') — bounded in [-1,1] and emphasizes coherent phase.
            denom = y_curr.abs() * y_prev.abs() + 1e-8
            sin_phi = cross.imag / denom                            # (T, B, W-1) in [-1,1]
            cos_phi = cross.real / denom                            # for std/skew via complex mean
            # Envelope weight (squared envelope at midpoint)
            w_env = (y_curr.abs() * y_prev.abs())                   # (T, B, W-1)
            w_sum = w_env.sum(dim=2) + 1e-8                         # (T, B)
            # Envelope-weighted mean of (sin_phi, cos_phi) -> resultant vector;
            # its angle is the dominant carrier-IF; its magnitude is phase
            # coherence ("voicing strength" within the band).
            mean_sin = (w_env * sin_phi).sum(dim=2) / w_sum         # (T, B)
            mean_cos = (w_env * cos_phi).sum(dim=2) / w_sum
            if_resultant_mag = torch.sqrt(mean_sin ** 2 + mean_cos ** 2 + 1e-12)
            # IF center via atan2 (small num -> stable)
            if_center = torch.atan2(mean_sin, mean_cos)
            if_mean_per_band.append(torch.stack([if_center, if_resultant_mag], dim=-1))
            # Variance of phi' within the window (envelope-weighted): use
            #   var(phi') ≈ 1 - R^2  where R is the resultant mag of unit phasors,
            #   but our weighted variant: use w-weighted (sin^2+cos^2-R^2)
            ph_var = (w_env * (sin_phi - mean_sin.unsqueeze(2)) ** 2).sum(dim=2) / w_sum
            if_std_per_band.append(torch.sqrt(ph_var + 1e-8))       # (T, B)

            # FM-RATE SPECTRUM per band: rfft of env-weighted sin_phi (a real
            # 1-D signal of length W-1, pad to W). Project onto the existing
            # 2nd-order Morlet bank H2_sq.
            fm_sig = w_env * sin_phi                                # (T, B, W-1)
            # zero-pad to W and rfft
            FM_PAD = torch.nn.functional.pad(fm_sig, (0, 1))
            FM = torch.fft.rfft(FM_PAD, n=W, dim=2)
            pF = (FM.conj() * FM).real
            FM_e = pF @ self.H2_sq.T                                # (T, B, n_lambda2)
            if_fm_rate_chunks.append(torch.log(FM_e + 1e-8))

        S2_within = torch.cat(S2_within_chunks, dim=1)
        if_center_mag = torch.cat(if_mean_per_band, dim=1)          # (T, n_lambda1, 2)
        if_std = torch.cat(if_std_per_band, dim=1)                  # (T, n_lambda1)
        if_fm_rate = torch.cat(if_fm_rate_chunks, dim=1)            # (T, n_lambda1, n_lambda2)
        env_skew_all = torch.cat(env_skew_per_band, dim=1)           # (T, n_lambda1)
        env_kurt_all = torch.cat(env_kurt_per_band, dim=1)           # (T, n_lambda1)
        log_env_skew_all = torch.cat(log_env_skew_per_band, dim=1)
        log_env_kurt_all = torch.cat(log_env_kurt_per_band, dim=1)
        # Reshape to (T, n_lambda1, n_lambda2) then pool lambda1 into S2_WITHIN_GROUPS bands.
        S2_within = S2_within.reshape(T, self.n_lambda1, self.n_lambda2)
        edges = np.linspace(0, self.n_lambda1, self.S2_WITHIN_GROUPS + 1).astype(int)
        S2_pool = torch.stack([S2_within[:, edges[g]:edges[g + 1]].mean(dim=1)
                                for g in range(self.S2_WITHIN_GROUPS)], dim=1)
        # shape (T, S2_WITHIN_GROUPS, n_lambda2) -> flatten
        feats.append(S2_pool.reshape(T, self.S2_WITHIN_GROUPS * self.n_lambda2))

        # ---------- NEW v17: instantaneous-frequency (IF) features ----------
        # Pool IF stats (center, voicing magnitude, std) to S2_WITHIN_GROUPS bands.
        edges2 = np.linspace(0, self.n_lambda1, self.S2_WITHIN_GROUPS + 1).astype(int)
        if_pool = []
        for g in range(self.S2_WITHIN_GROUPS):
            sl = slice(edges2[g], edges2[g + 1])
            if_pool.append(if_center_mag[:, sl].mean(dim=1))         # (T, 2)
            if_pool.append(if_std[:, sl].mean(dim=1, keepdim=True))  # (T, 1)
        feats.append(torch.cat(if_pool, dim=1))                      # (T, 24*3) = 72

        # FM-RATE SPECTRUM: pool to a small number of band groups (8) ×
        # n_lambda2 to keep dim down.
        N_FM_GROUPS = 8
        edges_fm = np.linspace(0, self.n_lambda1, N_FM_GROUPS + 1).astype(int)
        fm_pool = torch.stack([
            if_fm_rate[:, edges_fm[g]:edges_fm[g + 1]].mean(dim=1)
            for g in range(N_FM_GROUPS)
        ], dim=1)
        feats.append(fm_pool.reshape(T, N_FM_GROUPS * self.n_lambda2))   # 8 * 14 = 112

        # ---------- ACROSS-FRAME slow AM ----------
        S2_across = apply_time_modulus_bank(S1_long, self.time_kernels)
        feats.append(S2_across)

        # ---------- 3RD-ORDER SCATTERING ----------
        # Apply across-frame Morlet bank to within-window 2nd-order output.
        # Captures "how the within-window AM spectrum evolves at syllabic rate"
        # (e.g., pitch energy modulated at 4 Hz = syllabic rhythm of voicing).
        # Two slicings to keep dim down: (a) sum-over-lambda1 -> per-AM-rate band; and
        # (b) sum-over-lambda2 -> per-lambda1-group AM-loudness.
        S2w_pool = S2_pool   # (T, S2_WITHIN_GROUPS=24, n_lambda2=14)
        S2w_per_amrate = S2w_pool.mean(dim=1)               # (T, n_lambda2)
        S2w_per_lambda1grp = S2w_pool.mean(dim=2)            # (T, S2_WITHIN_GROUPS)
        S3_a = apply_time_modulus_bank(S2w_per_amrate, self.s3_kernels)
        S3_b = apply_time_modulus_bank(S2w_per_lambda1grp, self.s3_kernels)
        feats.append(S3_a)
        feats.append(S3_b)

        power_long = (Xh_long.conj() * Xh_long).real
        # [v12 trim] Q24 dropped — kept short-tau adapt only.

        # ---------- v20: DIVISIVE-NORMALIZATION (LATERAL-CONTRAST) cochleagram ----------
        # For each band j, compute E_j / (eps + Gaussian_smooth_across_bands(E))
        # — a divisive-normalization "contrast" cochleagram. Captures formant
        # PEAK PROMINENCE directly: peaks become large when their neighborhood
        # is energy-poor (sharp resonance), independent of broadband loudness.
        # Different from log_S1 (absolute log energy) and from DoG/HPF lateral
        # inhibition (subtractive). Divisive contrast is what cortical models
        # (Heeger, Carandini) and the CARFAC cochlea both use.
        E_long_for_norm = power_long @ self.H1_sq_per_win[W].T   # (T, n_lambda1)
        # SPECTRAL divisive normalization at sigma=3 (v20 winner).
        sigma_band = 3.0
        k_band = int(np.ceil(3 * sigma_band))
        coords_b = torch.arange(-k_band, k_band + 1, device=self.device,
                                 dtype=E_long_for_norm.dtype)
        gauss_b = torch.exp(-(coords_b ** 2) / (2 * sigma_band ** 2))
        gauss_b = gauss_b / gauss_b.sum()
        gauss_b = gauss_b.view(1, 1, -1)
        E_smooth_b = torch.nn.functional.conv1d(
            E_long_for_norm.unsqueeze(1), gauss_b, padding=k_band).squeeze(1)
        E_contrast = torch.log((E_long_for_norm + 1e-10) /
                                (E_smooth_b + 1e-10))   # (T, n_lambda1)
        feats.append(E_contrast)

        # v25 (kept): per-band envelope CUMULANTS pooled to 24 groups × 2 stats.
        edges_ec = np.linspace(0, self.n_lambda1, self.S2_WITHIN_GROUPS + 1).astype(int)
        env_cum_pool = []
        for g in range(self.S2_WITHIN_GROUPS):
            sl = slice(edges_ec[g], edges_ec[g + 1])
            env_cum_pool.append(env_skew_all[:, sl].mean(dim=1, keepdim=True))
            env_cum_pool.append(env_kurt_all[:, sl].mean(dim=1, keepdim=True))
        env_cum_block = torch.cat(env_cum_pool, dim=1)
        feats.append(env_cum_block)                                   # 48 dim
        env_cum_pool = env_cum_block  # alias for v27 adaptation below
        # v28+v29: MULTI-TAU CAUSAL + ACAUSAL adaptation on env_cumulants.
        # Causal taus 15/60 = burstiness change "compared to past". Acausal
        # taus 15/60 = compared to future. The combination gives a Marr-style
        # center-surround in time around the current frame for each cumulant.
        for tau in (15.0, 60.0):
            feats.append(env_cum_block - self._causal_lowpass(env_cum_block, tau))
        for tau in (15.0, 60.0):
            feats.append(env_cum_block - self._acausal_lowpass(env_cum_block, tau))

        # ---------- SHORT-TAU adaptation on standard S1 (fast time-scale) ----------
        # v28: drop tau=4.5 to make room for multi-tau env_cumulant adaptation.
        feats.append(S1_long - self._causal_lowpass(S1_long, 1.5))

        # ---------- (v17) cube-root path dropped — replaced by IF features ----------

        # ---------- SPECTRAL MOMENTS, FLATNESS, ROLLOFF, HNR, PHONETIC TEMPLATES ----------
        # Compact closed-form summaries of the long-window spectrum.
        long_freqs = self._long_freqs_t
        p_norm = power_long / (power_long.sum(dim=1, keepdim=True) + 1e-12)
        mu = (p_norm * long_freqs).sum(dim=1, keepdim=True)
        df = long_freqs - mu
        var = (p_norm * df ** 2).sum(dim=1, keepdim=True)
        std = torch.sqrt(var + 1e-12)
        skew = ((p_norm * df ** 3).sum(dim=1, keepdim=True) /
                (std ** 3 + 1e-12)).clamp(-10, 10)
        kurt = ((p_norm * df ** 4).sum(dim=1, keepdim=True) /
                (var ** 2 + 1e-12)).clamp(0, 50)
        moments = torch.cat([torch.log(mu.clamp(min=1.0)),
                              torch.log(std.clamp(min=1.0)),
                              skew, kurt], dim=1)
        feats.append(moments)
        # Spectral flatness: log(geo_mean / arith_mean).
        log_p = torch.log(power_long.clamp(min=1e-10))
        geo = log_p.mean(dim=1, keepdim=True)
        arith = torch.log(power_long.mean(dim=1, keepdim=True) + 1e-10)
        feats.append(geo - arith)
        # Spectral rolloff at 0.85, 0.95.
        cs = torch.cumsum(power_long, dim=1)
        total = cs[:, -1:] + 1e-12
        for thr in (0.85, 0.95):
            mask = cs >= thr * total
            idx = mask.float().argmax(dim=1)
            feats.append(torch.log(long_freqs[idx].clamp(min=1.0)).unsqueeze(1))
        # HNR via normalized ACF peak.
        full = torch.fft.irfft(power_long, n=W)
        acf0 = full[:, :1] + 1e-12
        acf_n = full / acf0
        lag_min = max(1, int(SR / 400.0))
        lag_max = min(W - 1, int(SR / 75.0))
        region = acf_n[:, lag_min:lag_max + 1]
        peak_val, _ = region.max(dim=1, keepdim=True)
        peak_val = peak_val.clamp(0.0, 0.999)
        hnr_db = 10.0 * torch.log10(peak_val.clamp(min=1e-3) / (1.0 - peak_val).clamp(min=1e-3))
        hnr = torch.cat([peak_val, hnr_db], dim=1)
        feats.append(hnr)
        # Phonetic templates on log-cochleagram S1_long.
        nF_p = S1_long.shape[1]
        hi = S1_long[:, max(0, nF_p - 6):].mean(dim=1, keepdim=True)
        lo = S1_long[:, :max(1, nF_p // 3)].mean(dim=1, keepdim=True)
        mid = S1_long[:, nF_p // 4:3 * nF_p // 4].mean(dim=1, keepdim=True)
        very_hi = S1_long[:, max(0, nF_p - 3):].mean(dim=1, keepdim=True)
        narrow_lo = S1_long[:, 1:6].mean(dim=1, keepdim=True)
        notch = S1_long[:, 8:14].mean(dim=1, keepdim=True) if nF_p >= 14 else mid
        flatness = S1_long.std(dim=1, keepdim=True)
        sibilant = hi - lo
        vowel = mid - flatness
        nasal = narrow_lo - notch
        plosive = flatness - S1_long.mean(dim=1, keepdim=True)
        fricative = very_hi - mid
        phon = torch.cat([sibilant, vowel, nasal, plosive, fricative], dim=1)
        feats.append(phon)
        # Multi-tau adaptation on selected 1-D signals
        sig_1d = torch.cat([moments, hnr, phon], dim=1)
        for tau in (15.0, 60.0):
            feats.append(sig_1d - self._causal_lowpass(sig_1d, tau))

        # ---------- SOFT-ARGMAX FORMANT TRACKING ----------
        # Split log_S1 into 3 freq bands (F1, F2, F3 ranges); soft-argmax peak
        # location and amplitude per band. Captures formant frequencies + dynamics.
        nF = S1_long.shape[1]
        b1 = S1_long[:, :nF // 3]
        b2 = S1_long[:, nF // 3:2 * nF // 3]
        b3 = S1_long[:, 2 * nF // 3:]
        coords = torch.arange(nF, device=self.device, dtype=S1_long.dtype).view(1, -1)
        coords1 = coords[:, :nF // 3]
        coords2 = coords[:, nF // 3:2 * nF // 3]
        coords3 = coords[:, 2 * nF // 3:]
        fmt_feats = []
        for b, c in ((b1, coords1), (b2, coords2), (b3, coords3)):
            w = torch.softmax(b * 2.0, dim=1)
            loc = (w * c).sum(dim=1, keepdim=True)
            amp = (w * b).sum(dim=1, keepdim=True)
            fmt_feats.append(loc)
            fmt_feats.append(amp)
        formants = torch.cat(fmt_feats, dim=1)   # (T, 6) = [F1_loc, F1_amp, F2_loc, ...]
        feats.append(formants)
        # Formant diffs and dynamics
        feats.append(formants[:, [2, 4]] - formants[:, [0, 2]])   # F2-F1, F3-F2
        for tau in (4.5, 15.0, 60.0):
            feats.append(formants - self._causal_lowpass(formants, tau))
        # Formant velocity (derivative) and HWR
        fmt_d = torch.zeros_like(formants)
        fmt_d[1:] = formants[1:] - formants[:-1]
        feats.append(torch.cat([fmt_d, fmt_d.clamp(min=0.0), (-fmt_d).clamp(min=0.0)], dim=1))

        # [v11 trim] dropped |mean_z| too — fits memory.

        # ---------- ADAPTATION features on long S1 ----------
        # Causal: log_S1 - lowpass(log_S1, tau): per-band dynamic deviation from past.
        # Acausal: log_S1 - lowpass_reversed(log_S1, tau): deviation from FUTURE context.
        for tau in (15.0, 60.0, 150.0):
            lp = self._causal_lowpass(S1_long, tau)
            feats.append(S1_long - lp)
            feats.append(lp)
        for tau in (15.0, 60.0):
            lpf = self._acausal_lowpass(S1_long, tau)
            feats.append(S1_long - lpf)
            feats.append(lpf - self._causal_lowpass(S1_long, tau))

        # ---------- Multi-Q 1st-order ----------
        # [v13 trim] Q=4 multi-Q path dropped (small contribution, makes room for 3rd-order).

        # [trimmed: Q4 transients]

        # ---------- HARMONIC STACKING (F0 salience) — light: salience only, no dynamics ----------
        ps = self._harmonic_stack(power_long, n_harmonics=10)
        feats.append(ps)

        # ---------- TRANSIENT / ONSET features ----------
        for k in self.ONSET_LAGS:
            fwd = torch.cat([S1_long[k:], S1_long[-1:].expand(k, -1)], dim=0)
            delta = fwd - S1_long
            feats.append(delta.clamp(min=0.0))   # positive (onset) part per band
            feats.append((-delta).clamp(min=0.0))  # negative (offset) part per band
        # Broadband loudness onset: HWR of d(log_total_energy)/dt at multiple lags
        loud = S1_long.mean(dim=1, keepdim=True)  # (T, 1)
        for k in (1, 2, 4, 8, 16):
            fwd = torch.cat([loud[k:], loud[-1:].expand(k, -1)], dim=0)
            d = fwd - loud
            feats.append(torch.cat([d, d.clamp(min=0.0), (-d).clamp(min=0.0)], dim=1))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "SCAT-v29"
model_description = (
    "SCAT v29: SCAT-v28 (0.0687) + ACAUSAL multi-tau adaptation on env_"
    "cumulants at taus 15 and 60 (in addition to v28's causal multi-tau). "
    "Past + future context around current frame = center-surround time "
    "filter on cumulants. +96 dim. Drop TIME_MOD_RATES 0.75 Hz (-84 dim) "
    "to make room. ~3080 dim/frame."
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
