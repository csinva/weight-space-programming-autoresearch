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

    WIN_LIST = (1024, 2048, 4096)
    S2_WITHIN_GROUPS = 24  # collapse lambda1 axis of within-window 2nd order

    Q1 = 12
    N_OCT1 = 7
    F_MAX = 8000.0  # revert: 12000 lost low-freq pitch coverage

    Q2 = 2.0
    AM_F_MAX = 1024.0
    AM_F_MIN_TARGET = 6.0

    TIME_MOD_RATES = (0.75, 1.5, 3.0, 6.0, 10.0, 14.0)

    # Transient / onset lags (frames at 30 Hz)
    ONSET_LAGS = (1, 2, 4, 8)  # drop 16 to fit memory with formant features

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

        # ---------- WITHIN-WINDOW 2nd-order AM scattering + PHASE PHASORS ----------
        X_pad = torch.zeros((T, W), dtype=torch.complex64, device=self.device)
        X_pad[:, :self.n_freqs_long] = Xh_long
        BATCH = 12
        S2_within_chunks = []
        # Sum of per-band unit phasors across the analysis window — used for PLV.
        # For each lambda1 we keep two summed quantities:
        #   mean_z[t, lambda1]   = (1/W) * sum_tau (y[tau] / |y[tau]|)
        # We then form PLV-like couplings between (k_offset)-shifted bands as
        #   |conj(mean_z[lambda1])  *  mean_z[lambda1 + k]|
        # Note: this is the per-band PHASE-CONCENTRATION mean phasor — informative
        # of voicing even for k=0 (large magnitude => phase locked).
        mean_z_per_band = []
        for j0 in range(0, self.n_lambda1, BATCH):
            j1 = min(j0 + BATCH, self.n_lambda1)
            Hf_b = self.Hf[j0:j1].unsqueeze(0)
            Y = X_pad.unsqueeze(1) * Hf_b
            analytic = torch.fft.ifft(Y, n=W, dim=2)
            env = analytic.abs()
            z = analytic / (env + 1e-6)                           # (T, B, W) complex unit phasor
            mean_z_per_band.append(z.mean(dim=2))                  # (T, B) complex
            ENV = torch.fft.rfft(env, n=W, dim=2)
            pE = (ENV.conj() * ENV).real
            E2 = pE @ self.H2_sq.T
            S2_within_chunks.append(torch.log(E2 + 1e-8))
        S2_within = torch.cat(S2_within_chunks, dim=1)
        mean_z = torch.cat(mean_z_per_band, dim=1)                 # (T, n_lambda1) complex
        # Reshape to (T, n_lambda1, n_lambda2) then pool lambda1 into S2_WITHIN_GROUPS bands.
        S2_within = S2_within.reshape(T, self.n_lambda1, self.n_lambda2)
        edges = np.linspace(0, self.n_lambda1, self.S2_WITHIN_GROUPS + 1).astype(int)
        S2_pool = torch.stack([S2_within[:, edges[g]:edges[g + 1]].mean(dim=1)
                                for g in range(self.S2_WITHIN_GROUPS)], dim=1)
        # shape (T, S2_WITHIN_GROUPS, n_lambda2) -> flatten
        feats.append(S2_pool.reshape(T, self.S2_WITHIN_GROUPS * self.n_lambda2))

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

        # ---------- SHORT-TAU adaptation on standard S1 (fast time-scale) ----------
        for tau in (1.5, 4.5):
            feats.append(S1_long - self._causal_lowpass(S1_long, tau))

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


model_shorthand_name = "SCAT-v15"
model_description = (
    "SCAT v15: revert F_MAX to 8000 (12000 cost 0.0006). ADD soft-argmax formant "
    "tracking: 3 band-groups (F1/F2/F3 ranges), location+amplitude per group, "
    "diffs (F2-F1, F3-F2), multi-tau dynamics (4.5/15/60), velocity HWR. Classical "
    "speech feature scattering misses."
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
