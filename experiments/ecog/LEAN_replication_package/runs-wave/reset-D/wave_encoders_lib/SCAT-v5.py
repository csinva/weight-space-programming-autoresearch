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

    WIN_LIST = (512, 1024, 2048, 4096)

    Q1 = 12
    N_OCT1 = 7
    F_MAX = 8000.0

    Q2 = 1.0
    AM_F_MAX = 1024.0
    AM_F_MIN_TARGET = 6.0

    TIME_MOD_RATES = (0.75, 1.5, 3.0, 6.0, 10.0, 14.0)

    # Transient / onset lags (frames at 30 Hz)
    ONSET_LAGS = (1, 2, 4, 8, 16)

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

        # Multi-Q (Q=4) 1st-order bank at long window
        H1_Q4, _ = morlet_freq_bank(n_f, step, q=4.0, n_octaves=7,
                                    f_max=self.F_MAX, q_floor=24.0)
        self.H1_Q4_sq = torch.from_numpy(H1_Q4 * H1_Q4).to(device)
        self.n_lambda1_Q4 = H1_Q4.shape[0]

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

        # ---------- WITHIN-WINDOW 2nd-order AM scattering ----------
        X_pad = torch.zeros((T, W), dtype=torch.complex64, device=self.device)
        X_pad[:, :self.n_freqs_long] = Xh_long
        BATCH = 12
        S2_within_chunks = []
        for j0 in range(0, self.n_lambda1, BATCH):
            j1 = min(j0 + BATCH, self.n_lambda1)
            Hf_b = self.Hf[j0:j1].unsqueeze(0)
            Y = X_pad.unsqueeze(1) * Hf_b
            analytic = torch.fft.ifft(Y, n=W, dim=2)
            env = analytic.abs()
            ENV = torch.fft.rfft(env, n=W, dim=2)
            pE = (ENV.conj() * ENV).real
            E2 = pE @ self.H2_sq.T
            S2_within_chunks.append(torch.log(E2 + 1e-8))
        S2_within = torch.cat(S2_within_chunks, dim=1)
        S2_within = S2_within.reshape(T, self.n_lambda1 * self.n_lambda2)
        feats.append(S2_within)

        # ---------- ACROSS-FRAME slow AM ----------
        S2_across = apply_time_modulus_bank(S1_long, self.time_kernels)
        feats.append(S2_across)

        # ---------- ADAPTATION features on long S1 (normalization by slow context) ----------
        # log_S1 - lowpass(log_S1, tau): captures band-specific dynamic range vs slow context.
        for tau in (15.0, 60.0, 150.0):
            lp = self._causal_lowpass(S1_long, tau)
            feats.append(S1_long - lp)
            feats.append(lp)

        # ---------- Multi-Q 1st-order (broader bands, faster time) ----------
        Xh = Xh_long  # already Hann-windowed long-window rfft
        power_long = (Xh.conj() * Xh).real
        E_q4 = power_long @ self.H1_Q4_sq.T   # (T, n_lambda1_Q4)
        S1_q4 = torch.log(E_q4 + 1e-8)
        feats.append(S1_q4)
        # multi-tau adaptation on Q4 S1
        for tau in (15.0, 60.0):
            feats.append(S1_q4 - self._causal_lowpass(S1_q4, tau))

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


model_shorthand_name = "SCAT-v5"
model_description = (
    "SCAT v5: SCAT-v3 base (0.0585), drop joint-TF (no gain). ADD adaptation: "
    "log_S1 - lowpass(log_S1, tau=15,60,150 frames) on long S1 and Q=4 multi-Q "
    "1st-order. Q=4, 7-oct broader-band Morlet bank also adds a parallel 1st-order "
    "view with looser frequency / better time."
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
