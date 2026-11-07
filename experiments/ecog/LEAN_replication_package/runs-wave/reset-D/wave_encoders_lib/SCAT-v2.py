"""
Wavelet scattering transform front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

Iter-2: proper Mallat scattering.

  * Hann-windowed FFT per analysis window.
  * 1st order (log scattering) at long window: log of sub-band ENERGY in a
    constant-Q Morlet bank (Q1=12, 7 oct, fmax=8000) -> 84 channels.
  * Within-window 2nd-order AM scattering:
    For each lambda1, build the ANALYTIC bandpassed signal y_lambda1 (complex
    ifft of zero-extended X * 2H), take its modulus envelope, rfft the envelope,
    and project onto another constant-Q Morlet bank (Q2=1) over MODULATION
    rates (4..1024 Hz inside the window). Returns 84 * n_lambda2 channels of
    log AM energy. Captures fast amplitude modulation that 1st-order destroys.
  * Across-frame (slow AM) 2nd-order time scattering on log_S1: convolve with
    Morlet across t at modulation rates (1.5,3,6,10,14 Hz). Captures slow AM
    below the within-window Nyquist (~10 Hz).
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
    """Return (H, xi):
      H  (n_filters, n_freqs)
      xi (n_filters,)
    Constant-Q wavelet bank: xi_j = f_max * 2^(-j/q), bandwidth = xi_j/q (with floor).
    """
    n_filters = int(round(q * n_octaves))
    js = np.arange(n_filters, dtype=np.float64)
    xi = f_max * 2.0 ** (-js / q)
    sigma_f = np.maximum(xi / q, q_floor)
    freqs = np.arange(n_freqs, dtype=np.float64) * freq_step
    H = np.exp(-((freqs[None, :] - xi[:, None]) ** 2) /
               (2.0 * sigma_f[:, None] ** 2))
    return H.astype(np.float32), xi.astype(np.float32)


# ---------------------------------------------------------------------------
# Across-frame 1D Morlet bank on 30 Hz axis (slow AM scattering)
# ---------------------------------------------------------------------------

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
    xp = x.transpose(0, 1).unsqueeze(0)  # (1, C, T)
    for cos_k, sin_k in kernels:
        ck = torch.from_numpy(cos_k).to(device).view(1, 1, -1).expand(C, 1, -1).contiguous()
        sk = torch.from_numpy(sin_k).to(device).view(1, 1, -1).expand(C, 1, -1).contiguous()
        L = ck.shape[-1]
        pad = L // 2
        xpp = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
        c = torch.nn.functional.conv1d(xpp, ck, groups=C)[..., :T]
        s = torch.nn.functional.conv1d(xpp, sk, groups=C)[..., :T]
        env = torch.sqrt(c * c + s * s + 1e-12)
        outs.append(env.squeeze(0).transpose(0, 1))  # (T, C)
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0

    # 1st order
    Q1 = 12
    N_OCT1 = 7
    F_MAX = 8000.0

    # 2nd-order within-window AM (constant-Q on env DFT at sample rate SR)
    Q2 = 1.0
    N_OCT2 = 8       # 2^8 = 256x range
    AM_F_MAX = 1024.0   # max AM rate (Hz)
    AM_F_MIN_TARGET = 6.0  # ~floor for 2nd order (cycle fits in 167 ms; window is 93 ms)

    # Across-frame slow AM (1.5..14 Hz at 30 Hz frame rate)
    TIME_MOD_RATES = (1.5, 3.0, 6.0, 10.0, 14.0)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.W = self.win  # analysis window samples
        # freq grid for rfft of length W
        self.freqs = np.fft.rfftfreq(self.W, d=1.0 / SR).astype(np.float32)
        self.n_freqs = len(self.freqs)
        freq_step = SR / self.W   # Hz per bin

        # 1st order Morlet bank (freq-domain magnitude filter)
        H1, xi1 = morlet_freq_bank(self.n_freqs, freq_step, self.Q1, self.N_OCT1, self.F_MAX,
                                   q_floor=8.0)  # ~8 Hz min bandwidth
        self.n_lambda1 = H1.shape[0]
        # H1 squared for L2 (energy) projection
        self.H1 = torch.from_numpy(H1).to(device)               # (n_lambda1, n_freqs)
        self.H1_sq = torch.from_numpy(H1 * H1).to(device)
        self.xi1 = xi1

        # ANALYTIC version of H1 (one-sided full-length spectrum, zero negative freqs)
        # We assemble full-length filter Hf[lambda1, W] complex where
        #   Hf[:, :n_freqs] = 2*H1 (with DC/Nyquist halved)
        Hf = np.zeros((self.n_lambda1, self.W), dtype=np.complex64)
        Hf[:, :self.n_freqs] = 2.0 * H1
        Hf[:, 0] = H1[:, 0]               # DC not doubled
        if self.W % 2 == 0:
            Hf[:, self.n_freqs - 1] = H1[:, self.n_freqs - 1]  # Nyquist not doubled
        self.Hf = torch.from_numpy(Hf).to(device)  # (n_lambda1, W) complex64

        # Hann window for FFT of input
        self.hann = torch.hann_window(self.W, periodic=True, device=device)

        # 2nd-order AM Morlet bank operating on env's rfft (length W at SR=44100)
        # env's rfft has n_freqs bins, step = SR/W ≈ 10.77 Hz.
        # AM rate range: AM_F_MIN_TARGET .. AM_F_MAX. Build constant-Q bank.
        n_oct2 = int(np.floor(np.log2(self.AM_F_MAX / self.AM_F_MIN_TARGET)))
        H2, xi2 = morlet_freq_bank(self.n_freqs, freq_step, self.Q2, n_oct2, self.AM_F_MAX,
                                   q_floor=4.0)
        self.n_lambda2 = H2.shape[0]
        self.H2 = torch.from_numpy(H2).to(device)
        self.H2_sq = torch.from_numpy(H2 * H2).to(device)
        self.xi2 = xi2

        # across-frame Morlet kernels
        self.time_kernels = time_morlet_kernels(self.TIME_MOD_RATES, fs=30.0,
                                                widths_periods=1.0)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        T = x.shape[0]
        W = self.W
        feats = []

        # ---------- rfft of Hann-windowed input ----------
        Xh = torch.fft.rfft(x * self.hann.unsqueeze(0), n=W)   # (T, n_freqs)
        power_x = (Xh.conj() * Xh).real                          # (T, n_freqs)

        # ---------- 1st-order log-scattering ----------
        E1 = power_x @ self.H1_sq.T                              # (T, n_lambda1)
        S1 = torch.log(E1 + 1e-8)
        feats.append(S1)

        # ---------- WITHIN-WINDOW 2nd-order AM scattering ----------
        # Strategy: for each lambda1 (batched), build analytic bandpassed signal via
        # full-length complex ifft of (X_padded * Hf[lambda1]), take |.|, rfft of env,
        # project to 2nd-order Morlet bank.
        # X_padded: pad rfft (n_freqs) to W by zeros (since X for real input).
        X_pad = torch.zeros((T, W), dtype=torch.complex64, device=self.device)
        X_pad[:, :self.n_freqs] = Xh    # already Hann-windowed
        BATCH = 12  # lambda1 batch size to bound peak memory
        S2_within_chunks = []
        for j0 in range(0, self.n_lambda1, BATCH):
            j1 = min(j0 + BATCH, self.n_lambda1)
            Hf_b = self.Hf[j0:j1].unsqueeze(0)                # (1, B, W) complex
            Y = X_pad.unsqueeze(1) * Hf_b                      # (T, B, W) complex
            analytic = torch.fft.ifft(Y, n=W, dim=2)           # (T, B, W) complex
            env = analytic.abs()                                # (T, B, W) real
            # rfft of env -> AM spectrum
            ENV = torch.fft.rfft(env, n=W, dim=2)               # (T, B, n_freqs) complex
            # NOTE: don't kill DC here — the 2nd-order BANK does not include DC since
            #   H2 is centered at xi2 >= AM_F_MIN_TARGET.
            pE = (ENV.conj() * ENV).real                        # (T, B, n_freqs)
            # project onto 2nd-order bank: (T, B, n_lambda2)
            E2 = pE @ self.H2_sq.T
            S2_within_chunks.append(torch.log(E2 + 1e-8))
        S2_within = torch.cat(S2_within_chunks, dim=1)          # (T, n_lambda1 * n_lambda2)
        # Flatten to (T, n_lambda1 * n_lambda2)
        S2_within = S2_within.reshape(T, self.n_lambda1 * self.n_lambda2)
        feats.append(S2_within)

        # ---------- ACROSS-FRAME slow AM scattering on S1 ----------
        S2_across = apply_time_modulus_bank(S1, self.time_kernels)  # (T, n_lambda1 * n_rates)
        feats.append(S2_across)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "SCAT-v2"
model_description = (
    "SCAT v2: Hann-windowed FFT. 1st-order Morlet log-scattering (Q1=12, 7 oct, "
    "fmax=8000) -> 84 ch. WITHIN-WINDOW 2nd-order AM scattering: analytic envelope "
    "via complex ifft of one-sided Morlet * X, rfft of |env|, projected onto 2nd "
    "Morlet bank (Q2=1, n_oct2 from 6Hz floor to 1024Hz) -> 84*n_lambda2 ch. "
    "ACROSS-FRAME slow AM scattering on S1 at 1.5,3,6,10,14 Hz -> 84*5 ch."
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
