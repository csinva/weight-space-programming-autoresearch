"""
Wavelet scattering transform front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

Paradigm: Mallat/Anden-Mallat wavelet scattering. A deep, translation-stable,
fully training-free cascade of constant-Q Morlet filterbank + complex modulus +
lowpass averaging. Computes 1st-order scattering (analogous to a log-mel
envelope) AND higher-order scattering (cross-frame modulation, within-window
AM, joint time-frequency 2D scattering) which recover the amplitude-modulation
and transient information that 1st-order averaging destroys.

Iter-1 design:
  * 1st order: Morlet filterbank in freq domain (Q1=12, ~7 octaves) -> log
    sub-band energy per window. 4 analysis windows so phi (averaging scale T) is
    multi-scale.
  * 2nd-order TIME scattering ACROSS frames (30 Hz axis):
    |log_S1 * psi_lambda2| at modulation rates 1.5,3,6,10,15 Hz, per lambda1.
  * 2nd-order TIME scattering WITHIN window (high AM rates):
    For each lambda1, take |x * psi_lambda1| (full-rate envelope), apply
    psi_lambda2 at modulation rates 50,100,200 Hz inside the same 4096-sample
    window, take modulus and average -> per-band fast AM features.
  * Joint-TF scattering: 2D Morlet wavelets on the (t, lambda1) log-cochleagram
    over slow/syllabic rates × scale directions (up/down sweeps).
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
# Morlet wavelet filterbank (frequency domain, constant-Q)
# ---------------------------------------------------------------------------

def morlet_freq_bank(win: int, q: int, n_octaves: int, f_max: float,
                     sr: float = SR) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, xi):
      H  (n_filters, n_freqs)  freq-domain Gaussian magnitude filters
      xi (n_filters,)          center frequencies (Hz), descending
    A constant-Q tiling: center freq xi_j = f_max * 2^(-j/q), bandwidth xi_j/q.
    """
    n_filters = q * n_octaves
    js = np.arange(n_filters, dtype=np.float64)
    xi = f_max * 2.0 ** (-js / q)
    sigma_f = np.maximum(xi / q, 1.0)  # never go sub-Hz
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    H = np.exp(-((freqs[None, :] - xi[:, None]) ** 2) /
               (2.0 * sigma_f[:, None] ** 2))
    return H.astype(np.float32), xi.astype(np.float32)


def lowpass_freq(win: int, cutoff_hz: float, sr: float = SR) -> np.ndarray:
    """Gaussian lowpass phi (real, freq-domain), cutoff = -1 sigma point."""
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    sigma = max(cutoff_hz, 1.0)
    return np.exp(-(freqs ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Across-frame 1D scattering helpers (FFT-based circular conv on 30 Hz axis)
# ---------------------------------------------------------------------------

def time_morlet_kernels(rates_hz: tuple[float, ...], fs: float = 30.0,
                         widths_periods: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """For a list of modulation-frequency targets, build real (cos, sin) Morlet
    kernels centered at 0 with Gaussian envelope of widths_periods periods."""
    kernels = []
    for f in rates_hz:
        period = fs / f
        sigma = widths_periods * period
        L = int(np.ceil(4 * sigma))
        t = np.arange(-L, L + 1)
        env = np.exp(-(t ** 2) / (2 * sigma ** 2))
        cos_k = env * np.cos(2 * np.pi * f * t / fs)
        sin_k = env * np.sin(2 * np.pi * f * t / fs)
        # DC-remove
        cos_k -= cos_k.mean()
        sin_k -= sin_k.mean()
        # unit L2 norm
        n = np.sqrt((cos_k ** 2 + sin_k ** 2).sum()) + 1e-12
        cos_k /= n
        sin_k /= n
        kernels.append((cos_k.astype(np.float32), sin_k.astype(np.float32)))
    return kernels


def apply_time_modulus_bank(x: torch.Tensor, kernels: list) -> torch.Tensor:
    """x: (T, C). kernels: list of (cos_k, sin_k) np.float32.
    Returns (T, C * len(kernels)) modulus envelopes."""
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
# Joint TF 2D Morlet bank on (T, F=n_lambda1) log-cochleagram
# ---------------------------------------------------------------------------

def joint_tf_morlet_2d(rate_hz: float, scale_cpc: float, fs: float = 30.0,
                       sigma_periods: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """2D Morlet (cos, sin) over (time, log-freq channel).
       rate_hz: temporal modulation rate
       scale_cpc: spectral modulation rate in cycles per channel (sign = up/down)
    """
    period_t = fs / max(abs(rate_hz), 0.5)
    sigma_t = sigma_periods * period_t
    Lt = max(2, int(np.ceil(3 * sigma_t)))
    period_f = 1.0 / max(abs(scale_cpc), 0.1)
    sigma_f = sigma_periods * period_f
    Lf = max(2, int(np.ceil(3 * sigma_f)))
    t = np.arange(-Lt, Lt + 1)
    f = np.arange(-Lf, Lf + 1)
    env = np.exp(-((t[:, None] ** 2) / (2 * sigma_t ** 2) +
                   (f[None, :] ** 2) / (2 * sigma_f ** 2)))
    phase = 2 * np.pi * rate_hz * t[:, None] / fs + 2 * np.pi * scale_cpc * f[None, :]
    cos_k = env * np.cos(phase)
    sin_k = env * np.sin(phase)
    cos_k -= cos_k.mean()
    sin_k -= sin_k.mean()
    n = np.sqrt((cos_k ** 2 + sin_k ** 2).sum()) + 1e-12
    return cos_k.astype(np.float32) / n, sin_k.astype(np.float32) / n


def apply_joint_tf_modulus(log_tf: torch.Tensor, kernels: list) -> torch.Tensor:
    """log_tf: (T, F). kernels: list of (cos_k, sin_k) 2D arrays.
       Returns concat of per-kernel modulus envelopes, each (T, F)."""
    device = log_tf.device
    outs = []
    x = log_tf.unsqueeze(0).unsqueeze(0)  # (1,1,T,F)
    for cos_k, sin_k in kernels:
        ck = torch.from_numpy(cos_k).to(device).unsqueeze(0).unsqueeze(0)
        sk = torch.from_numpy(sin_k).to(device).unsqueeze(0).unsqueeze(0)
        Hh, Ww = ck.shape[-2], ck.shape[-1]
        pad = (Hh // 2, Ww // 2)
        c = torch.nn.functional.conv2d(x, ck, padding=pad)
        s = torch.nn.functional.conv2d(x, sk, padding=pad)
        env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
        outs.append(env[:log_tf.shape[0], :log_tf.shape[1]])
    return torch.cat([o for o in outs], dim=1)


# ---------------------------------------------------------------------------
# Within-window 2nd-order scattering (fast AM)
# ---------------------------------------------------------------------------

def within_window_am_features(X: torch.Tensor, H1: torch.Tensor,
                              am_kernels_freq: torch.Tensor,
                              win: int) -> torch.Tensor:
    """For each lambda1 wavelet:
        y_lambda1 = ifft(X * H_lambda1) -> length-win complex (we take real-only since H1 is real and X is from real signal — actually H1 acts on positive freqs of a real signal, but we apply it to rfft so y is real... oh wait — for a complex envelope we need an analytic filter.

       Implementation: we treat the rfft * H multiplication as a real-only
       transfer (Gaussian magnitude). irfft gives a real bandpassed signal;
       the analytic envelope is sqrt(BP^2 + Hilbert^2). For a constant-Q
       Gaussian bandpass we can build the analytic version by multiplying
       the rfft by 2 * H (zero negative-freq half) then irfft as complex -> abs.

    We do exactly that here.
    Then envelope = |y_analytic| -> shape (T, n_lambda1, win).
    For each AM rate (am_kernels_freq), compute power-sum:
      sum_t (env(t) * exp(-i 2pi rate t))^2 / win  via FFT.
    Returns (T, n_lambda1 * n_am).
    """
    T = X.shape[0]
    n_lambda1, n_freqs = H1.shape
    n_am = am_kernels_freq.shape[0]
    # Build analytic-filter version of each lambda1: shape (n_lambda1, n_freqs) complex
    # ifft of (X * 2H) gives analytic signal whose modulus is the envelope.
    # We do per-lambda1 separately to limit memory.
    chunks = []
    for j in range(n_lambda1):
        Hj = (2.0 * H1[j]).unsqueeze(0)  # (1, n_freqs)
        Yj = X * Hj                       # (T, n_freqs)
        yj = torch.fft.irfft(Yj, n=win)   # (T, win) - this is the analytic signal cosine part; we need full analytic.
        # Build analytic via Hilbert: treat Yj as one-sided spectrum and do complex ifft on (n_freqs + win//2) zeros.
        # Simpler: separate real and imag-shifted parts.
        # Actually for analytic signal: A = ifft(Y_pos, n=win), with Y_pos doubled on positive freqs (excluding DC and Nyquist).
        # torch.fft.irfft already does real ifft. For analytic, we need full complex ifft of zero-padded.
        # Build full spectrum: positive freqs from Yj, zeros for negative freqs.
        Yfull = torch.zeros((T, win), dtype=torch.complex64, device=X.device)
        Yfull[:, :n_freqs] = Yj
        # DC and Nyquist are unique; don't double them (we already doubled all in 2H, which OVER-doubles DC).
        # Fix DC and Nyquist back: divide by 2 (since we doubled with 2H).
        Yfull[:, 0] = Yj[:, 0] * 0.5  # but Yj already has the 2H multiplication; undo for DC
        if win % 2 == 0:
            Yfull[:, n_freqs - 1] = Yj[:, n_freqs - 1] * 0.5
        analytic = torch.fft.ifft(Yfull, n=win)  # (T, win) complex
        env = analytic.abs()                       # (T, win) real envelope
        # Compute amplitude at each AM rate: project env onto exp(2 pi i rate t/SR),
        # take mean magnitude over window.
        # am_kernels_freq is (n_am, win) precomputed complex exponentials.
        proj = env.unsqueeze(1) * am_kernels_freq.unsqueeze(0)  # (T, n_am, win) complex
        amp = proj.sum(dim=2).abs() / float(win)  # (T, n_am)
        chunks.append(amp)
    return torch.cat(chunks, dim=1)  # (T, n_lambda1 * n_am)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Wavelet scattering front-end.

    Iter-1: 1st order (log-energy in Morlet sub-bands at 4 averaging windows)
            + across-frame 2nd-order time scattering
            + joint-TF 2D scattering.
    """

    win = 4096
    n_params = 0

    WIN_LIST = (512, 1024, 2048, 4096)   # multiscale lowpass (phi) sizes
    Q1 = 12
    N_OCT = 7
    F_MAX = 8000.0
    # Across-frame 2nd-order modulation rates (Hz at 30 Hz frame rate)
    TIME_MOD_RATES = (1.5, 3.0, 6.0, 10.0, 14.0)
    # Joint-TF 2D Morlet kernels: (rate_hz, scale_cyc_per_channel)
    JOINT_TF_RATES = (
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

        # Build per-window Morlet filterbanks (H_w shape (n_filters, n_freqs_w))
        self.H_per_win = {}
        self.H2_per_win = {}
        for w in self.WIN_LIST:
            H, xi = morlet_freq_bank(w, q=self.Q1, n_octaves=self.N_OCT, f_max=self.F_MAX)
            self.H_per_win[w] = torch.from_numpy(H).to(device)
            self.H2_per_win[w] = torch.from_numpy(H * H).to(device)
        self.xi = xi  # same across windows
        self.n_lambda1 = self.Q1 * self.N_OCT

        # Across-frame time scattering kernels
        self.time_kernels = time_morlet_kernels(self.TIME_MOD_RATES, fs=30.0,
                                                widths_periods=1.0)

        # Joint TF kernels
        self.joint_kernels = [joint_tf_morlet_2d(r, s) for (r, s) in self.JOINT_TF_RATES]

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        T, win_full = x.shape
        feats = []

        # ---------------- 1st-order scattering at multiple averaging scales ----------------
        # For each averaging window size w, crop x centrally to w (frames already at right grid).
        S1_logs = []
        for w in self.WIN_LIST:
            if w == win_full:
                xw = x
            else:
                start = (win_full - w) // 2
                xw = x[:, start:start + w]
            # rfft of xw  → (T, n_freqs)
            X = torch.fft.rfft(xw, n=w)
            power = (X.conj() * X).real           # (T, n_freqs)
            # sub-band energy: (T, n_lambda1)
            E = power @ self.H2_per_win[w].T
            S1 = torch.log(E + 1e-8)
            S1_logs.append(S1)
            feats.append(S1)

        # Use the largest-window S1 as the primary "log-cochleagram"
        S1_long = S1_logs[-1]                       # (T, n_lambda1)

        # ---------------- Across-frame 2nd-order time scattering ----------------
        # Apply Morlet bank ACROSS T axis to each lambda1 channel.
        S2_time = apply_time_modulus_bank(S1_long, self.time_kernels)  # (T, n_lambda1 * n_rates)
        feats.append(S2_time)

        # ---------------- Joint-TF 2D scattering on (T, F=n_lambda1) ----------------
        S_joint = apply_joint_tf_modulus(S1_long, self.joint_kernels)  # (T, n_lambda1 * n_kernels)
        feats.append(S_joint)

        # ---------------- Output ----------------
        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "SCAT-v1"
model_description = (
    "SCAT v1: pure wavelet scattering front-end. 1st-order log-energy in Morlet "
    "filterbank (Q1=12, 7 oct, fmax=8000) at 4 averaging windows (512/1024/2048/4096); "
    "across-frame 2nd-order time scattering (Morlet at 1.5,3,6,10,14 Hz on each "
    "lambda1 channel); joint-TF 2D Morlet scattering (16 (rate,scale) pairs) on "
    "the (T, lambda1) log-cochleagram."
)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------

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
