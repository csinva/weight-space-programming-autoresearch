"""
Hand-designed auditory front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

The agent edits this file. The goal is to hand-design (NO TRAINING) an
interpretable auditory front-end that maps the raw audio to a per-frame feature
which predicts high-gamma cortical activity — modelling the *algorithmic priors*
of early hearing (cochlear filtering, compression, adaptation, modulation tuning,
onset enhancement, ...). A linear ridge readout to the 166 electrodes is fit on
top (the only data-fit step), exactly as for the mel and Whisper references.

Rules of the game:
  * You MAY use any deterministic signal-processing transform whose parameters are
    set by formula or by hand: FFT/STFT, gammatone/ERB/mel filterbanks, rectification,
    log/power compression, fixed convolutions, lateral inhibition, adaptation with
    fixed time constants, modulation filterbanks, divisive normalization, and
    hand-written weight tensors / hand-built transformer or conv circuits.
  * You MAY NOT estimate ANY front-end parameter from data: no gradient descent /
    optimizer / backprop, and no PCA/ICA/NMF/k-means/sparse-coding/regression FIT to
    the audio. Evolution already "found" these priors; you implement them in closed
    form. The ONLY data-fit step is the final ridge readout (in the eval harness).
  * Do NOT load pretrained model weights.

ALIGNMENT IS HANDLED FOR YOU. The pipeline hands your front-end pre-aligned audio
`windows` of shape (T, win): one window per 30 Hz frame, already centered on the
exact same grid as the mel-spectrogram / high-gamma (the 1 s extraction crop is
baked in). Set `WaveEmbedder.win` to choose the analysis window length (<= 4096);
the harness centers every window for you. Do NOT try to re-window or re-time the
audio yourself — just transform each (T, win) batch into (T, H) features. The
pipeline then applies the standard FIR delays + 100 ms lag + 30-frame trim.

Usage:
    python wave_encoder.py --num-train 500 --num-test 100 --device cuda
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
# DSP building blocks (fixed, closed-form — reuse / extend these)
# ---------------------------------------------------------------------------

def erb(fc):
    """Equivalent Rectangular Bandwidth (Glasberg & Moore 1990), Hz."""
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    """`n` center frequencies equally spaced on the ERB-rate scale (cochlear spacing)."""
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=50, f_min=50.0, f_max=8000.0, order=4):
    """Fixed gammatone power-response filterbank as a (n_freqs, n_filters) matrix.

    Each column is the squared magnitude response of a 4th-order gammatone filter
    centered at an ERB-spaced frequency, evaluated at the rFFT bins. Applying it to
    a power spectrum yields per-filter energy — a cochleagram channel."""
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)               # (n_freqs,)
    fcs = erb_space(f_min, f_max, n_filters)               # (n_filters,)
    b = 1.019 * erb(fcs)                                   # bandwidths
    # |H(f)|^2 for an order-N gammatone ≈ [1 + ((f-fc)/b)^2]^(-N)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))         # (n_freqs, n_filters)


def power_spectrum(windows_t, win, device):
    """(T, win) audio -> (T, n_freqs) power spectrum with a periodic Hann window."""
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2


# ---------------------------------------------------------------------------
# The hand-designed front-end (EDIT THIS — no training)
# ---------------------------------------------------------------------------

def leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Causal leaky-integrator low-pass along axis 0 of (T, H), vectorized via
    truncated exponential FIR convolution (kernel covers ~6 tau, >99% energy)."""
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t                       # impulse response
    # Convolve along time axis. x is (T, C); reshape to (1, C, T).
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def lateral_inhibition_dog(x: torch.Tensor, sigma_center: float = 0.5,
                            sigma_surround: float = 2.0) -> torch.Tensor:
    """Difference-of-Gaussians across the channel axis (axis=1) of a (T, C) map.

    Center-surround filter that enhances spectral contrast. Sigmas in channels."""
    C = x.shape[1]
    k = int(np.ceil(3 * sigma_surround))
    coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    g_c = torch.exp(-(coords ** 2) / (2 * sigma_center ** 2))
    g_s = torch.exp(-(coords ** 2) / (2 * sigma_surround ** 2))
    g_c = g_c / g_c.sum()
    g_s = g_s / g_s.sum()
    dog = (g_c - g_s).view(1, 1, -1)
    # 1D conv along channels via reshape to (T, 1, C)
    xc = x.unsqueeze(1)
    li = torch.nn.functional.conv1d(xc, dog, padding=k)
    return li.squeeze(1)


class WaveEmbedder:
    """Multi-window (multi-resolution) gammatone + multi-timescale adaptation.

    Two analysis windows from the same aligned 4096-sample frame:
      - short (1024 samples, 23 ms) → fine temporal resolution, coarse frequency
      - long (4096 samples, 93 ms)  → fine frequency resolution, coarse temporal
    Cochlear filterbanks of each yield two complementary cochleagrams.
    """

    win = 4096
    n_params = 0

    def __init__(self, device: str = "cuda", n_filters: int = 32):
        self.device = device
        self.n_filters = n_filters
        self.fb_short = gammatone_powerbank(1024, n_filters=n_filters).to(device)
        self.fb_long  = gammatone_powerbank(4096, n_filters=n_filters).to(device)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)  # (T, 4096)
        # Long window: use the full 4096-sample frame.
        p_long = power_spectrum(x, 4096, self.device)
        e_long = p_long @ self.fb_long
        l_long = torch.log(torch.clamp(e_long, min=1e-10))
        # Short window: center 1024 samples of the same aligned frame.
        x_short = x[:, (4096 - 1024) // 2:(4096 + 1024) // 2]
        p_short = power_spectrum(x_short, 1024, self.device)
        e_short = p_short @ self.fb_short
        l_short = torch.log(torch.clamp(e_short, min=1e-10))
        # Adaptation on each cochleagram (~150 ms timescale)
        a_long  = l_long  - leaky_lowpass(l_long,  tau_frames=4.5)
        a_short = l_short - leaky_lowpass(l_short, tau_frames=4.5)
        feat = torch.cat([l_long, l_short, a_long, a_short], dim=1)  # (T, 4*n_filters)
        return feat.float().cpu().numpy()


model_shorthand_name = "Gammatone-multiwin"
model_description = "Multi-resolution gammatone: short (23 ms, fine-time) + long (93 ms, fine-freq) cochleagrams of the same aligned frame, each with log compression and leaky-mean-subtracted adaptation."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
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
