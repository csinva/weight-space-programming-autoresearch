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

def gabor_modulation_filter(rate_hz: float, fs: float = 30.0,
                             sigma_periods: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal band-pass modulation kernel (cos + sin parts) at rate `rate_hz`.

    Gaussian-windowed: σ = sigma_periods periods (broader → narrower band-pass).
    Returns (cos_kernel, sin_kernel) of length ~4σ (in frames)."""
    period = fs / rate_hz
    sigma = sigma_periods * period
    L = int(np.ceil(4 * sigma))
    t = np.arange(-L, L + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
    sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
    # Zero-mean (DC-killed).
    cos_k = cos_k - cos_k.mean()
    cos_k = cos_k / (np.linalg.norm(cos_k) + 1e-12)
    sin_k = sin_k - sin_k.mean()
    sin_k = sin_k / (np.linalg.norm(sin_k) + 1e-12)
    return (torch.from_numpy(cos_k.astype(np.float32)),
            torch.from_numpy(sin_k.astype(np.float32)))


def modulation_energy(x: torch.Tensor, rate_hz: float, fs: float = 30.0) -> torch.Tensor:
    """Energy of a Gabor band-pass on each channel of (T, C). Returns (T, C)."""
    cos_k, sin_k = gabor_modulation_filter(rate_hz, fs=fs)
    cos_k, sin_k = cos_k.to(x.device), sin_k.to(x.device)
    L = cos_k.numel()
    pad = L // 2
    xp = x.transpose(0, 1).unsqueeze(0)  # (1, C, T)
    cs = cos_k.view(1, 1, -1).expand(x.shape[1], 1, L).contiguous()
    ss = sin_k.view(1, 1, -1).expand(x.shape[1], 1, L).contiguous()
    xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    c = torch.nn.functional.conv1d(xp_p, cs, groups=x.shape[1])
    s = torch.nn.functional.conv1d(xp_p, ss, groups=x.shape[1])
    # crop to T if needed
    c = c[..., :x.shape[0]]
    s = s[..., :x.shape[0]]
    env = torch.sqrt(c * c + s * s + 1e-12)   # modulation envelope (analytic mag)
    return env.squeeze(0).transpose(0, 1).contiguous()


def autocorr_pitch_voicing(power: torch.Tensor, win: int, fs: float = 44100.0,
                            f_min: float = 75.0, f_max: float = 400.0) -> torch.Tensor:
    """Per-window pitch/voicing from autocorrelation (cepstral peak).

    power: (T, n_freqs) power spectrum from rFFT of length `win`.
    Returns (T, 3): [voicing_strength, log_pitch_hz, hnr_db]."""
    # ACF = IFFT( |X|^2 ); we need full-length symmetric, so reconstruct via irfft.
    # The DC bin's irfft gives ACF[0]=sum(power) up to normalization.
    full = torch.fft.irfft(power, n=win)                 # (T, win) — ACF (un-normalized)
    acf0 = full[:, :1] + 1e-12                           # (T, 1)
    acf_n = full / acf0                                  # normalize so ACF[0] = 1
    lag_min = max(1, int(fs / f_max))                    # lag for f_max Hz
    lag_max = min(win - 1, int(fs / f_min))              # lag for f_min Hz
    region = acf_n[:, lag_min:lag_max + 1]               # (T, L)
    peak_val, peak_idx = region.max(dim=1)               # (T,)
    pitch_lag = peak_idx + lag_min                       # in samples
    pitch_hz = fs / pitch_lag.float().clamp(min=1.0)
    log_pitch = torch.log(pitch_hz)
    voicing = torch.clamp(peak_val, min=0.0)             # 0..1 (rough)
    hnr_db = 10.0 * torch.log10(voicing.clamp(min=1e-3) /
                                (1.0 - voicing + 1e-3).clamp(min=1e-3))
    return torch.stack([voicing, log_pitch, hnr_db], dim=1)


def dog_across_channels(x: torch.Tensor, sigma_c: float, sigma_s: float) -> torch.Tensor:
    """DoG conv across channels (axis=1) of (T, C)."""
    k = int(np.ceil(3 * sigma_s))
    coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    g_c = torch.exp(-(coords ** 2) / (2 * sigma_c ** 2)); g_c = g_c / g_c.sum()
    g_s = torch.exp(-(coords ** 2) / (2 * sigma_s ** 2)); g_s = g_s / g_s.sum()
    dog = (g_c - g_s).view(1, 1, -1)
    return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)


class WaveEmbedder:
    """Best 3-window × 6-tau base + within-frame AM-depth (sub-window energy variance).

    Inside each 4096-sample (93 ms) frame, we form 4 non-overlapping 23 ms sub-windows
    and compute the log-energy in each — then take the std across these 4 sub-energies
    per channel. This is per-frame amplitude-modulation depth at syllable-rate (≈11 Hz)
    — a feature that the FFT magnitude alone discards."""

    win = 4096
    n_params = 0
    WIN_LIST = (1024, 2048, 4096)
    TAUS = (1.5, 4.5, 15.0, 60.0, 150.0)
    SUB_WIN = 1024  # 4 sub-windows in 4096

    def __init__(self, device: str = "cuda", n_filters: int = 20):
        self.device = device
        self.fbs = [gammatone_powerbank(w, n_filters=n_filters).to(device)
                    for w in self.WIN_LIST]
        self.fb_sub = gammatone_powerbank(self.SUB_WIN, n_filters=n_filters).to(device)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []
        for w, fb in zip(self.WIN_LIST, self.fbs):
            if w == 4096:
                xw = x
            else:
                start = (4096 - w) // 2
                xw = x[:, start:start + w]
            p = power_spectrum(xw, w, self.device)
            log_e = torch.log(torch.clamp(p @ fb, min=1e-10))
            feats.append(log_e)
            for tau in self.TAUS:
                feats.append(log_e - leaky_lowpass(log_e, tau_frames=tau))
        # Within-frame envelope dynamics: AM depth (std), AM range (max-min),
        # and AM onset slope (max sub-to-sub jump). 4 non-overlapping 23 ms sub-windows.
        sub_logs = []
        for i in range(4):
            xs = x[:, i * self.SUB_WIN:(i + 1) * self.SUB_WIN]
            ps = power_spectrum(xs, self.SUB_WIN, self.device)
            le = torch.log(torch.clamp(ps @ self.fb_sub, min=1e-10))
            sub_logs.append(le)
        sub_stack = torch.stack(sub_logs, dim=2)              # (T, F, 4)
        am_depth = sub_stack.std(dim=2)
        am_range = sub_stack.amax(dim=2) - sub_stack.amin(dim=2)
        diffs = sub_stack[:, :, 1:] - sub_stack[:, :, :-1]
        am_onset = diffs.amax(dim=2).clamp(min=0.0)
        feats.append(am_depth)
        feats.append(am_range)
        feats.append(am_onset)
        feat = torch.cat(feats, dim=1)                        # (T, 360 + 3*20) = (T, 420)
        return feat.float().cpu().numpy()


model_shorthand_name = "Gammatone-3win-6tau-AMfull"
model_description = "Best base + within-frame envelope dynamics per channel: AM depth (std), AM range (max−min), AM onset (largest sub-to-sub jump) across 4 sub-windows. 420 features."


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
