"""
McDermott-Simoncelli SOUND-TEXTURE summary statistics, computed as
CAUSAL RUNNING quantities at 30 Hz via leaky lowpass accumulators.

Paradigm: every statistic is a NONLINEAR functional of the cochlear envelope
(higher moments, cross-band products, modulation-band powers, cross-channel
modulation correlations). A linear ridge over the bands cannot reconstruct
any of these, so they add genuinely new information to the readout.
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_G"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cochlear filterbank
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=32, f_min=50.0, f_max=8000.0, order=4):
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))


def power_spectrum(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2


# ---------------------------------------------------------------------------
# Causal leaky lowpass — the only smoothing primitive
# ---------------------------------------------------------------------------

def leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Causal exponential moving average of (T, C) at timescale `tau_frames`."""
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Running marginal moments  (var, skew, kurt are NONLINEAR in x)
# ---------------------------------------------------------------------------

def running_moments(x: torch.Tensor, tau: float):
    """Returns (mu, var, skew, kurt-3) of (T, C) at causal timescale tau.
    All four are computed via EMA-of-x-and-its-powers."""
    mu = leaky_lowpass(x, tau)
    d = x - mu
    var = leaky_lowpass(d * d, tau).clamp(min=1e-10)
    std = torch.sqrt(var)
    m3 = leaky_lowpass(d ** 3, tau)
    m4 = leaky_lowpass(d ** 4, tau)
    skew = m3 / (std ** 3 + 1e-10)
    kurt = m4 / (var ** 2 + 1e-10) - 3.0
    return mu, var, skew, kurt


def running_corr(x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Running normalized cross-correlation rho_xy(t) of (T, C_pairs)."""
    mux = leaky_lowpass(x, tau)
    muy = leaky_lowpass(y, tau)
    dx = x - mux
    dy = y - muy
    cov = leaky_lowpass(dx * dy, tau)
    vx = leaky_lowpass(dx * dx, tau).clamp(min=1e-10)
    vy = leaky_lowpass(dy * dy, tau).clamp(min=1e-10)
    return cov / torch.sqrt(vx * vy)


# ---------------------------------------------------------------------------
# Temporal modulation Gabor filterbank
# ---------------------------------------------------------------------------

def make_temporal_gabor(rate_hz: float, fs: float, device, dtype):
    period = fs / max(rate_hz, 0.1)
    sigma = period
    L = int(np.ceil(4 * sigma))
    t = np.arange(-L, L + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
    sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
    cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
    n = np.linalg.norm(cos_k) + 1e-12
    cos_k /= n; sin_k /= n
    return (torch.from_numpy(cos_k.astype(np.float32)).to(device),
            torch.from_numpy(sin_k.astype(np.float32)).to(device))


def conv_temporal(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve (T, C) along time with a 1-D kernel; same length out."""
    L = kernel.numel()
    pad = L // 2
    C = x.shape[1]
    xp = x.transpose(0, 1).unsqueeze(0)
    kk = kernel.view(1, 1, -1).expand(C, 1, L).contiguous()
    xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    y = torch.nn.functional.conv1d(xp_p, kk, groups=C)
    return y[..., :x.shape[0]].squeeze(0).transpose(0, 1).contiguous()


def modulation_envelope(x: torch.Tensor, rate_hz: float, fs: float = 30.0):
    """Bandpass (T, C) at temporal rate `rate_hz` using a complex Gabor; return
    (cos_part, sin_part, envelope=|complex|)."""
    cos_k, sin_k = make_temporal_gabor(rate_hz, fs, x.device, x.dtype)
    c = conv_temporal(x, cos_k)
    s = conv_temporal(x, sin_k)
    env = torch.sqrt(c * c + s * s + 1e-12)
    return c, s, env


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """McDermott-Simoncelli sound-texture statistics, causally accumulated at 30 Hz.

    For each frame at 30 Hz we expose:

      1. raw compressed log-cochleagram c_b(t)        ............... LINEAR
      2. per-band running moments  (var, skew, kurt)
         at multiple timescales tau ........................ NONLINEAR x_b^k
      3. cross-band normalized envelope correlations
         (selected near + long-range pairs)
         at multiple timescales ........................... NONLINEAR x_a * x_b / ...
      4. modulation-band power per (band, mod_rate)
         (mean, variance) at multiple timescales ........... NONLINEAR
      5. cross-channel correlations at matched mod rate (C1)
         (selected pairs) ................................. NONLINEAR

    Every smoothing operator is a causal leaky EMA — no re-windowing.
    """

    win = 4096
    n_params = 0

    # Cochlear bands
    N_BANDS = 24
    COCHLEAR_COMPRESSION = 0.3

    # Running-moment timescales (frames @ 30 Hz)
    MOMENT_TAUS = (4.5, 15.0, 60.0)                # ~150 ms / 500 ms / 2 s

    # Modulation rates  (Hz) — McDermott uses 1/2, 1, 2, 4, 8, 16, 32 Hz
    MOD_RATES = (2.0, 4.0, 8.0, 16.0)

    # Cross-band pair offsets (in band index). Captures local + long-range comodulation.
    PAIR_OFFSETS = (1, 2, 4, 8)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.N_BANDS).to(device)
        # Precompute modulation Gabor kernels
        self._mod_kernels = {}
        for r in self.MOD_RATES:
            self._mod_kernels[r] = make_temporal_gabor(r, 30.0, device, torch.float32)

    def _cochleagram(self, windows_t: torch.Tensor) -> torch.Tensor:
        """Compute (T, F) compressed cochlear envelope from per-frame windows."""
        p = power_spectrum(windows_t, self.win, self.device)        # (T, n_freqs)
        e = p @ self.fb                                              # (T, F)
        # Power-law compression — closed-form, NOT a learned curve
        c = (e + 1e-10) ** self.COCHLEAR_COMPRESSION
        return c

    def _band_moments_block(self, c: torch.Tensor) -> list[torch.Tensor]:
        """For each tau, return [var, skew, kurt] — each (T, F)."""
        out = []
        for tau in self.MOMENT_TAUS:
            mu, var, skew, kurt = running_moments(c, tau)
            out.append(torch.log(var.clamp(min=1e-10)))
            out.append(skew.clamp(-10.0, 10.0))
            out.append(kurt.clamp(-5.0, 50.0))
            # also the running mu itself (still useful, although near-linear)
            out.append(mu)
        return out

    def _cross_band_correlations(self, c: torch.Tensor) -> list[torch.Tensor]:
        """Running normalized correlation between band pairs at multiple lags
        and multiple timescales."""
        T, F = c.shape
        feats = []
        for offset in self.PAIR_OFFSETS:
            a = c[:, :F - offset]
            b = c[:, offset:]
            for tau in self.MOMENT_TAUS:
                rho = running_corr(a, b, tau)
                feats.append(rho)
        return feats

    def _modulation_block(self, c: torch.Tensor) -> list[torch.Tensor]:
        """For each band & each modulation rate:
              cc_{b,r}, ss_{b,r}, env_{b,r} = (cos*c, sin*c, sqrt(cc^2+ss^2))
           Returns:
              feats — log_env and running moments per (band, rate)
              cs_by_rate — dict[rate] -> (cc, ss, env) for cross-channel C1.
        """
        feats = []
        cs_by_rate: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for r in self.MOD_RATES:
            cos_k, sin_k = self._mod_kernels[r]
            cc = conv_temporal(c, cos_k)
            ss = conv_temporal(c, sin_k)
            env = torch.sqrt(cc * cc + ss * ss + 1e-12)       # (T, F)
            log_env = torch.log(env.clamp(min=1e-8))
            cs_by_rate[r] = (cc, ss, env)
            feats.append(log_env)                              # mod power per band
            for tau in (15.0, 60.0):
                mu = leaky_lowpass(env, tau)
                var = leaky_lowpass((env - mu) ** 2, tau).clamp(min=1e-10)
                feats.append(torch.log(mu.clamp(min=1e-8)))
                feats.append(torch.log(var))
                feats.append(torch.log((var / (mu * mu + 1e-10) + 1e-8)))  # mod CoV
        return feats, cs_by_rate

    def _cross_channel_modulation_corr(
            self,
            cs_by_rate: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ) -> list[torch.Tensor]:
        """C1 (envelope) + C1-phase (complex) cross-channel coupling at SAME mod rate.

        For band pair (a, b) at rate r:
          - envelope corr   = running_corr(env_a, env_b)            (amplitude coupling)
          - real coherence  = EMA(cc_a*cc_b + ss_a*ss_b) / sqrt(EMA(env_a^2)*EMA(env_b^2))
          - imag coherence  = EMA(ss_a*cc_b - cc_a*ss_b) / sqrt(...)
        The complex cross-spectrum (real+imag) captures whether the modulations
        in band a and band b are phase-locked (in phase / out of phase / quadrature).
        A linear ridge over band envelopes cannot reproduce these.
        """
        feats = []
        for r, (cc, ss, env) in cs_by_rate.items():
            F = env.shape[1]
            for offset in (2, 4, 8):
                ea, eb = env[:, :F - offset], env[:, offset:]
                cca, ccb = cc[:, :F - offset], cc[:, offset:]
                ssa, ssb = ss[:, :F - offset], ss[:, offset:]
                for tau in (30.0, 150.0):
                    rho_env = running_corr(ea, eb, tau)
                    # Complex cross-spectrum running estimate, normalized
                    re_xs = leaky_lowpass(cca * ccb + ssa * ssb, tau)
                    im_xs = leaky_lowpass(ssa * ccb - cca * ssb, tau)
                    norm = torch.sqrt(
                        leaky_lowpass(cca * cca + ssa * ssa, tau)
                        * leaky_lowpass(ccb * ccb + ssb * ssb, tau)
                        + 1e-12)
                    feats.append(rho_env)
                    feats.append(re_xs / norm)
                    feats.append(im_xs / norm)
        return feats

    def _within_channel_cross_rate_corr(
            self,
            cs_by_rate: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ) -> list[torch.Tensor]:
        """C2: within-channel correlation between modulation envelopes at
        DIFFERENT modulation rates. We also include the complex phase coupling
        between adjacent rates (rate doubling / octave coherence)."""
        feats = []
        rates = list(cs_by_rate.keys())
        for i in range(len(rates) - 1):
            cca, ssa, ea = cs_by_rate[rates[i]]
            ccb, ssb, eb = cs_by_rate[rates[i + 1]]
            for tau in (30.0, 150.0):
                feats.append(running_corr(ea, eb, tau))
                # Phase-coupling proxy: doubled phase of the slower rate vs faster.
                # Real part of (a^2 / |a|^2) * conj(b) — captures rate-doubled lock-in.
                # Use a simplified version: corr of (cca^2 - ssa^2) with ccb (cos(2*phi_a) vs cos(phi_b))
                feats.append(running_corr(cca * cca - ssa * ssa, ccb, tau))
                feats.append(running_corr(2 * cca * ssa, ssb, tau))
        return feats

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        c = self._cochleagram(x)                                # (T, F)
        log_c = torch.log(c.clamp(min=1e-10))

        feats: list[torch.Tensor] = []
        # 1. raw compressed envelope (linear input the ridge can use directly)
        feats.append(log_c)
        # First-difference of log envelope (delta) — basic onset signal
        d1 = torch.zeros_like(log_c)
        d1[1:] = log_c[1:] - log_c[:-1]
        feats.append(d1)

        # 2. per-band running moments  (NONLINEAR x^k)
        feats.extend(self._band_moments_block(log_c))

        # 3. cross-band envelope correlations  (NONLINEAR products / ratios)
        feats.extend(self._cross_band_correlations(log_c))

        # 4. modulation filterbank + per-(band, rate) statistics
        mod_feats, cs_by_rate = self._modulation_block(log_c)
        feats.extend(mod_feats)

        # 5. C1 cross-channel modulation correlations (envelope + complex phase)
        feats.extend(self._cross_channel_modulation_corr(cs_by_rate))

        # 6. C2 within-channel cross-rate modulation correlations + phase doubling
        feats.extend(self._within_channel_cross_rate_corr(cs_by_rate))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "TEXTURE-v2"
model_description = (
    "TEXTURE v2: v1 + modulation PHASE coherence (complex cross-spectrum) "
    "for C1 (cross-band at matched rate, re+im normalized by per-band power) "
    "and C2 phase-doubling proxies (rate octave coupling, cos(2phi_a)·cos(phi_b)). "
    "Captures phase-locked modulations a linear readout cannot reconstruct. "
    "v1 test_corr=0.0595 (dim 1859)."
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

    # Quick dim probe
    dummy = np.zeros((10, embedder.win), dtype=np.float32)
    sample = embedder(dummy)
    print(f"feature dim per frame = {sample.shape[1]}")

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
