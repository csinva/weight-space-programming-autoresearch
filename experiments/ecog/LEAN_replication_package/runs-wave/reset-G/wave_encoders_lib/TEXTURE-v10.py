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

    # 2-D STMF rate-scale (Chi-Shamma) — (temporal Hz, spectral cyc/channel).
    # Positive scale = upward sweep; negative = downward sweep; 0 = pure temporal.
    STMF_RS = (
        (1.0, 0.0),  (2.0, 0.0),  (4.0, 0.0),  (8.0, 0.0),  (12.0, 0.0),
        (2.0, 0.125), (2.0, -0.125),
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.125), (4.0, -0.125),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (0.0, 0.125), (0.0, 0.25), (0.0, 0.5),
    )

    SHORT_WIN = 1024     # second cochleagram window (~23 ms) — finer temporal grain
    SHORT_BANDS = 20

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.N_BANDS).to(device)
        self.fb_short = gammatone_powerbank(self.SHORT_WIN,
                                             n_filters=self.SHORT_BANDS).to(device)
        # Precompute modulation Gabor kernels
        self._mod_kernels = {}
        for r in self.MOD_RATES:
            self._mod_kernels[r] = make_temporal_gabor(r, 30.0, device, torch.float32)
        self._stmf_cache: dict[tuple, tuple] = {}

    def _cochleagram(self, windows_t: torch.Tensor) -> torch.Tensor:
        """Compute (T, F) compressed cochlear envelope from per-frame windows."""
        p = power_spectrum(windows_t, self.win, self.device)        # (T, n_freqs)
        e = p @ self.fb                                              # (T, F)
        # Power-law compression — closed-form, NOT a learned curve
        c = (e + 1e-10) ** self.COCHLEAR_COMPRESSION
        return c, p

    F0_CANDIDATES = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0, 230.0, 270.0,
                      320.0, 380.0, 450.0)

    def _pitch_block(self, p: torch.Tensor) -> list[torch.Tensor]:
        """Closed-form harmonic-comb pitch salience + autocorrelation HNR
        derived from the per-window power spectrum (no re-windowing).

        Harmonic salience: for each F0 candidate, sum log-power at the first n
        harmonic bins minus the global log-power baseline. The argmax over
        candidates gives the running pitch estimate.

        HNR: normalized autocorrelation from IFFT of power spectrum;
        peak/(1-peak) is the harmonic-to-noise ratio."""
        feats = []
        n_freqs = p.shape[1]
        freq_res = SR / self.win
        log_p = torch.log(torch.clamp(p, min=1e-10))
        baseline = log_p.mean(dim=1, keepdim=True)
        # Harmonic comb for each F0 candidate
        cand_feats = []
        f0_t = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32, device=p.device)
        for i, f0 in enumerate(self.F0_CANDIDATES):
            ks = np.arange(1, 9)
            bins = np.clip((f0 * ks / freq_res).astype(int), 0, n_freqs - 1)
            bins_t = torch.from_numpy(bins).to(p.device)
            sal = log_p[:, bins_t].mean(dim=1, keepdim=True) - baseline
            cand_feats.append(sal)
        salience = torch.cat(cand_feats, dim=1)                        # (T, F0)
        feats.append(salience)
        # Best F0 and its strength + log of pitch
        best_strength, best_idx = salience.max(dim=1, keepdim=True)
        log_f0 = torch.log(f0_t[best_idx.squeeze(1)].unsqueeze(1).clamp(min=20.0))
        feats.append(best_strength)
        feats.append(log_f0)
        # Multi-tau dynamics (highpass) of pitch + strength
        for tau in (4.5, 15.0, 60.0):
            feats.append(log_f0 - leaky_lowpass(log_f0, tau))
            feats.append(best_strength - leaky_lowpass(best_strength, tau))

        # Autocorrelation HNR
        full = torch.fft.irfft(p, n=self.win)                          # (T, win)
        acf0 = full[:, :1] + 1e-12
        acf_n = full / acf0
        lag_min = max(1, int(SR / 400.0))
        lag_max = min(self.win - 1, int(SR / 75.0))
        region = acf_n[:, lag_min:lag_max + 1]
        peak_val, peak_lag = region.max(dim=1, keepdim=True)
        peak_val = peak_val.clamp(min=0.0, max=0.999)
        hnr_db = 10.0 * torch.log10(peak_val.clamp(min=1e-3) /
                                      (1.0 - peak_val).clamp(min=1e-3))
        feats.append(peak_val)
        feats.append(hnr_db)
        feats.append(torch.log(peak_lag.float().clamp(min=1.0) + 0.0))
        # HNR derivative (voicing onset/offset)
        hd = torch.zeros_like(hnr_db)
        hd[1:] = hnr_db[1:] - hnr_db[:-1]
        feats.append(hd)
        feats.append(hd.clamp(min=0))
        feats.append((-hd).clamp(min=0))
        # Multi-tau dynamics of HNR
        for tau in (4.5, 15.0, 60.0):
            feats.append(hnr_db - leaky_lowpass(hnr_db, tau))
        return feats

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
              env_{b,r}(t) = |Gabor_r * c_b|
           Then compress and exponentially-smooth at multiple tau.
           ALSO compute running variance of env_{b,r} (modulation 'depth' nonlinearity).
        """
        feats = []
        envs_by_rate = {}                          # store for cross-channel C1
        for r in self.MOD_RATES:
            cos_k, sin_k = self._mod_kernels[r]
            cc = conv_temporal(c, cos_k)
            ss = conv_temporal(c, sin_k)
            env = torch.sqrt(cc * cc + ss * ss + 1e-12)       # (T, F)
            log_env = torch.log(env.clamp(min=1e-8))
            envs_by_rate[r] = env
            feats.append(log_env)                              # all bands' mod power
            for tau in (15.0, 60.0):
                # running mean+variance of modulation envelope (nonlinear in c)
                mu = leaky_lowpass(env, tau)
                var = leaky_lowpass((env - mu) ** 2, tau).clamp(min=1e-10)
                feats.append(torch.log(mu.clamp(min=1e-8)))
                feats.append(torch.log(var))
                feats.append(torch.log((var / (mu * mu + 1e-10) + 1e-8)))  # mod CoV
        return feats, envs_by_rate

    def _cross_channel_modulation_corr(
            self, envs_by_rate: dict[float, torch.Tensor]) -> list[torch.Tensor]:
        """C1: cross-channel running correlation of modulation envelopes at the
        SAME modulation rate. Selected band offsets."""
        feats = []
        for r, env in envs_by_rate.items():
            F = env.shape[1]
            for offset in (2, 4, 8):
                a = env[:, :F - offset]
                b = env[:, offset:]
                for tau in (30.0, 150.0):
                    rho = running_corr(a, b, tau)
                    feats.append(rho)
        return feats

    def _stmf_envelope(self, log_e: torch.Tensor, omega_t: float, omega_f: float,
                        fs: float = 30.0) -> torch.Tensor:
        """Closed-form Chi-Shamma 2-D Gabor envelope on (T, F) cochleagram.
        Returns (T, F). Stride-1 conv with cached kernels."""
        key = (round(omega_t, 3), round(omega_f, 3))
        if key not in self._stmf_cache:
            period_t = fs / max(abs(omega_t), 0.5)
            sigma_t = period_t / (2 * np.pi)
            Lt = max(1, int(np.ceil(3 * sigma_t)))
            sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi)
            Lf = max(1, int(np.ceil(3 * sigma_f)))
            t = np.arange(-Lt, Lt + 1)
            f = np.arange(-Lf, Lf + 1)
            g = np.exp(-((t[:, None] ** 2) / (2 * sigma_t ** 2) +
                          (f[None, :] ** 2) / (2 * sigma_f ** 2)))
            phase = 2 * np.pi * omega_t * t[:, None] / fs + 2 * np.pi * omega_f * f[None, :]
            cos_k = (g * np.cos(phase)).astype(np.float32)
            sin_k = (g * np.sin(phase)).astype(np.float32)
            cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
            self._stmf_cache[key] = (
                torch.from_numpy(cos_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1),
                torch.from_numpy(sin_k).to(self.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1),
                Lt, Lf,
            )
        cos_k, sin_k, Lt, Lf = self._stmf_cache[key]
        xp = log_e.unsqueeze(0).unsqueeze(0)            # (1,1,T,F)
        c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
        s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
        T, F = log_e.shape
        env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
        return env[:T, :F]

    def _stmf_block(self, log_c: torch.Tensor) -> list[torch.Tensor]:
        """For each (omega_t, omega_f), compute the 2-D Gabor envelope —
        full (T, F) feature map per filter."""
        feats = []
        for (ot, of) in self.STMF_RS:
            feats.append(self._stmf_envelope(log_c, ot, of))
        return feats

    def _within_channel_cross_rate_corr(
            self, envs_by_rate: dict[float, torch.Tensor]) -> list[torch.Tensor]:
        """C2: within-channel correlation between modulation envelopes at
        DIFFERENT modulation rates (amplitude-amplitude coupling across rates)."""
        feats = []
        rates = list(envs_by_rate.keys())
        # Pair rate i with rate i+1 (adjacent octave)
        for i in range(len(rates) - 1):
            ea = envs_by_rate[rates[i]]
            eb = envs_by_rate[rates[i + 1]]
            for tau in (30.0, 150.0):
                rho = running_corr(ea, eb, tau)
                feats.append(rho)
        return feats

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        c, p = self._cochleagram(x)                             # (T, F), (T, n_freqs)
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
        mod_feats, envs_by_rate = self._modulation_block(log_c)
        feats.extend(mod_feats)

        # 5. C1 cross-channel modulation correlations
        feats.extend(self._cross_channel_modulation_corr(envs_by_rate))

        # 6. C2 within-channel cross-rate modulation correlations
        feats.extend(self._within_channel_cross_rate_corr(envs_by_rate))

        # 7. 2-D STMF spectro-temporal modulation envelopes (rate-scale Gabors).
        #    Both on log_c (multiplicative) and on raw compressed c (additive).
        stmf_log = self._stmf_block(log_c)
        feats.extend(stmf_log)
        stmf_raw = self._stmf_block(c)
        feats.extend(stmf_raw)
        # Spectrally-pooled STMF (mean across bands per filter) — running
        # variance and skew capture the "amount of joint modulation activity".
        for f_log in stmf_log:
            pooled = f_log.mean(dim=1, keepdim=True)                  # (T, 1)
            _, var, skew, _ = running_moments(pooled, 60.0)
            feats.append(torch.log(var.clamp(min=1e-10)))
            feats.append(skew.clamp(-10.0, 10.0))

        # 8. Pitch / HNR autocorrelation block — strongly nonlinear ratios of
        #    ACF peaks and harmonic-comb sums of log power.
        feats.extend(self._pitch_block(p))

        # 9. ON/OFF rectified AGC: half-wave-rectified (log_c - EMA_mu)/EMA_std,
        #    pos and neg channels at one tau. Compact, biophysically motivated.
        mu_t = leaky_lowpass(log_c, 15.0)
        var_t = leaky_lowpass((log_c - mu_t) ** 2, 15.0).clamp(min=1e-10)
        z = (log_c - mu_t) / torch.sqrt(var_t)
        feats.append(z.clamp(min=0.0))
        feats.append((-z).clamp(min=0.0))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "TEXTURE-v10"
model_description = (
    "TEXTURE v10: v9 + dual-domain STMF (raw cube-root c AND log_c) — captures "
    "additive vs multiplicative spectro-temporal modulations. + spectrally-"
    "pooled STMF running variance/skew per filter (texture of texture). "
    "v9=0.0631."
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
