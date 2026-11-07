"""DIVISIVE NORMALIZATION front-end for patient1 waveform -> ECoG high-gamma.

Paradigm (Carandini-Heeger): the response of every cochlear channel at every
time is DIVIDED by a pooled drive from its spectrotemporal neighborhood. A bank
of pools with different shapes (narrow vs broad in frequency, fast vs slow in
time, causal vs acausal) and different exponents produces multiple normalized
representations -- each captures contrast / gain invariance that a LINEAR map
over the bands cannot reconstruct.

All operations are closed-form (FFT, cumsum, separable conv). No fitted
parameters. No optimizer.
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_E"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cochlear front-end
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=64, f_min=50.0, f_max=8000.0, order=4):
    """|H(f)|^2 of a 4th-order gammatone bank on the rFFT frequency grid."""
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
# Pooling primitives (separable Gaussian in frequency, EMA in time)
# ---------------------------------------------------------------------------

def gauss_freq_kernel(sigma_chans, device, exclude_self=False):
    """1D Gaussian kernel along channel axis (used for surround pooling)."""
    k = max(1, int(np.ceil(3.0 * sigma_chans)))
    coords = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma_chans ** 2))
    if exclude_self:
        g[k] = 0.0
    g = g / (g.sum() + 1e-12)
    return g.view(1, 1, -1), k


def conv_freq(x: torch.Tensor, kernel: torch.Tensor, pad: int) -> torch.Tensor:
    """Conv along channel axis of (T, C) using a (1,1,K) kernel."""
    xp = x.unsqueeze(1)  # (T, 1, C)
    xp = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    return torch.nn.functional.conv1d(xp, kernel).squeeze(1)


def causal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Causal exponential moving average along time axis 0."""
    if tau_frames <= 0.0:
        return x
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def acausal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Time-reversed EMA -- exponential lookahead."""
    if tau_frames <= 0.0:
        return x
    return torch.flip(causal_ema(torch.flip(x, dims=[0]), tau_frames), dims=[0])


def symmetric_time_pool(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Symmetric pool: average of causal + acausal EMA."""
    if tau_frames <= 0.0:
        return x
    return 0.5 * (causal_ema(x, tau_frames) + acausal_ema(x, tau_frames))


# ---------------------------------------------------------------------------
# Hilbert-style envelope extraction (NONLINEAR -- magnitude of complex demod)
# ---------------------------------------------------------------------------

_GABOR_KERNEL_CACHE: dict = {}


def _gabor_kernel_pair(rate_hz: float, fs: float, cycles: float, C: int, device):
    key = (round(rate_hz, 4), round(fs, 2), round(cycles, 2), C,
            device.type, device.index)
    if key not in _GABOR_KERNEL_CACHE:
        period = fs / max(rate_hz, 1e-3)
        sigma = cycles * period / (2.0 * np.pi)
        L = max(2, int(np.ceil(4.0 * sigma)))
        t = np.arange(-L, L + 1)
        g = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
        cos_k = g * np.cos(2.0 * np.pi * rate_hz * t / fs)
        sin_k = g * np.sin(2.0 * np.pi * rate_hz * t / fs)
        cos_k = cos_k - cos_k.mean(); sin_k = sin_k - sin_k.mean()
        norm = np.linalg.norm(cos_k) + 1e-12
        cos_k = cos_k / norm; sin_k = sin_k / norm
        ck = torch.from_numpy(cos_k.astype(np.float32)).to(device).view(1, 1, -1)
        sk = torch.from_numpy(sin_k.astype(np.float32)).to(device).view(1, 1, -1)
        ck = ck.expand(C, 1, 2 * L + 1).contiguous()
        sk = sk.expand(C, 1, 2 * L + 1).contiguous()
        _GABOR_KERNEL_CACHE[key] = (ck, sk, L)
    return _GABOR_KERNEL_CACHE[key]


def gabor_temporal_complex(x: torch.Tensor, rate_hz: float, fs: float = 30.0,
                            cycles: float = 1.5):
    """Returns (cos_resp, sin_resp) — the COMPLEX modulation response."""
    C = x.shape[1]
    ck, sk, L = _gabor_kernel_pair(rate_hz, fs, cycles, C, x.device)
    pad = L
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    c = torch.nn.functional.conv1d(xp_p, ck, groups=C)
    s = torch.nn.functional.conv1d(xp_p, sk, groups=C)
    c = c[..., :x.shape[0]].squeeze(0).transpose(0, 1).contiguous()
    s = s[..., :x.shape[0]].squeeze(0).transpose(0, 1).contiguous()
    return c, s


def gabor_temporal_envelope(x: torch.Tensor, rate_hz: float, fs: float = 30.0,
                             cycles: float = 1.5) -> torch.Tensor:
    """Magnitude of complex Gabor demod along time axis 0 of (T, C). Cached."""
    c, s = gabor_temporal_complex(x, rate_hz, fs, cycles)
    return torch.sqrt(c * c + s * s + 1e-12)


_STMF_KERNEL_CACHE: dict = {}


def stmf_envelope(x: torch.Tensor, omega_t: float, omega_f: float,
                   fs: float = 30.0, t_cycles: float = 1.5,
                   f_cycles: float = 1.5) -> torch.Tensor:
    """2D Chi-Shamma spectrotemporal modulation envelope. Cached."""
    key = (round(omega_t, 4), round(omega_f, 4), round(fs, 2),
            round(t_cycles, 2), round(f_cycles, 2),
            x.device.type, x.device.index)
    if key not in _STMF_KERNEL_CACHE:
        period_t = fs / max(abs(omega_t), 0.5)
        sigma_t = t_cycles * period_t / (2.0 * np.pi)
        Lt = max(1, int(np.ceil(3.0 * sigma_t)))
        sigma_f = f_cycles / max(abs(omega_f), 0.05) / (2.0 * np.pi)
        Lf = max(1, int(np.ceil(3.0 * sigma_f)))
        t = np.arange(-Lt, Lt + 1)
        f = np.arange(-Lf, Lf + 1)
        g = np.exp(-((t[:, None] ** 2) / (2.0 * sigma_t ** 2) +
                      (f[None, :] ** 2) / (2.0 * sigma_f ** 2)))
        phase = 2.0 * np.pi * omega_t * t[:, None] / fs + 2.0 * np.pi * omega_f * f[None, :]
        cos_k = (g * np.cos(phase)).astype(np.float32)
        sin_k = (g * np.sin(phase)).astype(np.float32)
        cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
        norm = float(np.linalg.norm(cos_k)) + 1e-12
        cos_k /= norm; sin_k /= norm
        cos_t = torch.from_numpy(cos_k).to(x.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
        sin_t = torch.from_numpy(sin_k).to(x.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
        _STMF_KERNEL_CACHE[key] = (cos_t, sin_t, Lt, Lf)
    cos_t, sin_t, Lt, Lf = _STMF_KERNEL_CACHE[key]
    xp = x.unsqueeze(0).unsqueeze(0)
    c = torch.nn.functional.conv2d(xp, cos_t, padding=(Lt, Lf))
    s = torch.nn.functional.conv2d(xp, sin_t, padding=(Lt, Lf))
    T, C = x.shape
    env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
    return env[:T, :C]


def gabor_spectral_envelope(x: torch.Tensor, omega_chans: float,
                              cycles: float = 1.5) -> torch.Tensor:
    """Magnitude of complex Gabor demod along channel axis 1 of (T, C).

    omega_chans = cycles per channel (the spectral 'ripple' frequency).
    Returns (T, C). Captures spectral ripple energy NONLINEARLY.
    """
    L = max(2, int(np.ceil(cycles / max(omega_chans, 1e-3)) + 4))
    c_coords = np.arange(-L, L + 1)
    sigma = cycles / max(omega_chans, 1e-3) / (2.0 * np.pi)
    g = np.exp(-(c_coords ** 2) / (2.0 * sigma ** 2))
    cos_k = g * np.cos(2.0 * np.pi * omega_chans * c_coords)
    sin_k = g * np.sin(2.0 * np.pi * omega_chans * c_coords)
    cos_k = cos_k - cos_k.mean()
    sin_k = sin_k - sin_k.mean()
    norm = np.linalg.norm(cos_k) + 1e-12
    cos_k = cos_k / norm
    sin_k = sin_k / norm
    ck = torch.from_numpy(cos_k.astype(np.float32)).to(x.device).view(1, 1, -1)
    sk = torch.from_numpy(sin_k.astype(np.float32)).to(x.device).view(1, 1, -1)
    pad = L
    xp = x.unsqueeze(1)  # (T, 1, C)
    xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
    c = torch.nn.functional.conv1d(xp_p, ck).squeeze(1)
    s = torch.nn.functional.conv1d(xp_p, sk).squeeze(1)
    env = torch.sqrt(c * c + s * s + 1e-12)
    return env


# ---------------------------------------------------------------------------
# Divisive normalization
# ---------------------------------------------------------------------------

# A "pool spec" = (freq_sigma_chans, time_tau_frames, time_kind, exclude_self)
# time_kind in {"causal", "acausal", "sym", "none"}
POOL_SPECS = [
    # Pure frequency surrounds (instantaneous DN across freq channels)
    (1.0,  0.0,  "none",    True),    # P0: narrow surround
    (3.0,  0.0,  "none",    True),    # P1
    (8.0,  0.0,  "none",    True),    # P2
    (24.0, 0.0,  "none",    True),    # P3: near-global freq context
    # Pure temporal adaptation (per channel, no freq pooling)
    (0.0,  1.5,  "causal",  False),   # P4: very fast adaptation
    (0.0,  4.5,  "causal",  False),   # P5
    (0.0,  15.0, "causal",  False),   # P6
    (0.0,  60.0, "causal",  False),   # P7
    # Joint freq + causal time pools (classic Carandini-Heeger DN)
    (3.0,  4.5,  "causal",  True),    # P8
    (3.0,  15.0, "causal",  True),    # P9
    (8.0,  4.5,  "causal",  True),    # P10
    (8.0,  15.0, "causal",  True),    # P11
    (8.0,  60.0, "causal",  True),    # P12
    (24.0, 15.0, "causal",  True),    # P13: wide freq + medium time
    # Symmetric joint pools -- DN with bidirectional context
    (3.0,  15.0, "sym",     True),    # P14
    (8.0,  15.0, "sym",     True),    # P15
    (8.0,  60.0, "sym",     True),    # P16
]

# Exponents for divisive normalization
# p=0.3 Stevens-loudness; p=1 raw; p=2 energy; p=3 superlinear contrast.
EXPONENTS = (0.3, 1.0, 2.0, 3.0)
# Semi-saturation fraction of per-chunk median pool value
SIGMA_FRAC = 0.1

# Center-surround opponency contrasts: pairs of pool indices into POOL_SPECS.
# Each pair (narrow_idx, wide_idx) yields log(pool_wide) - log(pool_narrow)
# = log-domain center-surround signal -- nonlinearly captures local prominence
# above background. Done per exponent.
OPPONENCY_PAIRS = (
    (0, 2),   # narrow freq vs wide freq, both instantaneous
    (0, 3),   # narrow freq vs near-global
    (8, 12),  # narrow joint pool vs wide joint pool
    (5, 7),   # fast vs slow temporal adaptation
    (8, 10),  # short-time joint vs long-time joint at same freq
)


def time_pool(x: torch.Tensor, tau: float, kind: str) -> torch.Tensor:
    if kind == "none" or tau <= 0.0:
        return x
    if kind == "causal":
        return causal_ema(x, tau)
    if kind == "acausal":
        return acausal_ema(x, tau)
    if kind == "sym":
        return symmetric_time_pool(x, tau)
    raise ValueError(kind)


def build_pool(e_p: torch.Tensor, freq_kernel, freq_pad,
               tau: float, time_kind: str) -> torch.Tensor:
    """Apply freq + time pooling to e_p. Returns (T, C) pool tensor."""
    if freq_kernel is not None:
        pool = conv_freq(e_p, freq_kernel, freq_pad)
    else:
        pool = e_p
    return time_pool(pool, tau, time_kind)


def dn_outputs(e_p: torch.Tensor, pool: torch.Tensor,
               sigma_frac: float = SIGMA_FRAC):
    """Carandini-Heeger DN outputs.
        log_dn   = log(e_p) - log(sigma + pool)        (response / context)
        log_pool = log(sigma + pool)                    (context magnitude)
        log_resid = log(1 + clip(e_p - pool, min=0))    (above-context drive)
    sigma = sigma_frac * median(pool) per chunk (scalar). All shape (T, C)."""
    sigma = sigma_frac * pool.median().detach()
    sigma = sigma.clamp(min=1e-8)
    denom = (sigma + pool).clamp(min=1e-12)
    log_dn = torch.log(e_p.clamp(min=1e-12)) - torch.log(denom)
    log_pool = torch.log(denom)
    resid = (e_p - pool).clamp(min=0.0)
    log_resid = torch.log1p(resid / sigma.clamp(min=1e-12))
    return log_dn, log_pool, log_resid


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Divisive-normalization bank on a 4096-sample gammatone cochleagram."""

    win = 4096
    n_params = 0
    N_FILTERS = 48

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.N_FILTERS).to(device)
        # Pre-build pool freq kernels
        self._freq_kernels = {}
        for spec in POOL_SPECS:
            fs, _, _, excl = spec
            key = (round(fs, 3), excl)
            if key not in self._freq_kernels and fs > 0.0:
                self._freq_kernels[key] = gauss_freq_kernel(fs, device, exclude_self=excl)

    def _get_freq_kernel(self, fs: float, excl: bool):
        if fs <= 0.0:
            return None, 0
        return self._freq_kernels[(round(fs, 3), excl)]

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        p = power_spectrum(x, self.win, self.device)
        e = (p @ self.fb).clamp(min=1e-12)  # (T, F) cochleagram energy
        log_e = torch.log(e)
        e_p2 = e ** 2.0

        feats = []
        # Anchor: raw log-cochleagram (the ridge's linear baseline)
        feats.append(log_e)

        # --- Stage 1: build the DIVISIVELY NORMALIZED COCHLEAGRAMS ----------
        # Each is a (T, F) map that has had its local context divided out.
        # These BECOME the input to the modulation analysis stage below.
        DN_FRONTENDS = [
            # (pool_spec_idx_in_POOL_SPECS, label)
            (9,  "dn_narrow_med"),   # narrow freq + 15-frame causal
            (11, "dn_wide_med"),     # wide freq + 15-frame causal
            (12, "dn_wide_slow"),    # wide freq + 60-frame causal
            (15, "dn_sym_med"),      # wide freq + 15-frame symmetric
        ]
        dn_maps = {}
        for idx, label in DN_FRONTENDS:
            fs, tau, kind, excl = POOL_SPECS[idx]
            fk, fp = self._get_freq_kernel(fs, excl)
            pool = build_pool(e_p2, fk, fp, tau, kind)
            log_dn, _, _ = dn_outputs(e_p2, pool)
            dn_maps[label] = log_dn
            feats.append(log_dn)  # also include the DN-cochleagram itself

        # --- Stage 2: NONLINEAR MODULATION ENVELOPES (Hilbert magnitude) ---
        # Computed on top of the DN-normalized cochleagrams (and log_e).
        # |H_omega * x| is intrinsically nonlinear -> ridge can never recover.
        BASE_MAPS = [
            ("log_e", log_e),
            ("dn_narrow_med", dn_maps["dn_narrow_med"]),
            ("dn_wide_med", dn_maps["dn_wide_med"]),
            ("dn_wide_slow", dn_maps["dn_wide_slow"]),
        ]
        TEMPORAL_RATES_HZ = (2.0, 4.0, 6.0, 8.0, 12.0)
        # Hold envelopes for stage 2c.
        env_bank = {}
        for name, m in BASE_MAPS:
            env_bank[name] = []
            for r in TEMPORAL_RATES_HZ:
                env = gabor_temporal_envelope(m, r)
                feats.append(env)
                env_bank[name].append(env)

        # --- Stage 2c: FUTURE + PAST DELTA per band --------------------------
        # Explicit lookahead/lookback: log_e[t+k] - log_e[t]. The harness's
        # ndelays only gives PAST delays of features; this gives the ridge
        # per-band future-context change signals.
        for k in (3, 6, 12):
            fut = torch.cat([log_e[k:], log_e[-1:].expand(k, -1)], dim=0)
            feats.append(fut - log_e)
            past = torch.cat([log_e[:1].expand(k, -1), log_e[:-k]], dim=0)
            feats.append(log_e - past)
        # Future delta on DN cochleagram (different gain regime)
        dn_main = dn_maps["dn_wide_med"]
        for k in (3, 6):
            fut_dn = torch.cat([dn_main[k:], dn_main[-1:].expand(k, -1)], dim=0)
            feats.append(fut_dn - dn_main)

        # --- Stage 2d: SLOW TONIC COCHLEAGRAMS ------------------------------
        # Low-passed log_e at multiple slow time constants. Provides per-band
        # prosodic-scale context (the "state" the brain is in).
        for tau in (60.0, 150.0):
            feats.append(causal_ema(log_e, tau_frames=tau))
            feats.append(acausal_ema(log_e, tau_frames=tau))

        # --- Stage 2b: JOINT spectrotemporal modulation (Chi-Shamma STMF).
        # 2D complex Gabor magnitudes on the DN-cochleagram. Sign of omega_f
        # picks upward (+) vs downward (-) ripples. Done on the most-canonical
        # DN cochleagram (wide freq, medium time) for richness without dim blow-up.
        STMF_BANK = (
            (2.0,  0.12), (2.0,  -0.12),
            (4.0,  0.12), (4.0,  -0.12),
            (4.0,  0.25), (4.0,  -0.25),
            (8.0,  0.12), (8.0,  -0.12),
            (8.0,  0.25), (8.0,  -0.25),
            (12.0, 0.12), (12.0, -0.12),
        )
        stmf_base = dn_maps["dn_wide_med"]
        for (omega_t, omega_f) in STMF_BANK:
            feats.append(stmf_envelope(stmf_base, omega_t, omega_f))

        # --- Stage 3: ONSET / TRANSIENT detector on the normalized cochleagram
        # Rectified positive temporal derivative -- captures band-wise onsets.
        for name in ("log_e", "dn_wide_med"):
            m = log_e if name == "log_e" else dn_maps[name]
            d = torch.zeros_like(m)
            d[1:] = m[1:] - m[:-1]
            feats.append(d.clamp(min=0.0))   # onset
            feats.append((-d).clamp(min=0.0))  # offset

        # --- Stage 4: Center-surround OPPONENCY at p=2.
        # log(pool_wide) - log(pool_narrow) for representative pairs.
        for exponent in (2.0,):
            e_p = e ** exponent
            pool_cache = {}
            for (i_n, i_w) in OPPONENCY_PAIRS:
                for idx in (i_n, i_w):
                    if idx not in pool_cache:
                        fs, tau, kind, excl = POOL_SPECS[idx]
                        fk, fp = self._get_freq_kernel(fs, excl)
                        pool_cache[idx] = build_pool(e_p, fk, fp, tau, kind)
                pn = pool_cache[i_n]; pw = pool_cache[i_w]
                sn = (0.01 * pn.median().detach()).clamp(min=1e-12)
                sw = (0.01 * pw.median().detach()).clamp(min=1e-12)
                opp = torch.log(pw + sw) - torch.log(pn + sn)
                feats.append(opp)

        feat = torch.cat(feats, dim=1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.float().cpu().numpy()


model_shorthand_name = "DN-v9"
model_description = (
    "DN-v9: v8-base swaps normalized tonotopy for explicit FUTURE+PAST DELTAS "
    "per band (k=3,6,12 on log_e + k=3,6 on dn_wide_med, 8*48=384 dim) and "
    "SLOW TONIC log_e cochleagrams (causal+acausal at tau=60,150, 4*48=192 dim). "
    "Total ~2784. v8=0.0593."
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
