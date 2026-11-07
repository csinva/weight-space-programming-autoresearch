"""
Hand-designed CORTICAL SPECTRO-TEMPORAL MODULATION (Chi-Ru-Shamma) auditory
front-end: raw waveform -> patient1 ECoG high-gamma.

Iteration 6: AUDIO-RATE ENVELOPE MODULATION SPECTRUM + slow STMF.

The 30 Hz cochleagram is Nyquist-limited to 15 Hz modulation, so the previous
pure-STMF on the 30 Hz cochleagram (test_corr 0.0537) cannot see voicing
(F0 = 100-300 Hz) or fast formant transitions inside a 93-ms window. This
iteration adds an AUDIO-RATE modulation spectrum:

  For each 4096-sample window, build the per-frame ANALYTIC SIGNAL of each
  gammatone channel (FFT, gate to positive frequencies, IFFT, magnitude =
  envelope at audio rate). Take the within-window FFT of the envelope and
  integrate its energy in log-spaced modulation bands spanning 8-1024 Hz
  (covers AM, voicing-F0, roughness, formant-rate).

Combined with a slow Chi-Shamma STMF on the 30 Hz log-cochleagram for slow
rates (<= 4 Hz), the union spans modulation across 5 orders of magnitude.
"""
from __future__ import annotations

import argparse
import hashlib
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_B"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cochleagram (closed-form gammatone power-bank)
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters, f_min, f_max, order=4):
    """Magnitude-SQUARED gammatone filterbank in freq domain (for the
    power-spectrum cochleagram path).
    Returns (n_rfft, F) real and the center freqs.
    """
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32)), fcs


def gammatone_mag_freqdomain(win, n_filters, f_min, f_max, order=4):
    """MAGNITUDE (not squared) gammatone filterbank, for analytic-signal
    extraction. Returns (n_rfft, F) real and the center freqs.
    """
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H = (1.0 + (df / b[None, :]) ** 2) ** (-order / 2.0)
    return torch.from_numpy(H.astype(np.float32)), fcs


# ---------------------------------------------------------------------------
# STMF filterbank — closed-form 2D Gabor in (omega_t, omega_f) FFT domain
# ---------------------------------------------------------------------------

def _build_stmf_filter_bank(T: int, F: int, fs_t: float, oct_per_ch: float,
                              rates: tuple, scales: tuple,
                              sigma_rate_oct: float = 0.6,
                              sigma_scale_oct: float = 0.6,
                              device: str = "cuda"):
    """Build a (K, T, F) complex filterbank in 2D FFT domain.
    Each filter is an analytic half-plane Gaussian centered at (rate, ±scale, dir).
    """
    omega_t = np.fft.fftfreq(T, d=1.0 / fs_t).astype(np.float32)
    omega_f_cpc = np.fft.fftfreq(F, d=1.0).astype(np.float32)
    omega_f_cpo = omega_f_cpc / oct_per_ch
    OT, OF = np.meshgrid(omega_t, omega_f_cpo, indexing="ij")

    pos_t = (OT > 0).astype(np.float32)

    def add_gauss(mu_t, mu_f, sig_t_oct, sig_f_oct, mask):
        d_t = np.where(np.abs(OT) > 1e-6,
                        np.log2(np.maximum(np.abs(OT), 1e-6) / max(mu_t, 1e-6)),
                        1e3)
        d_f = np.where(np.abs(OF) > 1e-6,
                        np.log2(np.maximum(np.abs(OF), 1e-6) / max(mu_f, 1e-6)),
                        1e3)
        if mu_t < 1e-6:
            d_t = np.where(np.abs(OT) < 0.5, 0.0, 1e3)
        if mu_f < 1e-6:
            d_f = np.where(np.abs(OF) < 0.05, 0.0, 1e3)
        g = np.exp(-0.5 * (d_t / sig_t_oct) ** 2 - 0.5 * (d_f / sig_f_oct) ** 2)
        return (g * mask).astype(np.float32)

    filters = []
    labels = []
    for r in rates:
        for s in scales:
            mask_up = pos_t * (OF > 0).astype(np.float32)
            filters.append(add_gauss(r, s, sigma_rate_oct, sigma_scale_oct, mask_up))
            labels.append((r, s, +1))
            mask_dn = pos_t * (OF < 0).astype(np.float32)
            filters.append(add_gauss(r, s, sigma_rate_oct, sigma_scale_oct, mask_dn))
            labels.append((r, s, -1))
    for r in rates:
        mask_t = pos_t * (np.abs(OF) < 0.1).astype(np.float32)
        filters.append(add_gauss(r, 0.0, sigma_rate_oct, 1.0, mask_t))
        labels.append((r, 0.0, 0))
    for s in scales:
        mask_s_up = (np.abs(OT) < 0.5).astype(np.float32) * (OF > 0).astype(np.float32)
        filters.append(add_gauss(0.0, s, 1.0, sigma_scale_oct, mask_s_up))
        labels.append((0.0, s, +1))
        mask_s_dn = (np.abs(OT) < 0.5).astype(np.float32) * (OF < 0).astype(np.float32)
        filters.append(add_gauss(0.0, s, 1.0, sigma_scale_oct, mask_s_dn))
        labels.append((0.0, s, -1))

    f = np.stack(filters, axis=0).astype(np.complex64)
    return torch.from_numpy(f).to(device), labels


def stmf_apply(log_e: torch.Tensor, filt_bank: torch.Tensor) -> torch.Tensor:
    """Apply (K, T, F) complex STMF bank to (T, F) cochleagram via FFT2.
    Returns magnitude responses (K, T, F) real."""
    L = log_e - log_e.mean(dim=0, keepdim=True)
    LF = torch.fft.fft2(L)
    out = torch.fft.ifft2(filt_bank * LF.unsqueeze(0))
    return out.abs().contiguous()


# ---------------------------------------------------------------------------
# Audio-rate envelope modulation spectrum
# ---------------------------------------------------------------------------

def envelope_modulation_features(x: torch.Tensor, gamma_H: torch.Tensor,
                                   mod_band_edges_hz: tuple,
                                   win: int, device: str) -> torch.Tensor:
    """For each window x[i,:] and each gammatone channel:
      - extract analytic-signal envelope at audio rate (FFT, gate positive
        freqs only, IFFT, magnitude).
      - compute envelope spectrum (rfft of envelope).
      - integrate energy in log-spaced modulation bands.
    Returns (T, F, n_modes) log-energy per (frame, channel, modband).
    """
    T = x.shape[0]
    F_ch = gamma_H.shape[1]
    hann = torch.hann_window(win, periodic=True, device=device)
    X = torch.fft.rfft(x * hann, n=win)                            # (T, n_rfft)
    # Apply analytic gammatone per channel: (T, n_rfft, F)
    # Build full-spectrum complex with neg freqs zeroed (analytic signal):
    n_full = win
    full = torch.zeros(T, n_full, F_ch, dtype=torch.complex64, device=device)
    # Positive freqs (0..win//2) get gain * 2 (except DC and Nyquist get gain * 1)
    gain = gamma_H * 2.0
    gain[0] = gamma_H[0]
    gain[-1] = gamma_H[-1]
    Yf = X.unsqueeze(-1) * gain.unsqueeze(0)                       # (T, n_rfft, F)
    full[:, :Yf.shape[1], :] = Yf
    y = torch.fft.ifft(full, dim=1)                                 # (T, win, F) complex
    env = y.abs()                                                    # (T, win, F)
    # Window the envelope to reduce edge leakage:
    env = env * hann.view(1, -1, 1)
    E = torch.fft.rfft(env, dim=1)                                  # (T, win//2+1, F)
    mod_bin_hz = SR / win
    feats = []
    for lo, hi in zip(mod_band_edges_hz[:-1], mod_band_edges_hz[1:]):
        b0 = max(1, int(round(lo / mod_bin_hz)))
        b1 = min(E.shape[1], int(round(hi / mod_bin_hz)))
        if b1 <= b0:
            b1 = b0 + 1
        band_energy = (E[:, b0:b1, :].abs() ** 2).mean(dim=1)        # (T, F)
        feats.append(torch.log(band_energy + 1e-10).unsqueeze(-1))
    return torch.cat(feats, dim=-1)                                  # (T, F, n_modes)


# ---------------------------------------------------------------------------
# Slow Gabor envelope along time (second-order modulation)
# ---------------------------------------------------------------------------

def slow_gabor_envelope(x: torch.Tensor, rates_hz: tuple, fs: float = 30.0
                          ) -> torch.Tensor:
    """For each rate, compute the quadrature magnitude of a Gabor band-pass
    filter on the time axis (T, C) of `x`. Returns (T, len(rates) * C).

    Implemented in FFT domain.
    """
    T, C = x.shape
    Xf = torch.fft.fft(x, dim=0)                             # (T, C) complex
    omega = np.fft.fftfreq(T, d=1.0 / fs).astype(np.float32)
    outs = []
    for r in rates_hz:
        # Gaussian centered at +r only (analytic), sigma = max(0.3*r, 0.3)
        sig = max(0.3 * r, 0.3)
        d = np.where(omega > 1e-6,
                      np.log2(np.maximum(omega, 1e-6) / r),
                      1e3)
        g = np.exp(-0.5 * (d / 0.8) ** 2).astype(np.float32)
        # zero negative freqs (analytic)
        g[omega <= 0] = 0.0
        gt = torch.from_numpy(g).to(x.device).unsqueeze(1)    # (T, 1)
        y = torch.fft.ifft(Xf * gt, dim=0)                    # (T, C) complex
        outs.append(y.abs())
    return torch.cat(outs, dim=1).contiguous()                # (T, len(rates)*C)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _chunk_hash(windows: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(np.asarray(windows.shape, dtype=np.int64).tobytes())
    h.update(windows.tobytes())
    return h.hexdigest()


def _load_cache(comp_key: str, chunk_hash: str):
    path = os.path.join(CACHE_DIR, f"{comp_key}_{chunk_hash}.npy")
    if os.path.exists(path):
        try:
            return np.load(path).astype(np.float32)
        except Exception:
            return None
    return None


def _save_cache(comp_key: str, chunk_hash: str, arr: np.ndarray) -> None:
    path = os.path.join(CACHE_DIR, f"{comp_key}_{chunk_hash}.npy")
    try:
        np.save(path, arr.astype(np.float16))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Audio-rate envelope modulation spectrum + slow Chi-Shamma STMF."""

    win = 4096

    # Slow STMF on 30 Hz cochleagram
    N_F = 32
    F_MIN = 80.0
    F_MAX = 8000.0
    RATES_HZ = (0.5, 1.0, 2.0, 4.0)
    SCALES_CPO = (0.5, 1.0, 2.0, 4.0, 8.0)
    SIGMA_RATE_OCT = 0.6
    SIGMA_SCALE_OCT = 0.6

    # Audio-rate envelope modulation spectrum
    # Denser cochlear channels + finer mod bands in the voicing range (60-400 Hz)
    N_F_AM = 40
    F_MIN_AM = 80.0
    F_MAX_AM = 8000.0
    # Log-spaced (geometric) edges from 4 Hz to 1024 Hz, denser in 60-400 Hz
    MOD_BAND_EDGES_HZ = (4.0, 6.0, 9.0, 14.0, 22.0, 35.0, 55.0,
                          80.0, 110.0, 150.0, 200.0, 270.0, 360.0, 500.0,
                          700.0, 1024.0)

    # Slow rates for second-order modulation on AM features
    AM_SLOW_RATES = (0.5, 1.0, 2.0)

    n_params = 0
    COMP_KEY = "stmf_v8_amspec_2nd"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb, self.fcs = gammatone_powerbank(self.win, self.N_F,
                                                  self.F_MIN, self.F_MAX)
        self.fb = self.fb.to(device)
        self.oct_per_ch = float(np.log2(self.fcs[-1] / self.fcs[0]) / (self.N_F - 1))
        self.gamma_H, self.fcs_am = gammatone_mag_freqdomain(
            self.win, self.N_F_AM, self.F_MIN_AM, self.F_MAX_AM)
        self.gamma_H = self.gamma_H.to(device)
        self._stmf_bank_cache: dict[int, tuple[torch.Tensor, list]] = {}

    def _bank(self, T: int):
        if T not in self._stmf_bank_cache:
            self._stmf_bank_cache[T] = _build_stmf_filter_bank(
                T, self.N_F, fs_t=30.0, oct_per_ch=self.oct_per_ch,
                rates=self.RATES_HZ, scales=self.SCALES_CPO,
                sigma_rate_oct=self.SIGMA_RATE_OCT,
                sigma_scale_oct=self.SIGMA_SCALE_OCT,
                device=self.device,
            )
        return self._stmf_bank_cache[T]

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        comp_key = (f"{self.COMP_KEY}_W{self.win}_F{self.N_F}"
                     f"_AM{self.N_F_AM}_M{len(self.MOD_BAND_EDGES_HZ)}")
        chunk_hash = _chunk_hash(windows)
        cached = _load_cache(comp_key, chunk_hash)
        if cached is not None:
            return cached

        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        T = x.shape[0]

        # --- 1) Slow STMF on 30 Hz log-cochleagram ---
        hann = torch.hann_window(self.win, periodic=True, device=self.device)
        power = torch.fft.rfft(x * hann, n=self.win).abs() ** 2
        coch = power @ self.fb                                    # (T, F)
        log_e = torch.log(torch.clamp(coch, min=1e-10))           # (T, F)
        bank, _labels = self._bank(T)                              # (K, T, F)
        resp = stmf_apply(log_e, bank)                             # (K, T, F)
        K = resp.shape[0]
        slow_feat = resp.permute(1, 0, 2).reshape(T, K * self.N_F)

        # --- 2) Audio-rate envelope modulation spectrum ---
        am_feat_full = envelope_modulation_features(
            x, self.gamma_H, self.MOD_BAND_EDGES_HZ, self.win, self.device)  # (T, F_AM, n_modes)
        am_feat = am_feat_full.reshape(T, -1)                       # (T, F_AM * n_modes)
        # --- 3) Second-order modulation: slow Gabor envelopes of channel-pooled AM ---
        # Pool cochlear channels into 4 log-spaced groups to keep dim manageable.
        F_AM = am_feat_full.shape[1]
        n_modes = am_feat_full.shape[2]
        N_GROUPS = 4
        edges = np.linspace(0, F_AM, N_GROUPS + 1).astype(int)
        groups = []
        for i in range(N_GROUPS):
            groups.append(am_feat_full[:, edges[i]:edges[i + 1], :].mean(dim=1))
        am_pooled = torch.stack(groups, dim=1).reshape(T, N_GROUPS * n_modes)
        am_2nd = slow_gabor_envelope(am_pooled, self.AM_SLOW_RATES)  # (T, R*N_GROUPS*n_modes)

        feat = torch.cat([slow_feat, am_feat, am_2nd], dim=1).contiguous()
        out = feat.float().cpu().numpy()
        _save_cache(comp_key, chunk_hash, out)
        return out


model_shorthand_name = "STMF-v8b-amspec-2nd-pooled"
model_description = (
    "v7 + SECOND-ORDER MODULATION on channel-pooled AM (4 log groups of "
    "F_AM channels x 14 mod bands), slow Gabor envelopes rates 0.5/1/2 Hz. "
    "Captures syllabic dynamics OF voicing energy. v7=0.0565, v6=0.0564. "
    "Per-frame: 1728 slow STMF + 560 AM + 3*4*14 AM-2nd = 2456."
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
