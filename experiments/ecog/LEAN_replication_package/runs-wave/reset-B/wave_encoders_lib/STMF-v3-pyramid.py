"""
Hand-designed CORTICAL SPECTRO-TEMPORAL MODULATION (Chi-Ru-Shamma) auditory
front-end: raw waveform -> patient1 ECoG high-gamma.

The whole representation lives in the modulation domain. Pipeline per chunk:

  1) Build a log-cochleagram on the 30 Hz grid (gammatone-power-bank, log
     compression). Each frame is the power spectrum of its centered 4096-sample
     window pooled into N_F ERB-log channels.
  2) Treat the (T, F) cochleagram as a 2-D image of "auditory cortex input".
  3) Apply a dense bank of 2-D Gabor STRF filters tiling
        RATE  (temporal mod, log-spaced ~1-32 Hz, both up- and downward-sweeping)
        SCALE (spectral mod, log-spaced ~0.25-8 cyc/oct)
     plus the boundary cases (pure temporal, pure spectral). Filters are
     constructed in the (omega_t, omega_f) 2-D FFT domain as analytic
     half-plane Gaussians, so a single FFT2 of the cochleagram + one elementwise
     multiply + IFFT2 + magnitude gives a quadrature ("complex") cortical
     response per filter, per (t, channel).
  4) The features per frame are the magnitude responses of every filter at
     every cochlear channel.  Nothing else. No raw cochleagram, no formants,
     no flux, no comodulation — all signal must come from the rate-scale
     representation.

Closed-form: all filters are defined analytically by their (rate, scale, dir)
parameters; no data-fitted weights anywhere except the harness ridge readout.
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
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32)), fcs


# ---------------------------------------------------------------------------
# STMF filterbank — closed-form 2D Gabor in (omega_t, omega_f) FFT domain
# ---------------------------------------------------------------------------

def _build_stmf_filter_bank(T: int, F: int, fs_t: float, oct_per_ch: float,
                              rates: tuple, scales: tuple,
                              sigma_rate_oct: float = 0.5,
                              sigma_scale_oct: float = 0.5,
                              device: str = "cuda") -> tuple[torch.Tensor, list]:
    """Build a (K, T, F) complex filterbank in 2D FFT domain.

    Each filter is an analytic half-plane Gaussian centered at (rate, ±scale, dir).
    We zero the complementary half-plane so that the IFFT2 gives a complex
    (quadrature) response whose magnitude is the cortical envelope.

    `oct_per_ch` = octaves per channel of the cochleagram (= log2(f_max/f_min)/(F-1)).

    Returns (filters (K, T, F) complex, labels list of (rate, scale, dir)).
    """
    # Mod frequency grids in physical units.
    # omega_t: cycles/second (Hz). 2D FFT2 of (T, F): axis 0 ~ time.
    omega_t = np.fft.fftfreq(T, d=1.0 / fs_t).astype(np.float32)         # (T,)
    omega_f_cpc = np.fft.fftfreq(F, d=1.0).astype(np.float32)             # cyc/channel
    omega_f_cpo = omega_f_cpc / oct_per_ch                                 # cyc/oct

    OT, OF = np.meshgrid(omega_t, omega_f_cpo, indexing="ij")              # (T, F)

    filters = []
    labels: list = []

    # Half-plane masks (analytic). step(omega_t > 0) keeps just positive temporal mod;
    # combined with positive/negative spectral mod we get direction selectivity.
    pos_t = (OT > 0).astype(np.float32)
    zero_t = (OT == 0).astype(np.float32)

    def add_gauss(mu_t, mu_f, sig_t_oct, sig_f_oct, mask):
        """Gaussian in (log|omega_t|, log|omega_f|) space, with given half-plane mask."""
        # Avoid log(0). Use log2 distance from center; for omega_t==0, set the
        # log distance to a large value so the gaussian is zero there.
        d_t = np.where(np.abs(OT) > 1e-6,
                        np.log2(np.maximum(np.abs(OT), 1e-6) / max(mu_t, 1e-6)),
                        1e3)
        d_f = np.where(np.abs(OF) > 1e-6,
                        np.log2(np.maximum(np.abs(OF), 1e-6) / max(mu_f, 1e-6)),
                        1e3)
        if mu_t < 1e-6:  # pure spectral
            d_t = np.where(np.abs(OT) < 0.5, 0.0, 1e3)
        if mu_f < 1e-6:  # pure temporal
            d_f = np.where(np.abs(OF) < 0.05, 0.0, 1e3)
        g = np.exp(-0.5 * (d_t / sig_t_oct) ** 2 - 0.5 * (d_f / sig_f_oct) ** 2)
        g = g * mask
        return g.astype(np.float32)

    # Rate x Scale up/down sweeping STRFs
    for r in rates:
        for s in scales:
            # upward (positive scale on positive temporal half-plane)
            mask_up = pos_t * (OF > 0).astype(np.float32)
            g_up = add_gauss(r, s, sigma_rate_oct, sigma_scale_oct, mask_up)
            filters.append(g_up); labels.append((r, s, +1))
            # downward (negative scale on positive temporal half-plane)
            mask_dn = pos_t * (OF < 0).astype(np.float32)
            g_dn = add_gauss(r, s, sigma_rate_oct, sigma_scale_oct, mask_dn)
            filters.append(g_dn); labels.append((r, s, -1))

    # Pure temporal (rate, 0 scale) — both upper and lower spec halves merged
    for r in rates:
        mask_t = pos_t * (np.abs(OF) < 0.1).astype(np.float32)
        g = add_gauss(r, 0.0, sigma_rate_oct, 1.0, mask_t)
        filters.append(g); labels.append((r, 0.0, 0))

    # Pure spectral (0 rate, scale) — upper and lower spec halves
    for s in scales:
        # upper half (positive spectral mod), zero temporal mod
        mask_s_up = (np.abs(OT) < 0.5).astype(np.float32) * (OF > 0).astype(np.float32)
        g_su = add_gauss(0.0, s, 1.0, sigma_scale_oct, mask_s_up)
        filters.append(g_su); labels.append((0.0, s, +1))
        mask_s_dn = (np.abs(OT) < 0.5).astype(np.float32) * (OF < 0).astype(np.float32)
        g_sd = add_gauss(0.0, s, 1.0, sigma_scale_oct, mask_s_dn)
        filters.append(g_sd); labels.append((0.0, s, -1))

    f = np.stack(filters, axis=0).astype(np.complex64)
    return torch.from_numpy(f).to(device), labels


def stmf_apply(log_e: torch.Tensor, filt_bank: torch.Tensor) -> torch.Tensor:
    """Apply (K, T, F) complex STMF bank to (T, F) cochleagram via FFT2.
    Returns magnitude responses (K, T, F) real.

    Pads time axis by reflection to mitigate FFT wrap.
    """
    T, F = log_e.shape
    K = filt_bank.shape[0]
    L = log_e - log_e.mean(dim=0, keepdim=True)
    # 2D FFT
    LF = torch.fft.fft2(L)                              # (T, F) complex
    out_F = filt_bank * LF.unsqueeze(0)                 # (K, T, F)
    out = torch.fft.ifft2(out_F)                        # (K, T, F) complex
    return out.abs().contiguous()                       # quadrature magnitude


# ---------------------------------------------------------------------------
# Caching (per-chunk by content hash + computation-key hash)
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
        pass  # quota / fs error — keep going without persistent cache


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Pure cortical STMF embedder — MULTI-WINDOW PYRAMID.

    Each cochleagram window size offers a different time/freq tradeoff:
      win=1024 (~23 ms): preserves fast temporal modulation, coarse spec res.
      win=2048 (~46 ms): mid tradeoff.
      win=4096 (~93 ms): fine spectral detail, smears fast temporal mod.

    We tune the STMF rate/scale tiling per window — fast rates to short
    windows, fine scales to long windows. Per-frame features are stacked
    magnitudes from all three streams.
    """

    win = 4096   # maximum window supported by aligned segment

    N_F = 32
    F_MIN = 80.0
    F_MAX = 8000.0

    # (window, rates_hz, scales_cpo) — chosen so each window's rates lie
    # within its own ~Nyquist and each contributes a distinct part of the
    # rate-scale plane.
    WIN_CONFIGS = (
        (1024, (4.0, 12.0),       (0.5, 2.0)),
        (2048, (2.0, 8.0),        (1.0, 4.0)),
        (4096, (0.5, 2.0),        (1.0, 4.0, 8.0)),
    )
    SIGMA_RATE_OCT = 0.6
    SIGMA_SCALE_OCT = 0.6

    n_params = 0
    COMP_KEY = "stmf_v3_pyr"

    def __init__(self, device: str = "cuda"):
        self.device = device
        # Build a gammatone bank per window
        self.fbs = []
        self.oct_per_ch_per_win = []
        for (w, _r, _s) in self.WIN_CONFIGS:
            fb, fcs = gammatone_powerbank(w, self.N_F, self.F_MIN, self.F_MAX)
            self.fbs.append(fb.to(device))
            self.oct_per_ch_per_win.append(
                float(np.log2(fcs[-1] / fcs[0]) / (self.N_F - 1)))
        # 2D filter bank caches per (T, win-idx)
        self._stmf_bank_cache: dict[tuple[int, int], tuple[torch.Tensor, list]] = {}

    def _bank(self, T: int, win_idx: int):
        key = (T, win_idx)
        if key not in self._stmf_bank_cache:
            w, rates, scales = self.WIN_CONFIGS[win_idx]
            self._stmf_bank_cache[key] = _build_stmf_filter_bank(
                T, self.N_F, fs_t=30.0,
                oct_per_ch=self.oct_per_ch_per_win[win_idx],
                rates=rates, scales=scales,
                sigma_rate_oct=self.SIGMA_RATE_OCT,
                sigma_scale_oct=self.SIGMA_SCALE_OCT,
                device=self.device,
            )
        return self._stmf_bank_cache[key]

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        comp_key = f"{self.COMP_KEY}_F{self.N_F}_nwin{len(self.WIN_CONFIGS)}"
        chunk_hash = _chunk_hash(windows)
        cached = _load_cache(comp_key, chunk_hash)
        if cached is not None:
            return cached

        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        T = x.shape[0]
        feats = []
        for win_idx, (w, _r, _s) in enumerate(self.WIN_CONFIGS):
            if w == self.win:
                xw = x
            else:
                start = (self.win - w) // 2
                xw = x[:, start:start + w]
            hann = torch.hann_window(w, periodic=True, device=self.device)
            power = torch.fft.rfft(xw * hann, n=w).abs() ** 2          # (T, n_freq)
            coch = power @ self.fbs[win_idx]                            # (T, F)
            log_e = torch.log(torch.clamp(coch, min=1e-10))             # (T, F)
            bank, _labels = self._bank(T, win_idx)                       # (K, T, F)
            resp = stmf_apply(log_e, bank)                               # (K, T, F)
            K = resp.shape[0]
            feats.append(resp.permute(1, 0, 2).reshape(T, K * self.N_F))

        feat = torch.cat(feats, dim=1).contiguous()
        out = feat.float().cpu().numpy()
        _save_cache(comp_key, chunk_hash, out)
        return out


model_shorthand_name = "STMF-v3-pyramid"
model_description = (
    "MULTI-WINDOW STMF pyramid: 3 cochleagrams (win=1024/2048/4096, F=32 ERB), "
    "log-compressed. STMF tiling tuned per window (fast rates -> short win, "
    "fine scales -> long win). All magnitudes stacked per frame. "
    "v1 pure-STMF (log,F=32,W=4096): 0.0537. v2 (cube,F=32): 0.0488."
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
