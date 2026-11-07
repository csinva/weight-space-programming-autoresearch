"""
Hand-designed auditory front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

LEAN-v21 = LEAN-v20 with bugs fixed (no new features):
  - voicing gates are driven by the ACF voicing peak (0..1). v20 used the harmonic-comb
    strength (~6-10, a log-power difference), which saturated every gate at exactly 1.0;
    v20's flatness gates (0.2-0.6 thresholds on a flatness that never exceeds ~0.12 here)
    never opened and are removed;
  - identical filters are removed: the -0.5 cyc/ch copy of every +0.5 cyc/ch 2-D Gabor
    (sweep direction is undefined at the channel Nyquist), the slow bank's raw 2 Hz rate
    (already in the quadrature bank), and the 12 Hz quadrature cos/sin outputs at scales
    0 and 0.5 (numerically equal to 8 Hz, because the Gabor time window is < 1 frame);
  - dead columns are removed: second-order scattering paths below 4 Hz (no support on
    the 10.8 Hz FFT grid of a 4096-sample window) and short-window contrast octaves with
    fewer than 2 FFT bins;
  - interact_set defaults to 'v3e' (the scored configuration); unused code is removed.
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
# /path/to/scratch quota exhausted — fall back to node-local NVMe for the
# per-call feature memmap (transient, single-run; not reused across runs).
CACHE_DIR = os.environ.get("WAVE_FEAT_CACHE", "/tmp/feat_cache_exceed_combine")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Memory-efficient feature extraction: write the per-chunk feature matrix to a
# single big memmap on /path/to/scratch and return a list of *views* into it.
# This keeps host anon-rss to one copy of the feature stack at runtime; the
# downstream np.vstack(ftr).astype(np.float32) then becomes a single in-RAM
# allocation instead of holding (list of arrays) + (vstack output) + (astype
# output) simultaneously. Required to fit LEAN-v107 + v3e + scattering at
# 500/100 inside the 196 GB slurm cgroup limit.
# ---------------------------------------------------------------------------

def _disk_features(stim_dict, resp_dict, chunks, embedder,
                    ndelays: int = 4, lag: int = 3, trim: int = 30, tag: str = "feat"):
    from src import data as _data
    from src.features import make_delayed as _make_delayed
    win = int(getattr(embedder, "win", 2048))

    # Pass 0: compute one chunk to discover P; record per-chunk T_aligned.
    keep_chunks = []
    Ts = []
    for c in chunks:
        T_raw = min(_data.seg_to_T(len(stim_dict[c])), resp_dict[c].shape[0])
        T_aligned = T_raw - lag - trim
        if T_aligned < 2:
            continue
        keep_chunks.append(c)
        Ts.append(T_aligned)
    if not keep_chunks:
        return [], []
    total_T = int(sum(Ts))

    # First chunk to discover P. Stream this chunk into memmap as well.
    c0 = keep_chunks[0]
    seg = stim_dict[c0]; resp = resp_dict[c0]
    T_raw = min(_data.seg_to_T(len(seg)), resp.shape[0])
    windows = _data.frame_windows(seg, win=win, n_frames=T_raw)
    emb = _make_delayed(np.asarray(embedder(windows), dtype=np.float32), ndelays=ndelays)
    emb = emb[:-lag][trim:] if lag > 0 else emb[trim:]
    r0 = resp[lag:T_raw][trim:] if lag > 0 else resp[:T_raw][trim:]
    s = emb.std(0); s[s == 0] = 1.0
    emb = ((emb - emb.mean(0)) / s).astype(np.float32, copy=False)
    s = r0.std(0); s[s == 0] = 1.0
    r0 = ((r0 - r0.mean(0)) / s).astype(np.float32, copy=False)
    P = emb.shape[1]; R = r0.shape[1]

    fp = os.path.join(CACHE_DIR, f"{tag}_F_{total_T}_{P}.bin")
    rp = os.path.join(CACHE_DIR, f"{tag}_R_{total_T}_{R}.bin")
    feat_mm = np.memmap(fp, dtype=np.float32, mode='w+', shape=(total_T, P))
    resp_mm = np.memmap(rp, dtype=np.float32, mode='w+', shape=(total_T, R))
    feat_list, resp_list = [], []
    off = 0
    n = emb.shape[0]
    feat_mm[off:off + n] = emb
    resp_mm[off:off + n] = r0
    feat_list.append(feat_mm[off:off + n])
    resp_list.append(resp_mm[off:off + n])
    off += n
    del emb, r0

    for c in keep_chunks[1:]:
        seg = stim_dict[c]; resp = resp_dict[c]
        T_raw = min(_data.seg_to_T(len(seg)), resp.shape[0])
        windows = _data.frame_windows(seg, win=win, n_frames=T_raw)
        emb = _make_delayed(np.asarray(embedder(windows), dtype=np.float32), ndelays=ndelays)
        if lag > 0:
            emb = emb[:-lag]; r = resp[lag:T_raw]
        else:
            r = resp[:T_raw]
        emb = emb[trim:]; r = r[trim:]
        if emb.shape[0] < 2:
            continue
        s = emb.std(0); s[s == 0] = 1.0
        emb = ((emb - emb.mean(0)) / s).astype(np.float32, copy=False)
        s = r.std(0); s[s == 0] = 1.0
        r = ((r - r.mean(0)) / s).astype(np.float32, copy=False)
        n = emb.shape[0]
        feat_mm[off:off + n] = emb
        resp_mm[off:off + n] = r
        feat_list.append(feat_mm[off:off + n])
        resp_list.append(resp_mm[off:off + n])
        off += n
        del emb, r
    feat_mm.flush(); resp_mm.flush()
    return feat_list, resp_list


def _install_disk_features():
    """Monkey-patch features.get_features_per_chunk to write to disk-backed
    memmaps. The returned per-chunk arrays are views into a single big memmap
    on /path/to/scratch, so they cost ~no host anon-rss until np.vstack reads
    through them."""
    from src import features as _feat
    _orig = _feat.get_features_per_chunk
    _counter = {'n': 0}
    def _patched(stim_dict, resp_dict, chunks, embedder, ndelays=4, lag=3, trim=30):
        _counter['n'] += 1
        return _disk_features(stim_dict, resp_dict, chunks, embedder,
                               ndelays=ndelays, lag=lag, trim=trim,
                               tag=f"call{_counter['n']}")
    _feat.get_features_per_chunk = _patched

    # Patch np.vstack so that consecutive views into the SAME memmap are
    # returned as a single zero-copy view, avoiding the 100GB temporary copy
    # that pushes us over the 192GB cgroup limit at high feature dim.
    # Use a memmap subclass with .astype overridden to also no-op for same dtype.
    class _SmartMemmap(np.memmap):
        def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
            if np.dtype(dtype) == self.dtype:
                return self
            return np.ndarray.astype(self, dtype, order, casting, subok, copy)

    _np_vstack_orig = np.vstack
    def _vstack_smart(tup, *args, **kwargs):
        arrs = list(tup)
        if not arrs:
            return _np_vstack_orig(tup, *args, **kwargs)
        a0 = arrs[0]
        base = getattr(a0, 'base', None)
        if base is None or not isinstance(base, np.memmap):
            return _np_vstack_orig(tup, *args, **kwargs)
        if not all((isinstance(a, np.ndarray) and a.base is base) for a in arrs):
            return _np_vstack_orig(tup, *args, **kwargs)
        if not all(a.shape[1:] == a0.shape[1:] for a in arrs):
            return _np_vstack_orig(tup, *args, **kwargs)
        itemsize = base.dtype.itemsize
        row_bytes = int(np.prod(base.shape[1:])) * itemsize
        base_addr = base.__array_interface__['data'][0]
        first_off_bytes = a0.__array_interface__['data'][0] - base_addr
        if first_off_bytes % row_bytes != 0:
            return _np_vstack_orig(tup, *args, **kwargs)
        first_row = first_off_bytes // row_bytes
        cur = first_row
        for a in arrs:
            off_bytes = a.__array_interface__['data'][0] - base_addr
            if off_bytes % row_bytes != 0 or off_bytes // row_bytes != cur:
                return _np_vstack_orig(tup, *args, **kwargs)
            if a.shape[1:] != base.shape[1:] or not a.flags['C_CONTIGUOUS']:
                return _np_vstack_orig(tup, *args, **kwargs)
            cur += a.shape[0]
        if cur > base.shape[0]:
            return _np_vstack_orig(tup, *args, **kwargs)
        view = base[first_row:cur]
        return view.view(_SmartMemmap)
    np.vstack = _vstack_smart


# ---------------------------------------------------------------------------
# Wavelet scattering helpers (closed-form, FFT-based) — ported from SCAT-v103
# ---------------------------------------------------------------------------

def morlet_bank_freq(n_fft: int, sr: float, fcs: np.ndarray, Q: float) -> np.ndarray:
    """Morlet wavelet bank in the frequency domain (analytic via rfft).
    Returns (n_wavelets, n_freqs) Gaussian centered at each fc with std=fc/Q."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    sigmas = np.maximum(fcs / max(Q, 0.5), 1.0)
    df = freqs[None, :] - fcs[:, None]
    psi = np.exp(-0.5 * (df / sigmas[:, None]) ** 2).astype(np.float32)
    psi[:, 0] = 0.0  # zero DC (Morlet has zero mean)
    return psi


def log_space(f_min: float, f_max: float, Q: float) -> np.ndarray:
    """Q wavelets per octave, log-spaced from f_min to f_max (inclusive)."""
    n_oct = np.log2(f_max / f_min)
    n = int(np.round(n_oct * Q)) + 1
    return (f_min * (2.0 ** (np.arange(n) / Q))).astype(np.float32)


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=50, f_min=50.0, f_max=8000.0, order=4):
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))


def power_spectrum(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2


def leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def acausal_leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    """Time-reversed EMA — exponential 'lookahead' smoothing.
    Provides future-context per channel without re-windowing the audio."""
    return torch.flip(leaky_lowpass(torch.flip(x, dims=[0]), tau_frames), dims=[0])


# ---------------------------------------------------------------------------
# High-level features (closed-form, all derived from power spectrum / cochleagram)
# ---------------------------------------------------------------------------

def harmonic_comb_salience(power: torch.Tensor, win: int, f0_candidates: torch.Tensor,
                            n_harmonics: int = 8) -> torch.Tensor:
    """Harmonic-comb pitch salience: for each F0 candidate, sum log-power at
    its first n harmonic bins (normalized by total log-power).
    power: (T, n_freqs). Returns (T, len(f0_candidates))."""
    n_freqs = power.shape[1]
    freq_res = SR / win  # Hz per bin
    log_p = torch.log(torch.clamp(power, min=1e-10))
    total = log_p.mean(dim=1, keepdim=True)
    out = []
    for f0 in f0_candidates:
        # Bins at k*f0, k=1..n_harmonics
        ks = torch.arange(1, n_harmonics + 1, device=power.device, dtype=torch.float32)
        bins = (f0 * ks / freq_res).long().clamp(max=n_freqs - 1)
        # Pick those bins and sum
        sal = log_p[:, bins].mean(dim=1, keepdim=True) - total
        out.append(sal)
    return torch.cat(out, dim=1)  # (T, F0)


def spectral_moments(power: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Spectral centroid, spread, skewness, kurtosis from raw power spectrum.
    Returns (T, 4)."""
    p = power / (power.sum(dim=1, keepdim=True) + 1e-12)
    mu = (p * freqs).sum(dim=1, keepdim=True)
    df = freqs - mu
    var = (p * df ** 2).sum(dim=1, keepdim=True)
    std = torch.sqrt(var + 1e-12)
    skew = (p * df ** 3).sum(dim=1, keepdim=True) / (std ** 3 + 1e-12)
    kurt = (p * df ** 4).sum(dim=1, keepdim=True) / (var ** 2 + 1e-12)
    return torch.cat([torch.log(mu.clamp(min=1.0)), torch.log(std.clamp(min=1.0)),
                       skew.clamp(-10, 10), kurt.clamp(0, 50)], dim=1)


def spectral_flatness(power: torch.Tensor) -> torch.Tensor:
    """log(geo_mean / arith_mean). Higher = more noise-like, lower = more tonal.
    Returns (T, 1)."""
    log_p = torch.log(torch.clamp(power, min=1e-10))
    geo = log_p.mean(dim=1, keepdim=True)
    arith = torch.log(power.mean(dim=1, keepdim=True) + 1e-10)
    return geo - arith  # log of ratio


def spectral_rolloff(power: torch.Tensor, freqs: torch.Tensor,
                      thresholds=(0.85, 0.95)) -> torch.Tensor:
    """Frequency (log) below which `threshold` fraction of energy lies.
    Returns (T, len(thresholds))."""
    cs = torch.cumsum(power, dim=1)
    total = cs[:, -1:] + 1e-12
    out = []
    for thr in thresholds:
        # First bin where cumulative >= thr * total
        mask = cs >= thr * total
        idx = mask.float().argmax(dim=1)  # (T,)
        out.append(torch.log(freqs[idx].clamp(min=1.0)).unsqueeze(1))
    return torch.cat(out, dim=1)


def hnr_from_acf(power: torch.Tensor, win: int,
                  f_min: float = 75.0, f_max: float = 400.0) -> torch.Tensor:
    """Harmonics-to-noise ratio from normalized autocorrelation peak.
    Returns (T, 2): [peak_value, hnr_db]."""
    full = torch.fft.irfft(power, n=win)
    acf0 = full[:, :1] + 1e-12
    acf_n = full / acf0
    lag_min = max(1, int(SR / f_max))
    lag_max = min(win - 1, int(SR / f_min))
    region = acf_n[:, lag_min:lag_max + 1]
    peak_val, _ = region.max(dim=1, keepdim=True)
    peak_val = peak_val.clamp(min=0.0, max=0.999)
    hnr_db = 10.0 * torch.log10(peak_val.clamp(min=1e-3) /
                                  (1.0 - peak_val).clamp(min=1e-3))
    return torch.cat([peak_val, hnr_db], dim=1)


def phonetic_templates(log_e_long: torch.Tensor, n_filters: int) -> torch.Tensor:
    """Closed-form phonetic band contrasts on the 20-channel long-window
    log-cochleagram (ranges for n_filters=20, ERB-spaced 50-8000 Hz):

      - sibilant:  mean(ch14..19) - mean(ch0..5)       (3.1-8 kHz vs 50-450 Hz)
      - vowel:     mean(ch5..14)  - std over channels  (450 Hz-3.1 kHz, peaky spectrum)
      - nasal:     mean(ch1..5)   - mean(ch8..13)      (100-450 Hz vs 0.9-2.6 kHz)
      - plosive:   std over channels - mean over channels
      - fricative: mean(ch17..19) - mean(ch5..14)      (5.5-8 kHz vs 450 Hz-3.1 kHz)

    Returns (T, 5)."""
    hi = log_e_long[:, max(0, n_filters - 6):].mean(dim=1, keepdim=True)
    lo = log_e_long[:, :max(1, n_filters // 3)].mean(dim=1, keepdim=True)
    mid = log_e_long[:, n_filters // 4:3 * n_filters // 4].mean(dim=1, keepdim=True)
    very_hi = log_e_long[:, max(0, n_filters - 3):].mean(dim=1, keepdim=True)
    narrow_lo = log_e_long[:, 1:6].mean(dim=1, keepdim=True)
    notch = log_e_long[:, 8:14].mean(dim=1, keepdim=True) if n_filters >= 14 else mid
    flatness = log_e_long.std(dim=1, keepdim=True)
    sibilant = hi - lo
    vowel = mid - flatness
    nasal = narrow_lo - notch
    plosive = flatness - (log_e_long.mean(dim=1, keepdim=True))
    fricative = very_hi - mid
    return torch.cat([sibilant, vowel, nasal, plosive, fricative], dim=1)


def cross_band_comodulation(log_e: torch.Tensor) -> torch.Tensor:
    """Products of band-energy pairs (cross-band correlation features).
    Returns (T, 6)."""
    nF = log_e.shape[1]
    # 6 representative bands evenly spaced
    band_idx = np.linspace(0, nF - 1, 7).astype(int)
    bands = []
    for i in range(6):
        bands.append(log_e[:, band_idx[i]:band_idx[i + 1] + 1].mean(dim=1, keepdim=True))
    bands_t = torch.cat(bands, dim=1)  # (T, 6)
    # Products of adjacent pairs + low*high + mid*high (6 pairs)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (2, 5)]
    out = []
    for a, b in pairs:
        out.append((bands_t[:, a] * bands_t[:, b]).unsqueeze(1))
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Best 3-window x 6-tau base + high-level features:
    pitch salience (harmonic comb), spectral moments, flatness, rolloff,
    HNR, phonetic templates, cross-band comodulation.
    """

    win = 4096
    n_params = 0
    WIN_LIST = (1024, 2048, 4096)
    TAUS = (1.5, 4.5, 15.0, 60.0, 150.0)
    F0_CANDIDATES = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0, 230.0, 270.0, 320.0, 380.0, 450.0)
    HL_TAUS = (15.0, 60.0, 150.0)  # multi-tau for high-level features
    # (temporal Hz, spectral cyc/channel) Chi-Shamma rate-scale tuning curves
    STMF_RATES = (
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),
        (12.0, 0.25), (12.0, -0.25),
        (4.0, 0.0),  (8.0, 0.0), (12.0, 0.0),
        (0.0, 0.25), (0.0, 0.5),
    )
    # +0.5 cyc/ch only: at the channel Nyquist the +0.5 and -0.5 kernels are identical.
    QUAD_RATES = (2.0, 4.0, 8.0, 12.0)
    QUAD_SCALES = (0.0, 0.25, -0.25, 0.5)
    SLOW_RATES = (0.5, 1.0, 1.5, 2.0)  # raw 2 Hz duplicates QUAD; only its slow dyn is kept
    SCAT_FC2_MIN = 4.0  # 2nd-order scattering wavelets below this have no FFT-grid support

    def __init__(self, device: str = "cuda", n_filters: int = 20):
        self.device = device
        self.n_filters = n_filters
        self.fbs = [gammatone_powerbank(w, n_filters=n_filters).to(device)
                    for w in self.WIN_LIST]
        # Finer cochleagram (40 channels) on the LONG window for STMF only
        self.fb_fine = gammatone_powerbank(4096, n_filters=40).to(device)
        self.channel_idx = torch.arange(n_filters, dtype=torch.float32, device=device).view(1, -1)
        self.f0_cands = torch.tensor(self.F0_CANDIDATES, dtype=torch.float32, device=device)
        self.long_freqs = torch.from_numpy(
            np.fft.rfftfreq(4096, d=1.0 / SR).astype(np.float32)).to(device)
        self._gabor_cache = {}
        self._stmf_cache = {}
        # ---------- Wavelet scattering bank (closed-form, FFT-based) ----------
        self.scat_n_fft = 4096
        Q1 = 4.0
        Q2 = 1.0
        # First-order: Q1=4/oct, 50 Hz .. 12.8 kHz
        self.scat_fcs1 = log_space(50.0, 12800.0, Q1)
        psi1_np = morlet_bank_freq(self.scat_n_fft, SR, self.scat_fcs1, Q1)
        self.scat_psi1 = torch.from_numpy(psi1_np).to(device)
        # Sub-set of first-order bands for S2 paths (every 4 → 9 paths)
        self.scat_j1_sub = np.arange(0, len(self.scat_fcs1), 4)
        # Second-order modulation wavelets: 4..4096 Hz at Q2=1/oct. v20 started at 0.5 Hz,
        # but the 0.5/1/2 Hz wavelets put <1e-4 weight on any non-DC bin of the 10.8 Hz grid.
        self.scat_fcs2 = log_space(self.SCAT_FC2_MIN, 4096.0, Q2)
        psi2_np = morlet_bank_freq(self.scat_n_fft, SR, self.scat_fcs2, Q2)
        self.scat_psi2 = torch.from_numpy(psi2_np).to(device)
        self.scat_n_bins = 1
        self._scat_logged = False

    def _stmf_quadrature(self, log_e: torch.Tensor, omega_t: float, omega_f: float,
                           fs: float = 30.0):
        """Return separate cos and sin convolutional responses (T, F) of the
        Chi-Shamma 2-D Gabor at (omega_t, omega_f). Shares the _stmf_cache kernels."""
        key = (round(omega_t, 3), round(omega_f, 3))
        if key not in self._stmf_cache:
            # Build cache same as _stmf_envelope
            period_t = fs / max(abs(omega_t), 0.5)
            sigma_t = 1.0 * period_t / (2 * np.pi)
            Lt = max(1, int(np.ceil(3 * sigma_t)))
            sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi) * 1.0
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
        xp = log_e.unsqueeze(0).unsqueeze(0)
        c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
        s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
        T, F = log_e.shape
        return c.squeeze(0).squeeze(0)[:T, :F], s.squeeze(0).squeeze(0)[:T, :F]

    def _stmf_envelope(self, log_e: torch.Tensor, omega_t: float, omega_f: float,
                        fs: float = 30.0) -> torch.Tensor:
        """Small Chi-Shamma 2-D Gabor envelope on (T, F) log-cochleagram.
        Cached kernel for speed; conv2d (small kernel)."""
        key = (round(omega_t, 3), round(omega_f, 3))
        if key not in self._stmf_cache:
            period_t = fs / max(abs(omega_t), 0.5)
            sigma_t = 1.0 * period_t / (2 * np.pi)
            Lt = max(1, int(np.ceil(3 * sigma_t)))
            sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi) * 1.0
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
        xp = log_e.unsqueeze(0).unsqueeze(0)
        c = torch.nn.functional.conv2d(xp, cos_k, padding=(Lt, Lf))
        s = torch.nn.functional.conv2d(xp, sin_k, padding=(Lt, Lf))
        T, F = log_e.shape
        env = torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)
        return env[:T, :F]

    def _gabor_envelope(self, x: torch.Tensor, rate_hz: float,
                         fs: float = 30.0) -> torch.Tensor:
        """Gabor band-pass envelope on a (T, C) signal along time."""
        if rate_hz not in self._gabor_cache:
            period = fs / rate_hz
            sigma = 1.0 * period
            L = int(np.ceil(4 * sigma))
            t = np.arange(-L, L + 1)
            g = np.exp(-(t ** 2) / (2 * sigma ** 2))
            cos_k = g * np.cos(2 * np.pi * rate_hz * t / fs)
            sin_k = g * np.sin(2 * np.pi * rate_hz * t / fs)
            cos_k = cos_k - cos_k.mean()
            sin_k = sin_k - sin_k.mean()
            cos_k = cos_k / (np.linalg.norm(cos_k) + 1e-12)
            sin_k = sin_k / (np.linalg.norm(sin_k) + 1e-12)
            self._gabor_cache[rate_hz] = (
                torch.from_numpy(cos_k.astype(np.float32)).to(self.device),
                torch.from_numpy(sin_k.astype(np.float32)).to(self.device),
            )
        cos_k, sin_k = self._gabor_cache[rate_hz]
        L = cos_k.numel()
        pad = L // 2
        C = x.shape[1]
        xp = x.transpose(0, 1).unsqueeze(0)
        ck = cos_k.view(1, 1, -1).expand(C, 1, L).contiguous()
        sk = sin_k.view(1, 1, -1).expand(C, 1, L).contiguous()
        xp_p = torch.nn.functional.pad(xp, (pad, pad), mode='replicate')
        c = torch.nn.functional.conv1d(xp_p, ck, groups=C)
        s = torch.nn.functional.conv1d(xp_p, sk, groups=C)
        c = c[..., :x.shape[0]]
        s = s[..., :x.shape[0]]
        env = torch.sqrt(c * c + s * s + 1e-12)
        return env.squeeze(0).transpose(0, 1).contiguous()

    # Stage-2 interaction set: 'v3e' (scored configuration) or 'none' (omit the 405 dims).
    interact_set: str = 'v3e'

    def _quad_bank(self, log_e: torch.Tensor, rates, scales, bin_edges, drop=()):
        """Quadrature 2-D Gabor bank averaged within frequency bins.

        Returns (raw, mags). raw is (T, n) with columns ordered rate -> scale -> bin ->
        (cos, sin, mag), omitting (rate, scale, phase) triples listed in `drop`; mags is
        (T, n_rates, n_scales, n_bins) with every magnitude."""
        cols, mags = [], []
        for wt in rates:
            m_r = []
            for wf in scales:
                c, s = self._stmf_quadrature(log_e, wt, wf)
                m_s = []
                for (lo, hi) in bin_edges:
                    cb = c[:, lo:hi].mean(dim=1, keepdim=True)
                    sb = s[:, lo:hi].mean(dim=1, keepdim=True)
                    mb = torch.sqrt(c[:, lo:hi] ** 2 + s[:, lo:hi] ** 2 + 1e-12).mean(dim=1, keepdim=True)
                    for ph, v in (('cos', cb), ('sin', sb), ('mag', mb)):
                        if (wt, wf, ph) not in drop:
                            cols.append(v)
                    m_s.append(mb)
                m_r.append(torch.cat(m_s, dim=1))
            mags.append(torch.stack(m_r, dim=1))
        return torch.cat(cols, dim=1), torch.stack(mags, dim=1)

    @torch.no_grad()
    def _wavelet_scattering(self, sig: torch.Tensor) -> torch.Tensor:
        """Closed-form 2-order Morlet wavelet scattering on (T, win) audio windows.
        Returns log-compressed (T, K) feature block per 30 Hz frame.

        - First-order |x * psi_j1| envelopes, sub-window mean + STD log → S1 stats.
        - Second-order |  |x*psi_j1| * psi_j2  | envelopes, sub-window mean + STD.
        - Joint TF first-difference of S1 across j1 (formant/sweep direction).
        """
        T, N = sig.shape
        NB = self.scat_n_bins
        bin_w = N // NB
        X = torch.fft.rfft(sig, n=N, dim=1)  # (T, n_freqs)
        psi1 = self.scat_psi1                # (n1, n_freqs)
        psi2 = self.scat_psi2                # (n2, n_freqs)
        n1, n2 = psi1.shape[0], psi2.shape[0]

        # First-order envelopes for sub-set bands only (used downstream for S2)
        sub_set = sorted(int(j) for j in self.scat_j1_sub)
        sub_envs = {}  # j1 -> (T, N)
        sub_if_mean = {}  # j1 -> (T, 1) frame-mean instantaneous frequency
        sub_coh = {}  # j1 -> (T, 1) phase coherence (tonal vs noise)
        sub_if_slope = {}  # j1 -> (T, 1) pitch jitter (std of per-sample IF in Hz)
        chunk = 8
        for s in range(0, len(sub_set), chunk):
            js = sub_set[s:s + chunk]
            psi_blk = psi1[js]              # (k, n_freqs)
            Xpsi = X.unsqueeze(1) * psi_blk.unsqueeze(0)  # (T,k,n_freqs) complex
            yr = torch.fft.irfft(Xpsi, n=N, dim=2)        # real bandpass
            yi = torch.fft.irfft(-1j * Xpsi, n=N, dim=2)  # Hilbert
            u = torch.sqrt(yr * yr + yi * yi + 1e-20)     # analytic envelope (T,k,N)
            # Instantaneous frequency: arg(<y[t+1]*conj(y[t])>) / (2π*dt) per frame
            yc_re = yr[..., 1:] * yr[..., :-1] + yi[..., 1:] * yi[..., :-1]
            yc_im = yi[..., 1:] * yr[..., :-1] - yr[..., 1:] * yi[..., :-1]
            yc_re_m = yc_re.mean(dim=2)                    # (T,k)
            yc_im_m = yc_im.mean(dim=2)
            yc_mag = torch.sqrt(yc_re_m * yc_re_m + yc_im_m * yc_im_m + 1e-20)
            u2_mean = (u * u).mean(dim=2)                   # (T,k)
            coherence = yc_mag / u2_mean.clamp(min=1e-12)   # (T,k) in [0,1]
            if_norm = torch.atan2(yc_im_m, yc_re_m) * (SR / (2 * np.pi))  # (T,k) Hz
            if_per_sample = torch.atan2(yc_im, yc_re) * (SR / (2 * np.pi))
            if_jitter_local = if_per_sample.std(dim=2)
            for k_local, j1 in enumerate(js):
                sub_envs[j1] = u[:, k_local, :].contiguous()
                sub_if_mean[j1] = if_norm[:, k_local:k_local + 1]
                sub_coh[j1] = coherence[:, k_local:k_local + 1]
                sub_if_slope[j1] = if_jitter_local[:, k_local:k_local + 1]
            del Xpsi, yr, yi, u, yc_re, yc_im, yc_re_m, yc_im_m, yc_mag, u2_mean, coherence, if_norm, if_per_sample, if_jitter_local

        # S1 stats: per-band, per-sub-window MEAN over selected j1 bands → log
        s1_means = []
        for j1 in sub_set:
            e = sub_envs[j1].view(T, NB, bin_w).mean(dim=2)  # (T, NB)
            s1_means.append(e)
        S1m = torch.stack(s1_means, dim=1)  # (T, K, NB)
        S1m_log = torch.log(S1m.clamp(min=1e-8))

        # Joint-TF: first-difference along j1 axis (sub-set sorted by fc1), NORMALIZED by mean S1
        joint = (S1m[:, 1:, :] - S1m[:, :-1, :]).abs()         # (T, K-1, NB)
        s1_local = 0.5 * (S1m[:, 1:, :] + S1m[:, :-1, :]).clamp(min=1e-6)
        joint_log = torch.log1p(joint / s1_local)             # normalized log-modulation

        # Second-order: |.| * psi_j2 only for valid pairs fc2 < fc1/2.
        s2_means = []
        s2_stds = []
        s2_peak_per_j1_list = []
        fcs1 = self.scat_fcs1
        fcs2 = self.scat_fcs2
        s1_per_j1 = {}
        for j1 in sub_set:
            e = sub_envs[j1]
            s1_per_j1[j1] = e.view(T, NB, bin_w).mean(dim=2)
        for j1 in sub_set:
            keep = [j2 for j2 in range(n2) if fcs2[j2] < fcs1[j1] / 2.0]
            if not keep:
                continue
            e = sub_envs[j1]
            E = torch.fft.rfft(e, n=N, dim=1)
            psi2_blk = psi2[keep]
            Epsi = E.unsqueeze(1) * psi2_blk.unsqueeze(0)
            yr = torch.fft.irfft(Epsi, n=N, dim=2)
            yi = torch.fft.irfft(-1j * Epsi, n=N, dim=2)
            u2 = torch.sqrt(yr * yr + yi * yi + 1e-20)
            u2_b = u2.view(T, len(keep), NB, bin_w)
            m = u2_b.mean(dim=3)
            sd = u2_b.std(dim=3)
            pk = u2_b.amax(dim=3)
            s1_norm = s1_per_j1[j1].unsqueeze(1).clamp(min=1e-6)
            m_norm = m / s1_norm
            s2_means.append(m_norm.reshape(T, len(keep) * NB))
            s2_stds.append((sd / s1_norm).reshape(T, len(keep) * NB))
            pk_per_j1 = pk.amax(dim=1)
            s2_peak_per_j1_list.append(pk_per_j1)
            del E, Epsi, yr, yi, u2, u2_b, pk
        S2m = torch.cat(s2_means, dim=1) if s2_means else torch.zeros(T, 0, device=sig.device)
        S2s = torch.cat(s2_stds, dim=1) if s2_stds else torch.zeros(T, 0, device=sig.device)
        S2peak = torch.cat(s2_peak_per_j1_list, dim=1) if s2_peak_per_j1_list else torch.zeros(T, 0, device=sig.device)
        S2peak_log = torch.pow(S2peak.clamp(min=0.0) + 1e-8, 1.0 / 3.0)
        S2m_log = torch.log1p(S2m.clamp(min=0.0))
        S2s_log = torch.log1p(S2s.clamp(min=0.0))

        # Intra-frame contour: log(env_last_quarter / env_first_quarter) per j1
        contour_blocks = []
        for j1 in sub_set:
            e = sub_envs[j1]
            q = N // 4
            ef = e[:, :q].mean(dim=1, keepdim=True).clamp(min=1e-8)
            el = e[:, -q:].mean(dim=1, keepdim=True).clamp(min=1e-8)
            contour_blocks.append(torch.log(el / ef))
        contour = torch.cat(contour_blocks, dim=1)
        # S1 sub-set log-frame-mean and log-frame-PEAK per j1 path
        s1_sub_log = torch.log(S1m.squeeze(2).clamp(min=1e-8))
        s1_peak_blocks = []
        for j1 in sub_set:
            s1_peak_blocks.append(sub_envs[j1].amax(dim=1, keepdim=True).clamp(min=1e-8))
        s1_peak_log = torch.log(torch.cat(s1_peak_blocks, dim=1))
        # Per-j1 instantaneous-frequency frame-mean (Hz). Log-scaled relative to fc.
        if_blocks = []
        coh_blocks = []
        for j1 in sub_set:
            fc = float(self.scat_fcs1[j1])
            ratio = sub_if_mean[j1] / fc
            if_blocks.append(ratio)
            coh_blocks.append(sub_coh[j1])
        if_feat = torch.cat(if_blocks, dim=1).clamp(-3.0, 3.0)
        coh_feat = torch.cat(coh_blocks, dim=1).clamp(0.0, 0.9999)
        coh_feat = torch.atanh(coh_feat)
        # Per-j1 pitch jitter (Hz), normalized by fc
        slope_blocks = []
        for j1 in sub_set:
            fc = float(self.scat_fcs1[j1])
            slope_blocks.append(sub_if_slope[j1] / fc)
        if_jitter = torch.cat(slope_blocks, dim=1).clamp(0.0, 3.0)
        # Per-j1 S2 distribution entropy
        s2_entropy_blocks = []
        s2_off = 0
        for j1 in sub_set:
            keep = [j2 for j2 in range(self.scat_psi2.shape[0]) if self.scat_fcs2[j2] < self.scat_fcs1[j1] / 2.0]
            if not keep:
                s2_entropy_blocks.append(torch.zeros(T, 1, device=sig.device))
                continue
            slice_size = len(keep) * NB
            s2_block = S2m[:, s2_off:s2_off + slice_size]
            p = s2_block / (s2_block.sum(dim=1, keepdim=True).clamp(min=1e-8))
            ent = -(p * torch.log(p.clamp(min=1e-10))).sum(dim=1, keepdim=True)
            s2_entropy_blocks.append(ent)
            s2_off += slice_size
        s2_entropy = torch.cat(s2_entropy_blocks, dim=1)
        # COMBINE-v5: COMBINE-v4 + restore S2s_log (modulation std per j2 path)
        # and add multi-tau dynamics on s1_sub_log (per-band envelope) at the
        # LEAN HL_TAUS. Disk-backed memmap removes the host RAM bottleneck.
        s1_sub_dyn = []
        for tau in self.HL_TAUS:  # (15, 60, 150) — phonetic/syllabic/word
            s1_sub_dyn.append(s1_sub_log - leaky_lowpass(s1_sub_log, tau_frames=tau))
        s1_sub_dyn = torch.cat(s1_sub_dyn, dim=1) if s1_sub_dyn else s1_sub_log[:, :0]
        out = torch.cat([
            S2m_log,        # 2nd-order modulation magnitudes, NORMALIZED by S1
            S2s_log,        # NEW (v5): normalized S2 std per j2 path
            S2peak_log,     # log S2 peak per j1 (max over j2)
            joint_log.reshape(T, -1),  # joint-TF |Δj1 envelope|, normalized
            contour,        # intra-frame onset/offset asymmetry per j1
            s1_sub_log,     # frame-mean log envelope per j1 path
            s1_sub_dyn,     # NEW (v5): multi-tau adapt-delta of s1_sub_log
            s1_peak_log,    # frame-PEAK log envelope per j1 path
            if_feat,        # instantaneous frequency / fc per j1
            coh_feat,       # Fisher-z phase coherence per j1
            if_jitter,      # per-j1 IF jitter (per-sample std / fc)
            s2_entropy,     # per-j1 entropy of S2 path distribution
        ], dim=1)
        if not self._scat_logged:
            print(f"  scattering: K={len(sub_set)} NB={NB} dim={out.shape[1]} "
                  f"(S1m {S1m_log.shape[1]}, joint {joint_log.shape[1]}, "
                  f"S2m {S2m_log.shape[1]}, S2s {S2s_log.shape[1]})")
            self._scat_logged = True
        return out

    def _concat_groups(self, groups: list) -> torch.Tensor:
        """Concatenate the named (name, tensor[T, d]) feature groups in order."""
        return torch.cat([t for _, t in groups], dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        # Named-group accumulator (Stage 3). Each entry: (group_name, tensor[T,d])
        groups: list[tuple[str, torch.Tensor]] = []
        def add(name: str, t: torch.Tensor):
            groups.append((name, t))
        log_es = []
        powers = []
        cochlea_blocks = []
        adapt_blocks = []
        for w, fb in zip(self.WIN_LIST, self.fbs):
            if w == 4096:
                xw = x
            else:
                start = (4096 - w) // 2
                xw = x[:, start:start + w]
            p = power_spectrum(xw, w, self.device)
            powers.append(p)
            log_e = torch.log(torch.clamp(p @ fb, min=1e-10))
            log_es.append(log_e)
            cochlea_blocks.append(log_e)
            for tau in self.TAUS:
                adapt_blocks.append(log_e - leaky_lowpass(log_e, tau_frames=tau))
        add('cochleagram', torch.cat(cochlea_blocks, dim=1))
        add('adapt_delta', torch.cat(adapt_blocks, dim=1))

        # [LEAN-v10] removed: sub-window AM stats (am_depth, am_range, am_onset, 60 dim)

        # [LEAN-v11] removed: Multi-tau loudness + centroid per window (36 dim)

        # ============== HIGH-LEVEL FEATURES from long-window power spectrum ==============
        power_long = powers[-1]  # (T, n_freqs)
        log_e_long = log_es[-1]   # (T, F)

        # 1. Harmonic-comb pitch salience: (T, 11)
        ps = harmonic_comb_salience(power_long, 4096, self.f0_cands, n_harmonics=8)
        add('pitch_salience', ps)
        # Best pitch and its strength (max over candidates) + multi-tau dynamic
        best_strength, best_idx = ps.max(dim=1, keepdim=True)
        best_f0 = self.f0_cands[best_idx.squeeze(1)].unsqueeze(1)
        log_f0 = torch.log(best_f0.clamp(min=20.0))
        add('pitch_summary', torch.cat([best_strength, log_f0], dim=1))
        pitch_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            pitch_dyn_blocks.append(log_f0 - leaky_lowpass(log_f0, tau_frames=tau))
            pitch_dyn_blocks.append(best_strength - leaky_lowpass(best_strength, tau_frames=tau))
        add('pitch_dyn', torch.cat(pitch_dyn_blocks, dim=1))

        # 2. Spectral moments (4) on long-window power spectrum
        mom = spectral_moments(power_long, self.long_freqs)
        add('spectral_moments', mom)

        # 3. Spectral flatness (1)
        flat = spectral_flatness(power_long)
        add('spectral_flatness', flat)

        # 4. Spectral rolloff (2): 85%, 95%
        roll = spectral_rolloff(power_long, self.long_freqs)
        add('spectral_rolloff', roll)

        # 5. HNR (2): autocorr peak + HNR_db
        hnr = hnr_from_acf(power_long, 4096)
        add('hnr', hnr)

        # 6. Phonetic templates (5)
        phon = phonetic_templates(log_e_long, self.n_filters)
        add('phonetic', phon)

        # 7. Cross-band comodulation (6) on long log_e — needed by mod_signals (not added as group)
        comod = cross_band_comodulation(log_e_long)

        # 8. Spectral flux (broadband pos/neg/abs per window). PROVEN +0.002.
        flux_feats = []
        for log_e_w in log_es:
            d = torch.zeros_like(log_e_w)
            d[1:] = log_e_w[1:] - log_e_w[:-1]
            flux_pos = d.clamp(min=0.0).sum(dim=1, keepdim=True)
            flux_neg = (-d).clamp(min=0.0).sum(dim=1, keepdim=True)
            flux_abs = d.abs().sum(dim=1, keepdim=True)
            flux_feats.append(torch.cat([flux_pos, flux_neg, flux_abs], dim=1))
        flux_all = torch.cat(flux_feats, dim=1)
        flux_dyn = [flux_all]
        for tau in self.HL_TAUS:
            flux_dyn.append(flux_all - leaky_lowpass(flux_all, tau_frames=tau))
        add('flux', torch.cat(flux_dyn, dim=1))

        # 9. MODULATION FILTERBANK on key 1-D signals at prosodic/syllabic rates.
        long_loud = log_es[-1].mean(dim=1, keepdim=True)
        e_w = torch.softmax(log_es[-1], dim=1)
        long_centroid = (e_w * self.channel_idx).sum(dim=1, keepdim=True)
        mod_signals = [long_loud, long_centroid, hnr, best_strength, log_f0,
                        flux_all, phon, mom, flat, roll, comod]
        mod_blocks = []
        for rate in (2.0, 4.0, 6.0, 10.0):
            for sig in mod_signals:
                mod_blocks.append(self._gabor_envelope(sig, rate))
        add('modulation', torch.cat(mod_blocks, dim=1))

        # 10. Chi-Shamma 2-D STMF on the long log-cochleagram (T, F=20).
        stmf_means = []
        stmf_blocks = []
        for (omega_t, omega_f) in self.STMF_RATES:
            stmf_e = self._stmf_envelope(log_e_long, omega_t, omega_f)
            stmf_blocks.append(stmf_e)
            stmf_means.append(stmf_e.mean(dim=1, keepdim=True))
        add('stmf', torch.cat(stmf_blocks, dim=1))
        stmf_mean_all = torch.cat(stmf_means, dim=1)
        add('stmf_mean_gabor', self._gabor_envelope(stmf_mean_all, 2.0))

        # 10b. FORMANT-LIKE peak tracking (2 formants).
        nF = self.n_filters
        f1_band = log_e_long[:, :nF // 2]
        f2_band = log_e_long[:, nF // 2:]
        w1 = torch.softmax(f1_band * 2.0, dim=1)
        w2 = torch.softmax(f2_band * 2.0, dim=1)
        ch1 = torch.arange(nF // 2, device=log_e_long.device,
                            dtype=log_e_long.dtype).view(1, -1)
        ch2 = torch.arange(nF // 2, device=log_e_long.device,
                            dtype=log_e_long.dtype).view(1, -1) + nF // 2
        f1_loc = (w1 * ch1).sum(dim=1, keepdim=True)
        f2_loc = (w2 * ch2).sum(dim=1, keepdim=True)
        f1_amp = (w1 * f1_band).sum(dim=1, keepdim=True)
        f2_amp = (w2 * f2_band).sum(dim=1, keepdim=True)
        f12_diff = f2_loc - f1_loc
        formants = torch.cat([f1_loc, f2_loc, f12_diff, f1_amp, f2_amp,
                                f2_amp - f1_amp], dim=1)
        formants_blocks = [formants]
        for tau in (4.5, 15.0, 60.0):
            formants_blocks.append(formants - leaky_lowpass(formants, tau_frames=tau))
        add('formants', torch.cat(formants_blocks, dim=1))
        fmt_loc = torch.cat([f1_loc, f2_loc, f12_diff], dim=1)
        fmt_d = torch.zeros_like(fmt_loc)
        fmt_d[1:] = fmt_loc[1:] - fmt_loc[:-1]
        add('formant_dyn', torch.cat([fmt_d, fmt_d.clamp(min=0),
                                       (-fmt_d).clamp(min=0)], dim=1))

        cube_long_var = torch.pow(power_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)

        # 10b4. PER-CHANNEL "modulation depth": HWR of high-pass cochleagram,
        # low-passed to get the slow envelope of fast modulation per band.
        mdepth_blocks = []
        for hp_tau, lp_tau in ((4.5, 15.0), (1.5, 15.0)):
            hp = log_e_long - leaky_lowpass(log_e_long, tau_frames=hp_tau)
            mdepth_blocks.append(leaky_lowpass(hp.abs(), tau_frames=lp_tau))
        add('mod_depth', torch.cat(mdepth_blocks, dim=1))

        # 10b5. SLOW TONIC smoothed cochleagrams (state/context at prosodic scale).
        tonic_blocks = []
        for i, (w, fb) in enumerate(zip(self.WIN_LIST, self.fbs)):
            le_w = log_es[i]
            cube_w = torch.pow(powers[i] @ fb + 1e-10, 1.0 / 3.0)
            for tau in (60.0, 150.0):
                tonic_blocks.append(leaky_lowpass(le_w, tau_frames=tau))
                tonic_blocks.append(leaky_lowpass(cube_w, tau_frames=tau))
            tonic_blocks.append(acausal_leaky_lowpass(le_w, tau_frames=60.0))
        add('tonic', torch.cat(tonic_blocks, dim=1))

        # 10b6. FUTURE + PAST DELTA per band: log_e[t±k] - log_e[t].
        fp_blocks = []
        for k in (3, 6, 12):
            fut_k = torch.cat([log_e_long[k:], log_e_long[-1:].expand(k, -1)], dim=0)
            future_delta = fut_k - log_e_long
            fp_blocks.append(future_delta)
            fp_blocks.append(future_delta.clamp(min=0.0))
            fp_blocks.append((-future_delta).clamp(min=0.0))
            past_k = torch.cat([log_e_long[:1].expand(k, -1), log_e_long[:-k]], dim=0)
            past_delta = log_e_long - past_k
            fp_blocks.append(past_delta)
        fut_c = torch.cat([cube_long_var[6:], cube_long_var[-1:].expand(6, -1)], dim=0)
        fp_blocks.append(fut_c - cube_long_var)
        add('future_past', torch.cat(fp_blocks, dim=1))

        # 11. F0 dynamics (slope, curvature, on/off) + HNR onset
        f0_d1 = torch.zeros_like(log_f0)
        f0_d1[1:] = log_f0[1:] - log_f0[:-1]
        f0_d2 = torch.zeros_like(log_f0)
        f0_d2[1:] = f0_d1[1:] - f0_d1[:-1]
        add('f0_dyn', torch.cat([f0_d1, f0_d2,
                                   f0_d1.clamp(min=0), (-f0_d1).clamp(min=0)], dim=1))
        hnr_db = hnr[:, 1:2]
        hnr_d = torch.zeros_like(hnr_db)
        hnr_d[1:] = hnr_db[1:] - hnr_db[:-1]
        add('hnr_dyn', torch.cat([hnr_d, hnr_d.clamp(min=0), (-hnr_d).clamp(min=0)], dim=1))

        # ======================================================================
        # STAGE 2 — INTERPRETABLE NONLINEAR INTERACTIONS
        # Hand-picked named products / ratios between existing features. Each
        # term has a clear auditory meaning a linear ridge cannot synthesize.
        # Added when self.interact_set == 'v3e' (the default).
        # ======================================================================
        if self.interact_set != 'none':
            hnr_peak = hnr[:, 0:1]                     # voicing confidence (0..1)
            hnr_dvox = hnr[:, 1:2]                     # voicing in dB
            long_loud_1d = long_loud                   # broadband loudness
            flux_long = flux_feats[-1]                 # (T, 3): pos / neg / abs at long-win
            flux_abs_long = flux_long[:, 2:3]
            flux_pos_long = flux_long[:, 0:1]
            f12_disp = (f2_loc - f1_loc)               # formant dispersion (in band-index units)
            f12_amp = (f1_amp + f2_amp)                # joint formant amplitude

            inter = []
            # 'v3e' (LEAN-v17): asymmetric (ReLU+, ReLU-, cube) transforms of 10 key signals,
            # of their deviations at 3 time scales, and of phonetic/formant deviations; 6
            # cross-lag products; phonetic x key-signal, phonetic x formant, key-signal x
            # formant and onset/offset-gated phonetic products. 405 dims.
            if self.interact_set == 'v3e':
                key_sigs10 = [long_loud_1d, best_strength, log_f0, flux_abs_long,
                              hnr_peak, long_centroid, flat, hnr_dvox,
                              f12_disp, f12_amp]
                for sig in key_sigs10:
                    m = sig.mean(dim=0, keepdim=True)
                    s = sig.std(dim=0, keepdim=True).clamp(min=1e-6)
                    z = (sig - m) / s
                    inter.append(z.clamp(min=0.0))
                    inter.append((-z).clamp(min=0.0))
                    inter.append(z * z * z)
                for sig in key_sigs10:
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for j in range(phon.shape[1]):
                    sig = phon[:, j:j+1]
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                for sig in (f1_loc, f2_loc, f1_amp, f2_amp):
                    for tau in (4.5, 15.0, 60.0):
                        dev = sig - leaky_lowpass(sig, tau_frames=tau)
                        m = dev.mean(dim=0, keepdim=True)
                        s = dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (dev - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                def _lag(sig, k):
                    return torch.cat([sig[:k], sig[:-k]], dim=0)
                inter.append(_lag(flux_abs_long, 2) * hnr_d)
                inter.append(_lag(hnr_d, 2) * flux_abs_long)
                inter.append(_lag(flux_abs_long, 2) * long_loud_1d)
                fwd_f0d1 = torch.cat([f0_d1[2:], f0_d1[-1:].expand(2, -1)], dim=0)
                inter.append(best_strength * fwd_f0d1)
                fwd_flux = torch.cat([flux_abs_long[2:], flux_abs_long[-1:].expand(2, -1)], dim=0)
                inter.append(long_loud_1d * fwd_flux)
                fwd_unv = torch.cat([(1.0 - hnr_peak)[2:],
                                     (1.0 - hnr_peak)[-1:].expand(2, -1)], dim=0)
                inter.append(hnr_peak * fwd_unv)
                # v2xx: PHON × KEY-SIG curated products × asym
                phon_factors = [phon[:, j:j+1] for j in range(phon.shape[1])]
                sig_factors  = [best_strength, long_loud_1d, flux_abs_long, hnr_d]
                for pf in phon_factors:
                    for sf in sig_factors:
                        p = pf * sf
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v2zz: PHON × FORMANT curated products × asym
                formant_factors = [f1_loc, f2_loc, f1_amp, f2_amp]
                for pf in phon_factors:
                    for ff in formant_factors:
                        p = pf * ff
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v3a: KEY-SIG × FORMANT curated products × asym
                for sf in sig_factors:
                    for ff in formant_factors:
                        p = sf * ff
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)
                # v3d: ONSET / OFFSET gated phon products
                flux_dev = flux_abs_long - leaky_lowpass(flux_abs_long, tau_frames=15.0)
                fm = flux_dev.mean(dim=0, keepdim=True)
                fs = flux_dev.std(dim=0, keepdim=True).clamp(min=1e-6)
                fz = (flux_dev - fm) / fs
                onset_gate = fz.clamp(min=0.0)
                offset_gate = (-fz).clamp(min=0.0)
                for pf in phon_factors:
                    for gate in (onset_gate, offset_gate):
                        p = pf * gate
                        m = p.mean(dim=0, keepdim=True)
                        s = p.std(dim=0, keepdim=True).clamp(min=1e-6)
                        z = (p - m) / s
                        inter.append(z.clamp(min=0.0))
                        inter.append((-z).clamp(min=0.0))
                        inter.append(z * z * z)

            if len(inter) > 0:
                add('interact_' + self.interact_set, torch.cat(inter, dim=1))

        # ============================================================
        # LEAN-v32-CPP: Cepstral Peak Prominence (1 scalar) - voice quality.
        # Cepstrum c = irfft(log_p). CPP = peak in pitch quefrency range minus
        # linear-trend baseline through the cepstrum. Multi-tau dynamics added.
        # Total: 1 raw + 3 multi-tau = 4 dim.
        # ============================================================
        log_pl = torch.log(torch.clamp(power_long, min=1e-10))
        cep = torch.fft.irfft(log_pl, n=4096)              # (T, 4096)
        q_min = int(SR / 400.0)
        q_max = int(SR / 60.0)
        cep_pitch = cep[:, q_min:q_max + 1]                # (T, ~625)
        peak_val_cep, _ = cep_pitch.max(dim=1, keepdim=True)
        # Linear trend baseline: fit slope+intercept across q_min..q_max
        L_ = cep_pitch.shape[1]
        idx = torch.arange(L_, device=cep.device, dtype=cep.dtype).view(1, -1)
        mu_i = idx.mean(dim=1, keepdim=True)
        mu_y = cep_pitch.mean(dim=1, keepdim=True)
        di = idx - mu_i
        dy = cep_pitch - mu_y
        slope = (di * dy).sum(dim=1, keepdim=True) / ((di * di).sum(dim=1, keepdim=True) + 1e-12)
        intercept = mu_y - slope * mu_i
        # Baseline at index where peak is found
        argmax = cep_pitch.argmax(dim=1, keepdim=True).float()
        baseline = intercept + slope * argmax
        cpp = peak_val_cep - baseline                       # (T, 1)
        cpp_blocks = [cpp]
        for tau in (4.5, 15.0, 60.0):
            cpp_blocks.append(cpp - leaky_lowpass(cpp, tau_frames=tau))
        add('cpp', torch.cat(cpp_blocks, dim=1))

        # ============================================================
        # LEAN-v33-REGIONFLAT: REGION-SPECIFIC spectral flatness scalars.
        # High-band (4000-8000 Hz) flatness distinguishes sibilance noise types.
        # Low-band (60-500 Hz) flatness distinguishes voiced/unvoiced low freq.
        # 2 scalars + multi-tau dyn.
        # ============================================================
        freqs_long = self.long_freqs
        hi_mask = (freqs_long >= 4000.0) & (freqs_long <= 8000.0)
        lo_mask = (freqs_long >= 60.0) & (freqs_long <= 500.0)
        p_hi = power_long[:, hi_mask]
        p_lo = power_long[:, lo_mask]
        flat_hi = (torch.log(p_hi.clamp(min=1e-10)).mean(dim=1, keepdim=True)
                   - torch.log(p_hi.mean(dim=1, keepdim=True) + 1e-10))
        flat_lo = (torch.log(p_lo.clamp(min=1e-10)).mean(dim=1, keepdim=True)
                   - torch.log(p_lo.mean(dim=1, keepdim=True) + 1e-10))
        region_flat = torch.cat([flat_hi, flat_lo], dim=1)
        rf_blocks = [region_flat]
        for tau in (4.5, 15.0, 60.0):
            rf_blocks.append(region_flat - leaky_lowpass(region_flat, tau_frames=tau))
        add('region_flat', torch.cat(rf_blocks, dim=1))

        # ============================================================
        # LEAN-v44-CONTRAST: SPECTRAL CONTRAST per octave band:
        # log(top 25% bins) - log(bottom 25% bins) — captures harmonic
        # prominence vs noise floor in each octave. 6 octaves.
        # 6 raw + multi-tau dyn.
        # ============================================================
        contrast_list = []
        for (f_lo, f_hi) in ((62.0, 125.0), (125.0, 250.0), (250.0, 500.0),
                              (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)):
            mk_o = (freqs_long >= f_lo) & (freqs_long <= f_hi)
            p_o = power_long[:, mk_o]
            if p_o.shape[1] < 4:  # too few FFT bins for a top/bottom contrast
                continue
            lp_o = torch.log(p_o.clamp(min=1e-10))
            n_o = lp_o.shape[1]
            k_top = max(1, n_o // 4)
            top_v, _ = lp_o.topk(k_top, dim=1, largest=True)
            bot_v, _ = lp_o.topk(k_top, dim=1, largest=False)
            cont_o = top_v.mean(dim=1, keepdim=True) - bot_v.mean(dim=1, keepdim=True)
            contrast_list.append(cont_o)
        contrast = torch.cat(contrast_list, dim=1)
        ct_blocks = [contrast]
        for tau in (4.5, 15.0, 60.0):
            ct_blocks.append(contrast - leaky_lowpass(contrast, tau_frames=tau))
        add('spectral_contrast', torch.cat(ct_blocks, dim=1))

        # ============================================================
        # LEAN-v45-SHORTCONTRAST: spectral contrast on SHORT 1024 window.
        # Captures transient/local peak-valley structure.
        # 6 octaves + multi-tau dyn.
        # ============================================================
        power_s = powers[0]  # (T, n_freqs_short)
        freqs_short_np = np.fft.rfftfreq(1024, d=1.0 / SR).astype(np.float32)
        freqs_short = torch.from_numpy(freqs_short_np).to(self.device)
        contrast_short_list = []
        for (f_lo, f_hi) in ((62.0, 125.0), (125.0, 250.0), (250.0, 500.0),
                              (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)):
            mk_so = (freqs_short >= f_lo) & (freqs_short <= f_hi)
            p_so = power_s[:, mk_so]
            if p_so.shape[1] < 2:  # 62-125 Hz has 1 bin at 1024 samples: skipped (v20 wrote zeros)
                continue
            lp_so = torch.log(p_so.clamp(min=1e-10))
            n_so = lp_so.shape[1]
            k_top_s = max(1, n_so // 3)
            top_v_s, _ = lp_so.topk(k_top_s, dim=1, largest=True)
            bot_v_s, _ = lp_so.topk(k_top_s, dim=1, largest=False)
            contrast_short_list.append(top_v_s.mean(dim=1, keepdim=True) -
                                         bot_v_s.mean(dim=1, keepdim=True))
        contrast_short = torch.cat(contrast_short_list, dim=1)
        cts_blocks = [contrast_short]
        for tau in (4.5, 15.0, 60.0):
            cts_blocks.append(contrast_short - leaky_lowpass(contrast_short, tau_frames=tau))
        add('contrast_short', torch.cat(cts_blocks, dim=1))

        # ============================================================
        # LEAN-v46-MEDCONTRAST: spectral contrast on MEDIUM 2048 window.
        # ============================================================
        power_m = powers[1]
        freqs_med_np = np.fft.rfftfreq(2048, d=1.0 / SR).astype(np.float32)
        freqs_med = torch.from_numpy(freqs_med_np).to(self.device)
        contrast_med_list = []
        for (f_lo, f_hi) in ((62.0, 125.0), (125.0, 250.0), (250.0, 500.0),
                              (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)):
            mk_m = (freqs_med >= f_lo) & (freqs_med <= f_hi)
            p_m = power_m[:, mk_m]
            if p_m.shape[1] < 2:  # too few FFT bins for a top/bottom contrast
                continue
            lp_m = torch.log(p_m.clamp(min=1e-10))
            n_m = lp_m.shape[1]
            k_top_m = max(1, n_m // 4)
            top_v_m, _ = lp_m.topk(k_top_m, dim=1, largest=True)
            bot_v_m, _ = lp_m.topk(k_top_m, dim=1, largest=False)
            contrast_med_list.append(top_v_m.mean(dim=1, keepdim=True) -
                                       bot_v_m.mean(dim=1, keepdim=True))
        contrast_med = torch.cat(contrast_med_list, dim=1)
        ctm_blocks = [contrast_med]
        for tau in (4.5, 15.0, 60.0):
            ctm_blocks.append(contrast_med - leaky_lowpass(contrast_med, tau_frames=tau))
        add('contrast_med', torch.cat(ctm_blocks, dim=1))

        # ============================================================
        # LEAN-v52-COCHLEACONTRAST: peak-valley contrast on the 20-channel
        # log-cochleagram itself. 5 groups of 4 adjacent channels, top-2 vs bot-2.
        # ============================================================
        cochlea_groups = []
        for gi in range(5):
            band = log_e_long[:, gi * 4:(gi + 1) * 4]
            top_v_c, _ = band.topk(2, dim=1, largest=True)
            bot_v_c, _ = band.topk(2, dim=1, largest=False)
            cochlea_groups.append(top_v_c.mean(dim=1, keepdim=True) -
                                    bot_v_c.mean(dim=1, keepdim=True))
        cochlea_contrast = torch.cat(cochlea_groups, dim=1)
        cc_blocks = [cochlea_contrast]
        for tau in (4.5, 15.0, 60.0):
            cc_blocks.append(cochlea_contrast - leaky_lowpass(cochlea_contrast, tau_frames=tau))
        add('cochlea_contrast', torch.cat(cc_blocks, dim=1))

        # ============================================================
        # LEAN-v54-MEDCOCHLEACONTRAST: cochleagram contrast on MEDIUM window.
        # 5 groups of 4 channels, top-2 vs bot-2.
        # ============================================================
        log_e_med = log_es[1]
        med_groups = []
        for gi in range(5):
            band = log_e_med[:, gi * 4:(gi + 1) * 4]
            top_v_m, _ = band.topk(2, dim=1, largest=True)
            bot_v_m, _ = band.topk(2, dim=1, largest=False)
            med_groups.append(top_v_m.mean(dim=1, keepdim=True) -
                                bot_v_m.mean(dim=1, keepdim=True))
        med_cochlea_contrast = torch.cat(med_groups, dim=1)
        mcc_blocks = [med_cochlea_contrast]
        for tau in (4.5, 15.0, 60.0):
            mcc_blocks.append(med_cochlea_contrast - leaky_lowpass(med_cochlea_contrast, tau_frames=tau))
        add('med_cochlea_contrast', torch.cat(mcc_blocks, dim=1))

        # ============================================================
        # LEAN-v55-SHORTCOCHLEACONTRAST: cochleagram contrast on SHORT 1024 win.
        # ============================================================
        log_e_short = log_es[0]
        short_groups = []
        for gi in range(5):
            band = log_e_short[:, gi * 4:(gi + 1) * 4]
            top_v_s2, _ = band.topk(2, dim=1, largest=True)
            bot_v_s2, _ = band.topk(2, dim=1, largest=False)
            short_groups.append(top_v_s2.mean(dim=1, keepdim=True) -
                                  bot_v_s2.mean(dim=1, keepdim=True))
        short_cochlea_contrast = torch.cat(short_groups, dim=1)
        scc_blocks = [short_cochlea_contrast]
        for tau in (4.5, 15.0, 60.0):
            scc_blocks.append(short_cochlea_contrast - leaky_lowpass(short_cochlea_contrast, tau_frames=tau))
        add('short_cochlea_contrast', torch.cat(scc_blocks, dim=1))

        # ============================================================
        # LEAN-v64-MULTITAUCONTRAST: cochleagram contrast on adapt-delta at
        # multiple taus (1.5, 15) — fast onset & syllabic contrast.
        # ============================================================
        mt_contrast_blocks = []
        for tau in (1.5, 15.0):
            le_adapt = log_e_long - leaky_lowpass(log_e_long, tau_frames=tau)
            for gi in range(5):
                band = le_adapt[:, gi * 4:(gi + 1) * 4]
                top_v_mt, _ = band.topk(2, dim=1, largest=True)
                bot_v_mt, _ = band.topk(2, dim=1, largest=False)
                mt_contrast_blocks.append(top_v_mt.mean(dim=1, keepdim=True) -
                                            bot_v_mt.mean(dim=1, keepdim=True))
        mtc = torch.cat(mt_contrast_blocks, dim=1)
        add('multitau_contrast', mtc)

        # ============================================================
        # LEAN-v67-MEDADAPTCONTRAST: adapt-delta cochleagram contrast on
        # MEDIUM 2048 window. tau=4.5 (phonetic). 5 groups of 4 channels.
        # ============================================================
        log_e_med2 = log_es[1]
        le_med_adapt = log_e_med2 - leaky_lowpass(log_e_med2, tau_frames=4.5)
        med_adapt_contrast = []
        for gi in range(5):
            band = le_med_adapt[:, gi * 4:(gi + 1) * 4]
            top_v_ma, _ = band.topk(2, dim=1, largest=True)
            bot_v_ma, _ = band.topk(2, dim=1, largest=False)
            med_adapt_contrast.append(top_v_ma.mean(dim=1, keepdim=True) -
                                        bot_v_ma.mean(dim=1, keepdim=True))
        mac = torch.cat(med_adapt_contrast, dim=1)
        add('med_adapt_contrast', mac)

        # ============================================================
        # LEAN-v82-TOPBOT: separate TOP-2 and BOT-2 cochleagram bands per group.
        # ============================================================
        topbot_lst = []
        for gi in range(5):
            band = log_e_long[:, gi * 4:(gi + 1) * 4]
            top_v_tb, _ = band.topk(2, dim=1, largest=True)
            bot_v_tb, _ = band.topk(2, dim=1, largest=False)
            topbot_lst.append(top_v_tb.mean(dim=1, keepdim=True))
            topbot_lst.append(bot_v_tb.mean(dim=1, keepdim=True))
        topbot = torch.cat(topbot_lst, dim=1)
        add('topbot', topbot)

        # ============================================================
        # LEAN-v84-TOPBOTSHORT: TOP-2/BOT-2 on SHORT 1024 cochleagram.
        # ============================================================
        log_e_short84 = log_es[0]
        topbot_short_lst = []
        for gi in range(5):
            band = log_e_short84[:, gi * 4:(gi + 1) * 4]
            tv, _ = band.topk(2, dim=1, largest=True)
            bv, _ = band.topk(2, dim=1, largest=False)
            topbot_short_lst.append(tv.mean(dim=1, keepdim=True))
            topbot_short_lst.append(bv.mean(dim=1, keepdim=True))
        topbot_short = torch.cat(topbot_short_lst, dim=1)
        add('topbot_short', topbot_short)

        # ============================================================
        # LEAN-v89-FUTCONTRAST: cochleagram contrast on FUTURE-shifted log_e (k=3).
        # ============================================================
        k_fut = 3
        log_e_fut = torch.cat([log_e_long[k_fut:], log_e_long[-1:].expand(k_fut, -1)], dim=0)
        fut_contrast = []
        for gi in range(5):
            band = log_e_fut[:, gi * 4:(gi + 1) * 4]
            tv, _ = band.topk(2, dim=1, largest=True)
            bv, _ = band.topk(2, dim=1, largest=False)
            fut_contrast.append(tv.mean(dim=1, keepdim=True) -
                                  bv.mean(dim=1, keepdim=True))
        fc = torch.cat(fut_contrast, dim=1)
        add('fut_cochlea_contrast', fc)

        # ============================================================
        # LEAN-v92-FUTCONTRASTMED: cochleagram contrast on FUTURE-shifted MED log_e.
        # ============================================================
        log_e_med92 = log_es[1]
        k_futm = 3
        log_e_fut_m = torch.cat([log_e_med92[k_futm:], log_e_med92[-1:].expand(k_futm, -1)], dim=0)
        fut_contrast_m = []
        for gi in range(5):
            band = log_e_fut_m[:, gi * 4:(gi + 1) * 4]
            tv, _ = band.topk(2, dim=1, largest=True)
            bv, _ = band.topk(2, dim=1, largest=False)
            fut_contrast_m.append(tv.mean(dim=1, keepdim=True) -
                                    bv.mean(dim=1, keepdim=True))
        fcm_ = torch.cat(fut_contrast_m, dim=1)
        add('fut_cochlea_contrast_med', fcm_)

        # ============================================================
        # LEAN-v96-FUTMED6: future-shifted MED cochleagram contrast k=6 (200ms).
        # ============================================================
        log_e_med96 = log_es[1]
        log_e_med_fut6 = torch.cat([log_e_med96[6:], log_e_med96[-1:].expand(6, -1)], dim=0)
        fut_med6 = []
        for gi in range(5):
            band = log_e_med_fut6[:, gi * 4:(gi + 1) * 4]
            tv, _ = band.topk(2, dim=1, largest=True)
            bv, _ = band.topk(2, dim=1, largest=False)
            fut_med6.append(tv.mean(dim=1, keepdim=True) -
                              bv.mean(dim=1, keepdim=True))
        fm6 = torch.cat(fut_med6, dim=1)
        add('fut_med_contrast_6', fm6)

        # ============================================================
        # LEAN-v100-FUTMINUSCUR: difference between future-shifted MED contrast
        # and current LONG contrast.
        # ============================================================
        cur_contrast_lst = []
        for gi in range(5):
            band = log_e_long[:, gi * 4:(gi + 1) * 4]
            tv, _ = band.topk(2, dim=1, largest=True)
            bv, _ = band.topk(2, dim=1, largest=False)
            cur_contrast_lst.append(tv.mean(dim=1, keepdim=True) -
                                      bv.mean(dim=1, keepdim=True))
        cur_contrast = torch.cat(cur_contrast_lst, dim=1)
        fmc_minus_cur = fm6 - cur_contrast
        add('fut_minus_cur_contrast', fmc_minus_cur)

        # ============================================================
        # LEAN-v107-FUTCUBELONG: future-shifted CUBE-ROOT LONG cochleagram contrast.
        # ============================================================
        cube_long = torch.pow(power_long @ self.fbs[-1] + 1e-10, 1.0 / 3.0)
        cube_long_fut = torch.cat([cube_long[3:], cube_long[-1:].expand(3, -1)], dim=0)
        cube_fut_contrast = []
        for gi in range(5):
            band = cube_long_fut[:, gi * 4:(gi + 1) * 4]
            tv, _ = band.topk(2, dim=1, largest=True)
            bv, _ = band.topk(2, dim=1, largest=False)
            cube_fut_contrast.append(tv.mean(dim=1, keepdim=True) -
                                       bv.mean(dim=1, keepdim=True))
        cfc = torch.cat(cube_fut_contrast, dim=1)
        add('cube_fut_contrast', cfc)

        # ============================================================
        # LEAN-v18.q1-QUADBANK: DENSE 2-D quadrature MODULATION BANK on the FINE
        # (40-channel) LONG log-cochleagram, keeping even (cos) and odd (sin) outputs
        # alongside the magnitude, averaged in three frequency bins (low/mid/high).
        # v21: scales (0, +/-0.25, +0.5), since the -0.5 kernel equals +0.5; and the 12 Hz
        # cos/sin outputs at scales 0 and 0.5 are dropped because they equal the 8 Hz ones
        # (sigma_t = period/2pi is < 1 frame at 8-12 Hz, so both kernels reduce to the same
        # 3-tap shape). 4 rates x 4 scales x 3 bins x 3 phases - 12 = 132 raw dims.
        # ============================================================
        log_e_fine_long = torch.log(torch.clamp(power_long @ self.fb_fine, min=1e-10))  # (T, 40)
        nF40 = log_e_fine_long.shape[1]
        bin_edges = [(0, nF40 // 3), (nF40 // 3, 2 * nF40 // 3), (2 * nF40 // 3, nF40)]
        quad_drop = {(12.0, wf, ph) for wf in (0.0, 0.5) for ph in ('cos', 'sin')}
        quad_raw, quad_mags = self._quad_bank(log_e_fine_long, self.QUAD_RATES,
                                              self.QUAD_SCALES, bin_edges, drop=quad_drop)
        add('quad_modbank', quad_raw)
        # LEAN-v18.q2/q14: multi-tau adapt-delta (phonetic 4.5, syllabic 15, prosodic 60)
        quad_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            quad_dyn_blocks.append(quad_raw - leaky_lowpass(quad_raw, tau_frames=tau))
        add('quad_modbank_dyn', torch.cat(quad_dyn_blocks, dim=1))
        # LEAN-v18.q15: ACAUSAL prosodic dyn -- deviation from the next ~2 s
        add('quad_modbank_acausal', quad_raw - acausal_leaky_lowpass(quad_raw, tau_frames=60.0))

        # LEAN-v18.q17: cross-rate ENERGY coincidence. Energy per (rate, bin) = sum over
        # scales of the magnitude; log1p of products across rate pairs (slow x fast
        # co-occurrence), 6 pairs x 3 bins = 18. v21: the sum runs over 4 distinct scales
        # (v20 counted the 0.5 scale twice through its -0.5 copy).
        energy_per_rate_bin = quad_mags.sum(dim=2)          # (T, 4 rates, 3 bins)
        xrate_pairs = [(0, 2), (0, 3), (1, 3), (1, 2), (2, 3), (0, 1)]
        xrate_raw = torch.cat([energy_per_rate_bin[:, i, :] * energy_per_rate_bin[:, j, :]
                               for (i, j) in xrate_pairs], dim=1)          # (T, 18)
        xrate_log = torch.log1p(xrate_raw.clamp(min=0.0))
        add('quad_xrate_energy', xrate_log)
        xrate_dyn = []
        for tau in (4.5, 15.0):
            xrate_dyn.append(xrate_log - leaky_lowpass(xrate_log, tau_frames=tau))
        add('quad_xrate_energy_dyn', torch.cat(xrate_dyn, dim=1))

        # (q18 COCHLEAACAUSALLONG REVERTED 0.09005 < q15 0.09015)

        # LEAN-v18.q19: PERIODICITY-GATED cochleagram. Multiply each of 20 gammatone
        # channels (long-window log_e) by a soft "voiced" probability: voiced 20 +
        # complement 20 = 40, plus the syllabic (tau=15) deviation of the voiced part = 20.
        # v21 fix: the gate is driven by the ACF voicing peak hnr_peak (0..1). v20 used
        # best_strength (~6-10), so the gate was exactly 1.0 on every frame and these two
        # blocks duplicated cochleagram and adapt_delta.
        hnr_peak = hnr[:, 0:1]
        gate = torch.sigmoid((hnr_peak - 0.3) * 8.0)             # (T, 1) soft
        gated_cochlea = log_e_long * gate                         # (T, 20)
        unvoiced_cochlea = log_e_long * (1.0 - gate)              # (T, 20)
        add('cochlea_voiced_gated', torch.cat([gated_cochlea, unvoiced_cochlea], dim=1))
        gated_dyn = gated_cochlea - leaky_lowpass(gated_cochlea, tau_frames=15.0)
        add('cochlea_voiced_gated_dyn', gated_dyn)

        # (q20 VOICEDQUADMAG REVERTED 0.0902 < q19 0.09021)
        # (q21 VOICINGONSET REVERTED 0.09018 < q19 0.09021)
        # (q22 MULTITHRESHGATE REVERTED 0.09014 < q19 0.09021)
        # (q23 FLATGATE REVERTED 0.09019 < q19 0.09021)
        # (q24 VOICEDFINE REVERTED 0.0901 < q19 0.09021)
        # (q25 VOICEDMULTITAU REVERTED 0.09011 < q19 0.09021)
        # (q26 ZCR REVERTED 0.08991 < q19 0.09021)
        # (q27 CUBEFINE REVERTED 0.08997 < q19 0.09021)
        # (q28 FUTURELAGS REVERTED 0.09015 < q19 0.09021)

        # LEAN-v18.q29: MULTI-THRESHOLD VOICING gate SIGNALS + multi-tau deviations.
        # v21 fix: driven by hnr_peak (0..1) instead of the saturated best_strength; v20's
        # flatness gates (thresholds 0.2-0.6 on log-flatness <= 0) never opened: removed.
        gate_cat = torch.cat([torch.sigmoid((hnr_peak - th) * 8.0)
                              for th in (0.15, 0.3, 0.6)], dim=1)   # (T, 3)
        add('gate_signals', gate_cat)
        gate_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            gate_dyn_blocks.append(gate_cat - leaky_lowpass(gate_cat, tau_frames=tau))
        add('gate_signals_dyn', torch.cat(gate_dyn_blocks, dim=1))

        # (q31 DIVERSEGATES REVERTED 0.09027 < q29 0.09033)

        # (q32 GATEXLAG REVERTED 0.09019 < q29 0.09033)
        # (q33 GATExKEYSIG REVERTED 0.09029 < q29 0.09033)
        # (q34 WAVESTATS REVERTED 0.0901 < q29 0.09033)
        # (q35 SHARPGATES REVERTED 0.09029 < q29 0.09033)
        # (q36 BANDRATIOS REVERTED 0.09013 < q29 0.09033)
        # (q37 LOUDHNRGATES REVERTED 0.09029 < q29 0.09033)
        # (q38 ALTGATES REVERTED 0.09025 < q29 0.09033)
        # (q39 GATEEXTRATAU tied 0.09033, neutral — REVERTED to keep dim low)
        # (q40 SOFTARGMAX REVERTED 0.08983 < q29 0.09033)
        # (q41 GATEFUTURE REVERTED 0.09017 < q29 0.09033)

        # LEAN-v18.q42-q45: SLOW quadrature bank at prosodic rates + 1 s deviation (tau=30).
        # v21: same 4 scales as the quadrature bank; the raw 2 Hz outputs are dropped
        # because they equal quad_modbank's 2 Hz outputs, but the 2 Hz tau=30 deviation (not
        # in quad_modbank_dyn) is kept. Raw 3 rates x 36 = 108; dyn 4 rates x 36 = 144.
        slow_all, _ = self._quad_bank(log_e_fine_long, self.SLOW_RATES, self.QUAD_SCALES, bin_edges)
        per_rate = slow_all.shape[1] // len(self.SLOW_RATES)
        keep = [i for i, r in enumerate(self.SLOW_RATES) if r not in self.QUAD_RATES]
        add('slow_quad_modbank', torch.cat([slow_all[:, i * per_rate:(i + 1) * per_rate]
                                            for i in keep], dim=1))
        add('slow_quad_modbank_dyn', slow_all - leaky_lowpass(slow_all, tau_frames=30.0))

        feat = self._concat_groups(groups)
        # ---------------- SCAT block ported from SCAT-v103 ----------------
        if getattr(self, 'scat_enabled', True):
            scatter = self._wavelet_scattering(x)
            feat = torch.cat([feat, scatter], dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "LEAN-v21"
model_description = (
    "LEAN-v21 = LEAN-v20 with bugs fixed, no new features: voicing gates driven by the ACF "
    "voicing peak (v20's were saturated at 1.0, duplicating cochleagram/adapt_delta); flatness "
    "gates removed (never opened); identical filters removed (-0.5 cyc/ch copies, slow-bank raw "
    "2 Hz = quad 2 Hz, 12 Hz quad cos/sin at scales 0/0.5 = 8 Hz); dead columns removed "
    "(scattering S2 paths < 4 Hz, short-window contrast octave with 1 FFT bin); interact_set "
    "defaults to v3e; unused code removed. Parent LEAN-v20: all-166 0.0808 / grid 0.0904."
)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------

def build_embedder(device: str = "cuda", interact_set: str = "v3e") -> WaveEmbedder:
    emb = WaveEmbedder(device=device)
    emb.interact_set = interact_set
    return emb


if __name__ == "__main__":
    import argparse, time, csv as _csv
    import numpy as np
    from src import data, features, encoding
    from src.eval import _per_chunk_elec_corr
    _install_disk_features()  # disk-backed feat memmap → fits 500/100 at high P
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interact-set", default="v3e")
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-grid", type=int, default=166)  # patient1 surface grid = electrodes 0..124
    args = parser.parse_args()
    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        args.device = "cpu"
    t0 = time.time()
    emb = build_embedder(device=args.device, interact_set=args.interact_set)
    train, test = data.get_chunk_names(args.num_train, args.num_test)
    stim = data.load_stim(train + test); resp = data.load_responses(train + test)
    ftr, rtr = features.get_features_per_chunk(stim, resp, train, emb)
    fte, rte = features.get_features_per_chunk(stim, resp, test, emb)
    G = args.n_grid
    rtr = [r[:, :G] for r in rtr]; rte = [r[:, :G] for r in rte]   # GRID-ONLY responses
    wt, corrs, _ = encoding.bootstrap_ridge(
        np.vstack(ftr).astype(np.float32), np.vstack(rtr).astype(np.float32),
        np.vstack(fte).astype(np.float32), np.vstack(rte).astype(np.float32), np.logspace(1, 4, 12))
    test_corr, _ = _per_chunk_elec_corr(fte, rte, wt)
    train_corr, _ = _per_chunk_elec_corr(ftr, rtr, wt)
    frac = float(np.mean(np.nan_to_num(corrs) > 0.1))
    p = int(np.vstack(fte[:1]).shape[1])
    name = args.name or model_shorthand_name
    with open(os.path.join(RESULTS_DIR, "overall_results.csv"), "a", newline="") as fh:
        _csv.writer(fh).writerow(["patient1", round(test_corr,5), round(train_corr,5), round(frac,4),
                                  round(time.time()-t0,1), "success", name, p, model_description+" GRID-ONLY"])
    print()
    print("---")
    print(f"subject:        patient1")
    print(f"test_corr:      {test_corr:.4f}  (train_corr={train_corr:.4f}, frac_elec>0.1={frac:.4f})  [GRID-ONLY n={G}]")
    print(f"total_seconds:  {time.time()-t0:.1f}s")
