"""Textbook acoustic feature space for ECoG high-gamma encoding-model baseline.

All features are fixed, closed-form signal processing: no learning, no fitting,
no data-estimated parameters. Features are drawn from standard practice in
auditory neuroscience, speech science, and audio signal processing.

Feature groups (dimensionalities in parentheses):

- Log-mel spectrogram, 64 bands, 50 Hz to Nyquist, Slaney-normalized (64)
- Gammatone cochleagram, 32 ERB-spaced bands, 4th-order mag^2 approximation (32)
- MFCCs (13) + Delta MFCC (13) + Delta-Delta MFCC (13) (39)
- Spectral shape: centroid, bandwidth, 5 rolloff percentiles, flatness,
  slope, skewness, excess kurtosis, entropy, crest, flux, zero-crossing rate (15)
- Short-time log-RMS energy and log-peak amplitude (2)
- Log chroma, 12 pitch classes (12)
- Autocorrelation log-F0 and voicing (2)
- Octave spectral contrast (Jiang et al.), 6 bands peaks + valleys (12)
- Per-frame temporal modulation energies, 32 mel bands x 5 modulation rates
  {0.5, 1, 2, 4, 8} Hz, zero-phase Butterworth bandpass across frames (160)
- Half-wave rectified log-mel onset envelope (1)

Total H = 339.
"""

import numpy as np
from scipy import signal as _signal
from scipy.fft import rfft as _rfft, dct as _dct

SR = 44100
HOP = 1470  # 1/30 s
FRAME_RATE = SR / HOP  # 30.0 Hz
EPS = 1e-10


def _safe_log(x):
    return np.log(np.maximum(x, EPS))


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_bands, n_fft, sr, fmin=50.0, fmax=None):
    """Slaney-normalized triangular mel filterbank."""
    if fmax is None:
        fmax = sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_bands + 2)
    hzs = _mel_to_hz(mels)
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_bands, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_bands):
        lo, ctr, hi = hzs[m], hzs[m + 1], hzs[m + 2]
        up = (freqs - lo) / max(ctr - lo, 1e-12)
        dn = (hi - freqs) / max(hi - ctr, 1e-12)
        tri = np.maximum(0.0, np.minimum(up, dn))
        fb[m] = tri * (2.0 / max(hi - lo, 1e-12))
    return fb


def _erb(f):
    """Glasberg & Moore ERB bandwidth, Hz."""
    return 24.7 * (1.0 + 4.37 * f / 1000.0)


def _gammatone_filterbank(n_bands, n_fft, sr, fmin=50.0, fmax=None):
    """4th-order gammatone |H(f)|^2 magnitude-response approximation,
    ERB-rate-spaced center frequencies (Patterson/Slaney style)."""
    if fmax is None:
        fmax = sr / 2.0
    # ERB-rate (Glasberg & Moore cam scale)
    erb_lo = 21.4 * np.log10(1.0 + 0.00437 * fmin)
    erb_hi = 21.4 * np.log10(1.0 + 0.00437 * fmax)
    erb_n = np.linspace(erb_lo, erb_hi, n_bands)
    cfs = (10.0 ** (erb_n / 21.4) - 1.0) / 0.00437
    bws = 1.019 * _erb(cfs)
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_bands, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_bands):
        fb[m] = 1.0 / (1.0 + ((freqs - cfs[m]) / bws[m]) ** 2) ** 4
    fb /= (fb.max(axis=1, keepdims=True) + 1e-12)
    return fb


def _chroma_filterbank(fft_freqs, n_chroma=12, fref=440.0,
                       f_lo=80.0, f_hi=5000.0):
    """Triangular-soft chroma projection matrix (n_chroma, F)."""
    F = len(fft_freqs)
    chroma = np.zeros((n_chroma, F), dtype=np.float32)
    f = np.maximum(fft_freqs, 1.0)
    pc = (12.0 * np.log2(f / fref) + 69.0) % 12.0
    mask = (fft_freqs >= f_lo) & (fft_freqs <= f_hi)
    for c in range(n_chroma):
        d = np.abs(((pc - c + 6.0) % 12.0) - 6.0)
        w = np.maximum(0.0, 1.0 - d).astype(np.float32)
        w = w * mask
        chroma[c] = w
    return chroma


def _deltas(x, N=2):
    """Standard delta coefficients (HTK/Kaldi): sum_{n=1..N} n*(x[t+n]-x[t-n]) / (2*sum n^2)."""
    T = x.shape[0]
    denom = 2.0 * sum(n * n for n in range(1, N + 1))
    padded = np.pad(x, ((N, N), (0, 0)), mode='edge')
    out = np.zeros_like(x)
    for n in range(1, N + 1):
        out += n * (padded[N + n:N + n + T] - padded[N - n:N - n + T])
    return out / denom


def _autocorr_f0(x, sr=SR, fmin=60.0, fmax=500.0):
    """Autocorrelation-based F0 and voicing per frame (Boersma-style, no interp)."""
    T, W = x.shape
    lag_min = max(2, int(np.floor(sr / fmax)))
    lag_max = min(W - 1, int(np.ceil(sr / fmin)))
    xm = x - x.mean(axis=1, keepdims=True)
    nfft = 1
    while nfft < 2 * W:
        nfft <<= 1
    X = np.fft.rfft(xm, n=nfft, axis=1)
    acf = np.fft.irfft(X * np.conj(X), n=nfft, axis=1)[:, :lag_max + 1]
    zero = acf[:, 0:1] + EPS
    acfn = acf / zero
    sub = acfn[:, lag_min:lag_max + 1]
    best = np.argmax(sub, axis=1) + lag_min
    peak = acfn[np.arange(T), best]
    f0 = sr / np.maximum(best.astype(np.float32), 1.0)
    log_f0 = _safe_log(f0).astype(np.float32)
    voicing = np.clip(peak, 0.0, 1.0).astype(np.float32)
    return log_f0, voicing


def _spectral_contrast(power, fft_freqs, n_bands=6, alpha=0.02, f_lo=200.0):
    """Jiang et al. 2002: log peak minus log valley in octave-spaced bands."""
    T, F = power.shape
    hi = fft_freqs[-1]
    edges = np.logspace(np.log10(f_lo), np.log10(hi), n_bands + 1)
    peaks = np.zeros((T, n_bands), dtype=np.float32)
    valleys = np.zeros((T, n_bands), dtype=np.float32)
    for b in range(n_bands):
        lo, hi_b = edges[b], edges[b + 1]
        idxs = np.where((fft_freqs >= lo) & (fft_freqs < hi_b))[0]
        if len(idxs) < 4:
            continue
        sub = np.sort(power[:, idxs], axis=1)
        k = max(1, int(np.ceil(alpha * sub.shape[1])))
        peaks[:, b] = _safe_log(sub[:, -k:].mean(axis=1))
        valleys[:, b] = _safe_log(sub[:, :k].mean(axis=1))
    return peaks, valleys


def _modulation_energies(env, frame_rate=FRAME_RATE,
                         mod_freqs=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Per-frame envelope modulation energy at each (band, mod-rate).
    Zero-phase Butterworth bandpass across frames; output |filtered|.
    env: (T, B). Returns (T, B * len(mod_freqs))."""
    T, B = env.shape
    ny = frame_rate / 2.0
    env = env - env.mean(axis=0, keepdims=True)
    parts = []
    for fm in mod_freqs:
        bw = max(fm / 2.0, 0.5)
        lo = max(fm - bw / 2.0, 0.05) / ny
        hi = min(fm + bw / 2.0, ny * 0.98) / ny
        sos = _signal.butter(4, [lo, hi], btype='bandpass', output='sos')
        y = _signal.sosfiltfilt(sos, env, axis=0)
        parts.append(np.abs(y).astype(np.float32))
    return np.concatenate(parts, axis=1)


class FeatureSpace:
    """Fixed-parameter textbook acoustic features for ECoG encoding models.

    Input per segment: windows (T, win) float32, 44.1 kHz, frames 1/30 s apart.
    Output: (T, 339) float32."""

    win = 2048  # ~46 ms analysis window at 44.1 kHz

    def __init__(self, device='cpu'):
        self.device = device
        self.n_fft = 2048
        self.hann = _signal.windows.hann(self.win, sym=False).astype(np.float32)
        self.n_mel = 64
        self.n_gt = 32
        self.n_mfcc = 13
        self.n_mod_mel = 32
        self.mod_freqs = (0.5, 1.0, 2.0, 4.0, 8.0)
        self.mel_fb = _mel_filterbank(self.n_mel, self.n_fft, SR,
                                      fmin=50.0, fmax=SR / 2.0)
        self.gt_fb = _gammatone_filterbank(self.n_gt, self.n_fft, SR,
                                           fmin=50.0, fmax=SR / 2.0 * 0.95)
        self.mod_mel_fb = _mel_filterbank(self.n_mod_mel, self.n_fft, SR,
                                          fmin=50.0, fmax=SR / 2.0)
        self.fft_freqs = np.linspace(0.0, SR / 2.0,
                                     self.n_fft // 2 + 1).astype(np.float32)
        self.chroma_fb = _chroma_filterbank(self.fft_freqs, n_chroma=12)
        self.log_freqs = _safe_log(self.fft_freqs + 1.0).astype(np.float32)
        self._log_f_c = self.log_freqs - self.log_freqs.mean()
        self._log_f_var = float((self._log_f_c * self._log_f_c).sum())

    def __call__(self, windows):
        windows = np.ascontiguousarray(windows, dtype=np.float32)
        T, W = windows.shape

        # Normalize int16-scale audio to [-1, 1] range (fixed scaling).
        x = windows * np.float32(1.0 / 32768.0)

        # Pre-emphasis y[n] = x[n] - 0.97*x[n-1] (standard speech preproc).
        xp = np.empty_like(x)
        xp[:, 0] = x[:, 0]
        xp[:, 1:] = x[:, 1:] - np.float32(0.97) * x[:, :-1]

        # Windowed FFT.
        xw = xp * self.hann
        X = _rfft(xw, n=self.n_fft, axis=1)
        mag = np.abs(X).astype(np.float32)
        power = (mag * mag).astype(np.float32)

        # Log-mel spectrogram.
        mel_pow = power @ self.mel_fb.T
        log_mel = _safe_log(mel_pow).astype(np.float32)

        # Log-gammatone cochleagram.
        gt_pow = power @ self.gt_fb.T
        log_gt = _safe_log(gt_pow).astype(np.float32)

        # MFCCs + deltas + delta-deltas.
        mfcc = _dct(log_mel, type=2, norm='ortho', axis=1)[:, :self.n_mfcc]
        mfcc = mfcc.astype(np.float32)
        dmfcc = _deltas(mfcc).astype(np.float32)
        ddmfcc = _deltas(dmfcc).astype(np.float32)

        # Spectral shape.
        mag_sum = mag.sum(axis=1, keepdims=True) + EPS
        p = mag / mag_sum
        freqs = self.fft_freqs[None, :]
        centroid = (p * freqs).sum(axis=1)
        bw2 = ((freqs - centroid[:, None]) ** 2 * p).sum(axis=1)
        bw = np.sqrt(np.maximum(bw2, EPS))
        # Rolloff at 25/50/75/85/95% cumulative magnitude.
        cum = np.cumsum(mag, axis=1)
        tot = cum[:, -1:] + EPS
        rolls = []
        for pct in (0.25, 0.5, 0.75, 0.85, 0.95):
            idx = np.argmax(cum >= pct * tot, axis=1)
            rolls.append(self.fft_freqs[idx])
        rolloff = np.stack(rolls, axis=1).astype(np.float32)
        # Flatness (Wiener entropy), log.
        log_pow = _safe_log(power)
        log_flatness = (log_pow.mean(axis=1)
                        - _safe_log(power.mean(axis=1))).astype(np.float32)
        # Slope (linear fit of log power vs log freq).
        log_p_c = log_pow - log_pow.mean(axis=1, keepdims=True)
        slope = ((log_p_c * self._log_f_c[None, :]).sum(axis=1)
                 / (self._log_f_var + EPS)).astype(np.float32)
        # Skewness and excess kurtosis of normalized spectrum.
        dn = (freqs - centroid[:, None]) / bw[:, None]
        skew = ((dn ** 3) * p).sum(axis=1).astype(np.float32)
        kurt = (((dn ** 4) * p).sum(axis=1) - 3.0).astype(np.float32)
        # Entropy.
        entropy = (-(p * _safe_log(p)).sum(axis=1)).astype(np.float32)
        # Crest factor (log).
        log_crest = _safe_log(mag.max(axis=1) / (mag.mean(axis=1) + EPS)).astype(np.float32)
        # Spectral flux (log L2 of positive differences).
        prev = np.vstack([mag[:1], mag[:-1]])
        flux = np.sqrt(((mag - prev) ** 2).sum(axis=1) + EPS).astype(np.float32)
        log_flux = _safe_log(flux)
        # Zero-crossing rate (on raw waveform).
        sgn = (x >= 0.0)
        zcr = (sgn[:, 1:] != sgn[:, :-1]).mean(axis=1).astype(np.float32)

        # Log-RMS and log-peak amplitude.
        rms = np.sqrt((x ** 2).mean(axis=1) + EPS).astype(np.float32)
        log_rms = _safe_log(rms).astype(np.float32)
        log_peak = _safe_log(np.abs(x).max(axis=1) + EPS).astype(np.float32)

        # Log chroma.
        chroma = mag @ self.chroma_fb.T
        log_chroma = _safe_log(chroma).astype(np.float32)

        # F0 and voicing from autocorrelation.
        log_f0, voicing = _autocorr_f0(x, sr=SR)

        # Octave spectral contrast.
        peaks, valleys = _spectral_contrast(power, self.fft_freqs, n_bands=6)

        # Modulation energies on mel log-envelopes.
        mod_env = _safe_log(power @ self.mod_mel_fb.T).astype(np.float32)
        mod = _modulation_energies(mod_env, frame_rate=FRAME_RATE,
                                   mod_freqs=self.mod_freqs).astype(np.float32)

        # Onset envelope (half-wave rectified mean log-mel diff).
        prev_lm = np.vstack([log_mel[:1], log_mel[:-1]])
        onset = np.maximum(log_mel - prev_lm, 0.0).mean(axis=1).astype(np.float32)

        out = np.concatenate([
            log_mel,                       # 64
            log_gt,                        # 32
            mfcc,                          # 13
            dmfcc,                         # 13
            ddmfcc,                        # 13
            centroid[:, None].astype(np.float32),
            bw[:, None].astype(np.float32),
            rolloff,                       # 5
            log_flatness[:, None],
            slope[:, None],
            skew[:, None],
            kurt[:, None],
            entropy[:, None],
            log_crest[:, None],
            log_flux[:, None],
            zcr[:, None],                  # shape total: 15
            log_rms[:, None],
            log_peak[:, None],             # energy: 2
            log_chroma,                    # 12
            log_f0[:, None],
            voicing[:, None],              # F0: 2
            peaks,                         # 6
            valleys,                       # 6
            mod,                           # 160
            onset[:, None],                # 1
        ], axis=1).astype(np.float32)

        # Safety: replace any non-finite with 0 (should not occur with safe_log).
        if not np.all(np.isfinite(out)):
            out = np.where(np.isfinite(out), out, np.float32(0.0)).astype(np.float32)
        return out
