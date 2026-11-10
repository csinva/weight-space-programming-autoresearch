"""Conventional interpretable acoustic feature space.

Textbook features from auditory neuroscience, speech science and audio signal
processing: log-mel spectrogram, MFCCs with delta / delta-delta, spectral shape
descriptors, broadband envelope and zero-crossing rate, autocorrelation pitch
with voicing confidence, 12-bin chroma, and temporal modulation of the log-mel
subband envelopes (Shamma / Elhilali-style). Closed-form, fixed parameters.
"""

import numpy as np
import scipy.fft
import scipy.signal


SR = 44100
FRAME_RATE = 30.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _make_mel_fb(n_fft, sr, n_mels, fmin, fmax):
    """Slaney-normalised triangular mel filter bank."""
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    hz_pts = _mel_to_hz(np.linspace(mel_min, mel_max, n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (freqs - l) / (c - l)
        right = (r - freqs) / (r - c)
        tri = np.clip(np.minimum(left, right), 0.0, None)
        fb[i] = tri * (2.0 / (r - l))
    return fb


def _make_dct(n_out, n_in):
    """Type-II orthonormal DCT matrix (HTK-compatible), shape (n_out, n_in)."""
    n = np.arange(n_in)
    k = np.arange(n_out).reshape(-1, 1)
    m = np.cos(np.pi * (n + 0.5) * k / n_in) * np.sqrt(2.0 / n_in)
    m[0] *= 1.0 / np.sqrt(2.0)
    return m.astype(np.float32)


def _make_chroma_fb(n_fft, sr, n_chroma=12, fmin=55.0, fmax=8000.0):
    """Gaussian-tapered pitch-class filterbank (chroma)."""
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    freqs_safe = np.where(freqs > 0, freqs, 1e-6)
    midi = 69.0 + 12.0 * np.log2(freqs_safe / 440.0)
    pc = np.mod(midi, 12.0)
    fb = np.zeros((n_chroma, n_fft // 2 + 1), dtype=np.float32)
    for k in range(n_chroma):
        diff = np.minimum(np.abs(pc - k), 12.0 - np.abs(pc - k))
        w = np.exp(-0.5 * (diff / 1.0) ** 2)  # sigma = 1 semitone
        mask = (freqs >= fmin) & (freqs <= fmax)
        fb[k] = (w * mask).astype(np.float32)
    # L1-normalise rows so each pitch class has unit response
    s = fb.sum(axis=1, keepdims=True) + 1e-8
    return (fb / s).astype(np.float32)


def _htk_delta(x, N=2):
    """Standard HTK regression deltas across +/- N frames, edge-padded."""
    T = x.shape[0]
    denom = 2.0 * sum(n * n for n in range(1, N + 1))
    padded = np.pad(x, ((N, N), (0, 0)), mode="edge")
    d = np.zeros_like(x)
    for n in range(1, N + 1):
        d += n * (padded[N + n : N + n + T] - padded[N - n : N - n + T])
    return d / denom


class FeatureSpace:
    win = 2048  # ~46 ms analysis window at 44.1 kHz

    def __init__(self, device="cpu"):
        self.device = device
        self.sr = SR
        self.n_fft = self.win

        # Pre-emphasis coefficient (standard speech processing choice)
        self.preemph = 0.97

        # Hann analysis window (standard choice in speech/audio)
        self.window = np.hanning(self.win).astype(np.float32)

        # Log-mel spectrogram: 80 mel bands, 50 Hz - 20 kHz
        self.n_mels = 80
        self.fmin = 50.0
        self.fmax = 20000.0
        self.mel_fb = _make_mel_fb(self.n_fft, SR, self.n_mels, self.fmin, self.fmax)

        # MFCC: 20 cepstral coefficients (standard C0..C19)
        self.n_mfcc = 20
        self.dct_mat = _make_dct(self.n_mfcc, self.n_mels)

        # Chroma: 12 pitch classes
        self.n_chroma = 12
        self.chroma_fb = _make_chroma_fb(self.n_fft, SR, self.n_chroma)

        # Pre-computed FFT bin frequencies for spectral descriptors
        self.freqs = (np.arange(self.n_fft // 2 + 1) * SR / self.n_fft).astype(np.float32)
        self.log_freqs = np.log(self.freqs + 1.0).astype(np.float32)

        # Temporal modulation of mel subband envelopes (Shamma / Elhilali style).
        # Butterworth bandpasses tiling 1-12 Hz over 30 Hz frame rate.
        self.mod_rates = (1.0, 2.0, 4.0, 8.0, 12.0)
        self.mod_sos = []
        nyq = FRAME_RATE / 2.0
        for r in self.mod_rates:
            lo = max(r / np.sqrt(2.0), 0.3)
            hi = min(r * np.sqrt(2.0), nyq - 0.2)
            sos = scipy.signal.butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
            self.mod_sos.append(sos.astype(np.float64))

        # Spectral-contrast octave sub-band edges (Jiang et al., ICASSP 2002).
        # 7 bands: [0, 200, 400, 800, 1600, 3200, 6400, 12800) Hz
        self.sc_edges = np.array([0.0, 200.0, 400.0, 800.0, 1600.0,
                                  3200.0, 6400.0, 12800.0], dtype=np.float32)
        self.sc_alpha = 0.02  # neighborhood fraction for peak / valley estimation

        # Autocorrelation pitch range (speech-biased): 70 - 500 Hz
        self.pitch_fmin = 70.0
        self.pitch_fmax = 500.0

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def __call__(self, windows):
        x = np.asarray(windows, dtype=np.float32)
        # Normalise by nominal int16 range so log-based features are well-conditioned
        x = x / 32768.0
        T, W = x.shape

        # --- pre-emphasis & windowing ---------------------------------
        pre = np.empty_like(x)
        pre[:, 0] = x[:, 0]
        pre[:, 1:] = x[:, 1:] - self.preemph * x[:, :-1]
        xw = pre * self.window[None, :]

        # --- STFT -----------------------------------------------------
        X = scipy.fft.rfft(xw, axis=-1)
        P = (X.real.astype(np.float32) ** 2 + X.imag.astype(np.float32) ** 2)  # power
        M = np.sqrt(P + 1e-12).astype(np.float32)  # magnitude

        # --- log-mel spectrogram -------------------------------------
        mel_p = P @ self.mel_fb.T
        log_mel = np.log(mel_p + 1e-6).astype(np.float32)
        d_log_mel = _htk_delta(log_mel).astype(np.float32)
        dd_log_mel = _htk_delta(d_log_mel).astype(np.float32)

        # --- MFCC + deltas -------------------------------------------
        mfcc = (log_mel @ self.dct_mat.T).astype(np.float32)
        d_mfcc = _htk_delta(mfcc).astype(np.float32)
        dd_mfcc = _htk_delta(d_mfcc).astype(np.float32)

        # --- chroma --------------------------------------------------
        chroma_raw = M @ self.chroma_fb.T
        # L2-normalise per frame (standard) with floor to avoid zero frames
        n = np.linalg.norm(chroma_raw, axis=1, keepdims=True) + 1e-6
        chroma = (chroma_raw / n).astype(np.float32)

        # --- spectral shape descriptors ------------------------------
        Psum = P.sum(axis=-1) + 1e-10
        # spectral centroid
        centroid = (P * self.freqs[None, :]).sum(axis=-1) / Psum
        # spectral bandwidth (p=2 variance)
        bw = np.sqrt(((self.freqs[None, :] - centroid[:, None]) ** 2 * P).sum(axis=-1) / Psum)
        # spectral rolloff (85 %)
        cumP = np.cumsum(P, axis=-1)
        thr = 0.85 * Psum[:, None]
        idx = np.argmax(cumP >= thr, axis=-1)
        rolloff = idx.astype(np.float32) * (SR / self.n_fft)
        # spectral flatness (Wiener entropy): geometric / arithmetic mean of power
        logP = np.log(P + 1e-10)
        flatness = np.exp(logP.mean(axis=-1)) / (Psum / P.shape[-1] + 1e-10)
        # spectral slope: regression of log-power on log-freq
        lf = self.log_freqs
        lf_mean = lf.mean()
        lf_c = lf - lf_mean
        denom = (lf_c * lf_c).sum() + 1e-10
        logP_mean = logP.mean(axis=-1, keepdims=True)
        slope = ((logP - logP_mean) * lf_c[None, :]).sum(axis=-1) / denom
        # spectral flux (half-wave rectified, L2)
        flux = np.zeros(T, dtype=np.float32)
        if T > 1:
            diff = np.maximum(M[1:] - M[:-1], 0.0)
            flux[1:] = (diff * diff).sum(axis=-1)
        # spectral crest (peak / mean)
        crest = P.max(axis=-1) / (Psum / P.shape[-1] + 1e-10)
        # spectral entropy (normalised Shannon entropy of power distribution)
        Pn = P / Psum[:, None]
        sp_ent = -(Pn * np.log(Pn + 1e-10)).sum(axis=-1) / np.log(P.shape[-1])

        spectral = np.stack(
            [
                np.log(centroid + 1.0),
                np.log(bw + 1.0),
                np.log(rolloff + 1.0),
                np.log(flatness + 1e-10),
                slope.astype(np.float32),
                np.log(flux + 1e-10),
                np.log(crest + 1e-10),
                sp_ent.astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        # --- broadband envelope, ZCR ---------------------------------
        rms = np.sqrt((x * x).mean(axis=-1) + 1e-12).astype(np.float32)
        log_rms = np.log(rms + 1e-8).astype(np.float32)
        peak = np.abs(x).max(axis=-1).astype(np.float32)
        log_peak = np.log(peak + 1e-8).astype(np.float32)
        # Zero-crossing rate on windowed (post pre-emphasis) signal
        signs = np.sign(pre)
        zcr = (np.abs(np.diff(signs, axis=-1)) > 0).sum(axis=-1).astype(np.float32) / (W - 1)
        # Delta of broadband log energy
        d_log_rms = _htk_delta(log_rms[:, None])[:, 0].astype(np.float32)
        dd_log_rms = _htk_delta(d_log_rms[:, None])[:, 0].astype(np.float32)

        # --- autocorrelation pitch + voicing -------------------------
        f0, pitch_conf, voiced = self._pitch(xw, W)
        log_f0 = np.log(f0 + 1.0).astype(np.float32)

        envelope = np.stack(
            [log_rms, d_log_rms, dd_log_rms, log_peak, zcr, log_f0, pitch_conf, voiced],
            axis=-1,
        ).astype(np.float32)

        # --- temporal modulation of log-mel subbands ------------------
        # Butterworth bandpass along time axis for each mel band.
        # Downsample mel bands by 2 (40 channels) to keep H within budget.
        lm_down = 0.5 * (log_mel[:, 0::2] + log_mel[:, 1::2])  # (T, n_mels//2)
        lm_t = np.ascontiguousarray(lm_down.astype(np.float64).T)  # (C, T)
        mod_blocks = []
        for sos in self.mod_sos:
            y = scipy.signal.sosfiltfilt(sos, lm_t, axis=-1)
            mod_blocks.append(y.T.astype(np.float32))
        mod_feats = np.concatenate(mod_blocks, axis=-1)  # (T, (n_mels/2) * n_rates)

        # --- spectral contrast (Jiang et al.) -------------------------
        # For each octave band, log(peak) - log(valley) over top/bottom alpha fraction.
        logM = np.log(M + 1e-10)
        sc_peak = []
        sc_valley = []
        sc_contrast = []
        for i in range(len(self.sc_edges) - 1):
            lo_hz, hi_hz = self.sc_edges[i], self.sc_edges[i + 1]
            band = (self.freqs >= lo_hz) & (self.freqs < hi_hz)
            if not band.any():
                continue
            sub = logM[:, band]
            sub_sorted = np.sort(sub, axis=-1)
            k = max(1, int(self.sc_alpha * sub.shape[-1]))
            valley = sub_sorted[:, :k].mean(axis=-1)
            peak = sub_sorted[:, -k:].mean(axis=-1)
            sc_peak.append(peak)
            sc_valley.append(valley)
            sc_contrast.append(peak - valley)
        spectral_contrast = np.stack(sc_peak + sc_valley + sc_contrast, axis=-1).astype(np.float32)

        # --- concatenate ---------------------------------------------
        feats = np.concatenate(
            [
                log_mel,       # 80
                d_log_mel,     # 80
                dd_log_mel,    # 80
                mfcc,          # 20
                d_mfcc,        # 20
                dd_mfcc,       # 20
                chroma,              # 12
                spectral,            # 8
                envelope,            # 8
                mod_feats,           # 40 * 5 = 200
                spectral_contrast,   # 7 * 3 = 21
            ],
            axis=-1,
        ).astype(np.float32)

        # Safety: enforce finiteness.
        if not np.all(np.isfinite(feats)):
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    # ------------------------------------------------------------------
    def _pitch(self, xw, W):
        """Autocorrelation pitch (f0) and voicing confidence per frame."""
        min_period = max(int(SR / self.pitch_fmax), 2)
        max_period = min(int(SR / self.pitch_fmin), W // 2)
        # Autocorrelation via Wiener-Khinchin
        nfft = 1 << (2 * W - 1).bit_length()
        X = scipy.fft.rfft(xw, n=nfft, axis=-1)
        ac = scipy.fft.irfft(X * np.conj(X), n=nfft, axis=-1).real[:, :max_period + 2]
        energy = ac[:, 0:1] + 1e-10
        ac_n = ac / energy
        # Search peak over valid lag range
        valid = ac_n[:, min_period : max_period + 1]
        peak_idx = np.argmax(valid, axis=-1)
        peak_val = np.take_along_axis(valid, peak_idx[:, None], axis=-1).squeeze(-1)
        period = (peak_idx + min_period).astype(np.float32)
        # Parabolic interpolation around the peak for sub-sample accuracy
        left = np.take_along_axis(ac_n, (peak_idx + min_period - 1)[:, None], axis=-1).squeeze(-1)
        right = np.take_along_axis(ac_n, (peak_idx + min_period + 1)[:, None], axis=-1).squeeze(-1)
        denom = (left - 2.0 * peak_val + right)
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        shift = 0.5 * (left - right) / denom
        shift = np.clip(shift, -1.0, 1.0)
        period = period + shift
        f0 = SR / np.maximum(period, 1.0)
        voiced = (peak_val > 0.3).astype(np.float32)
        return f0.astype(np.float32), peak_val.astype(np.float32), voiced
