"""Textbook interpretable acoustic feature space for an ECoG high-gamma encoding model.

All features are closed-form signal processing with fixed parameters; no learning / fitting.
See FEATURES.md for a description of each group and the literature it comes from.
"""
import numpy as np
from scipy.signal import fftconvolve

SR = 44100
HOP = 1470  # 30 Hz frame rate


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


class FeatureSpace:
    win = 2048  # ~46.4 ms analysis window at 44.1 kHz

    def __init__(self, device='cpu'):
        self.device = device
        self.sr = SR
        self.n_fft = self.win
        # Hann window, standard for STFT
        self.window = np.hanning(self.win).astype(np.float32)
        self.preemph = 0.97  # standard first-order pre-emphasis

        # Mel filterbank (Slaney-style, 64 bands from 20 Hz to 16 kHz)
        self.n_mels = 64
        self.f_min = 20.0
        self.f_max = 16000.0
        self.mel_fb = self._build_mel_fb(self.n_mels, self.n_fft, self.sr, self.f_min, self.f_max)

        # DCT-II matrix for MFCCs (standard orthonormal, keep 13 coefficients incl. C0)
        self.n_mfcc = 13
        self.dct_mat = self._build_dct(self.n_mfcc, self.n_mels)

        # FFT frequency axis
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr).astype(np.float32)

        # A-weighting (IEC 61672-1) power weights
        self.a_weights = self._a_weighting(self.freqs)

        # Chroma filterbank (Fujishima 1999): bin each FFT frequency to one of 12 pitch classes
        self.chroma_fb = self._build_chroma_fb(self.freqs)

        # Spectral contrast octave bands (Jiang et al. 2002)
        self.contrast_bands = [
            (0.0, 200.0),
            (200.0, 400.0),
            (400.0, 800.0),
            (800.0, 1600.0),
            (1600.0, 3200.0),
            (3200.0, 6400.0),
            (6400.0, 20000.0),
        ]

        # Temporal modulation filters applied along the frame axis (30 Hz sampling)
        self.mod_sr = self.sr / HOP  # 30 Hz
        self.mod_rates = [1.0, 2.0, 4.0, 8.0]  # Hz (Chi, Ru & Shamma 2005)
        self.mod_filters = self._build_mod_filters(self.mod_rates, self.mod_sr)

        # Pool 64 mel bands into 8 wider bands for modulation features (manageable dim)
        self.n_mod_bands = 8

    # -------- filterbanks / matrices ----------
    def _build_mel_fb(self, n_mels, n_fft, sr, fmin, fmax):
        mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
        hz = _mel_to_hz(mels)
        bins = np.floor((n_fft + 1) * hz / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            if c <= l:
                c = l + 1
            if r <= c:
                r = c + 1
            for k in range(l, c):
                fb[m - 1, k] = (k - l) / max(c - l, 1)
            for k in range(c, r):
                fb[m - 1, k] = (r - k) / max(r - c, 1)
        # Slaney normalization: unit area under each filter
        enorm = 2.0 / (hz[2:n_mels + 2] - hz[:n_mels] + 1e-10)
        fb *= enorm[:, None]
        return fb

    def _build_dct(self, n_mfcc, n_mels):
        k = np.arange(n_mfcc)[:, None]
        n = np.arange(n_mels)[None, :]
        dct = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
        dct[0] *= 1.0 / np.sqrt(2.0)
        dct *= np.sqrt(2.0 / n_mels)
        return dct.astype(np.float32)

    def _a_weighting(self, f):
        f2 = f.astype(np.float64) ** 2
        num = (12194.0 ** 2) * (f2 ** 2)
        den = (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2)
        ra = num / np.maximum(den, 1e-30)
        # Reference so that gain at 1 kHz == 1 (standard calibration, ~ +2 dB offset).
        ref_num = (12194.0 ** 2) * (1000.0 ** 4)
        ref_den = (1000.0 ** 2 + 20.6 ** 2) * np.sqrt((1000.0 ** 2 + 107.7 ** 2) * (1000.0 ** 2 + 737.9 ** 2)) * (1000.0 ** 2 + 12194.0 ** 2)
        ra = ra / (ref_num / ref_den)
        return ra.astype(np.float32)

    def _build_chroma_fb(self, freqs):
        n_chroma = 12
        fb = np.zeros((n_chroma, len(freqs)), dtype=np.float32)
        # Each FFT bin contributes to one pitch class (A4 = 440 Hz reference)
        for k, f in enumerate(freqs):
            if f < 50.0 or f > 5000.0:
                continue
            midi = 12.0 * np.log2(f / 440.0) + 69.0
            cls = int(round(midi)) % 12
            fb[cls, k] = 1.0
        return fb

    def _build_mod_filters(self, rates, sr):
        """Hann-windowed complex Gabor filters (cos, sin) for each temporal modulation rate.

        Envelope of the convolution is sqrt(cos_out^2 + sin_out^2); i.e., the amplitude of
        modulation at the target rate. 4-cycle Hann window per filter.
        """
        pairs = []
        for rate in rates:
            L = max(int(4 * sr / rate), 15)
            if L % 2 == 0:
                L += 1
            n = np.arange(L) - L // 2
            win = np.hanning(L).astype(np.float32)
            g_cos = (win * np.cos(2 * np.pi * rate * n / sr)).astype(np.float32)
            g_sin = (win * np.sin(2 * np.pi * rate * n / sr)).astype(np.float32)
            norm = np.sqrt((g_cos ** 2).sum() + (g_sin ** 2).sum()) + 1e-10
            pairs.append((g_cos / norm, g_sin / norm))
        return pairs

    # -------- per-frame helpers ----------
    @staticmethod
    def _delta(x, M=2):
        """Furui (1986) regression-coefficient delta along the time axis."""
        T = x.shape[0]
        denom = 2.0 * sum(m * m for m in range(1, M + 1))
        pad = np.pad(x, ((M, M), (0, 0)), mode='edge')
        d = np.zeros_like(x)
        for m in range(1, M + 1):
            d += m * (pad[M + m:M + m + T] - pad[M - m:M - m + T])
        return (d / denom).astype(np.float32)

    @staticmethod
    def _peak_rate(log_mel):
        """Half-wave rectified first temporal derivative (Oganian & Chang 2019)."""
        d = np.empty_like(log_mel)
        d[0] = 0.0
        d[1:] = log_mel[1:] - log_mel[:-1]
        return np.maximum(d, 0.0).astype(np.float32)

    def _spectral_descriptors(self, power):
        T, K = power.shape
        f = self.freqs
        p_sum = power.sum(axis=1) + 1e-10
        pn = power / p_sum[:, None]

        centroid = (pn * f).sum(axis=1)
        var = ((f - centroid[:, None]) ** 2 * pn).sum(axis=1)
        spread = np.sqrt(np.maximum(var, 0.0))
        spread_safe = np.maximum(spread, 1.0)
        skew = ((f - centroid[:, None]) ** 3 * pn).sum(axis=1) / (spread_safe ** 3)
        kurt = ((f - centroid[:, None]) ** 4 * pn).sum(axis=1) / (spread_safe ** 4)

        log_p = np.log(power + 1e-10)
        flat = np.exp(log_p.mean(axis=1)) / (power.mean(axis=1) + 1e-10)

        cum = np.cumsum(power, axis=1) / p_sum[:, None]
        ro85 = f[np.argmax(cum >= 0.85, axis=1)]
        ro95 = f[np.argmax(cum >= 0.95, axis=1)]

        f_mean = f.mean()
        f_var = ((f - f_mean) ** 2).sum() + 1e-10
        log_p_c = log_p - log_p.mean(axis=1, keepdims=True)
        slope = (log_p_c * (f - f_mean)).sum(axis=1) / f_var

        flux = np.zeros(T, dtype=np.float32)
        sqp = np.sqrt(power)
        flux[1:] = np.sqrt(((sqp[1:] - sqp[:-1]) ** 2).sum(axis=1))

        entropy = -(pn * np.log(pn + 1e-10)).sum(axis=1)

        return np.stack([
            centroid.astype(np.float32),
            spread.astype(np.float32),
            skew.astype(np.float32),
            kurt.astype(np.float32),
            flat.astype(np.float32),
            ro85.astype(np.float32),
            ro95.astype(np.float32),
            slope.astype(np.float32),
            flux.astype(np.float32),
            entropy.astype(np.float32),
        ], axis=1)

    def _spectral_contrast(self, power):
        T = power.shape[0]
        f = self.freqs
        results = []
        alpha = 0.02
        for low, high in self.contrast_bands:
            mask = (f >= low) & (f < high)
            if not mask.any():
                results.append(np.zeros(T, dtype=np.float32))
                continue
            p = power[:, mask]
            p_sorted = np.sort(p, axis=1)
            n = p.shape[1]
            k = max(1, int(np.floor(alpha * n)))
            peaks = np.log(p_sorted[:, -k:].mean(axis=1) + 1e-10)
            valleys = np.log(p_sorted[:, :k].mean(axis=1) + 1e-10)
            results.append((peaks - valleys).astype(np.float32))
        return np.stack(results, axis=1)

    @staticmethod
    def _zcr(x):
        sign = np.signbit(x).astype(np.int8)
        zc = np.abs(np.diff(sign.astype(np.int16), axis=1))
        return zc.mean(axis=1, keepdims=True).astype(np.float32)

    def _pitch(self, x):
        """Autocorrelation-based F0 (classical method; Rabiner 1977 / Boersma 1993 variant).

        Returns [f0_hz, log_f0, voicing, hnr] per frame.
        """
        T, W = x.shape
        sr = self.sr
        f0_min, f0_max = 50.0, 500.0
        lag_min = max(2, int(sr / f0_max))
        lag_max = min(W - 1, int(sr / f0_min))

        xc = x - x.mean(axis=1, keepdims=True)
        nfft = 1
        while nfft < 2 * W:
            nfft <<= 1
        X = np.fft.rfft(xc, n=nfft, axis=1)
        r = np.fft.irfft(X * np.conj(X), n=nfft, axis=1)[:, :W]
        r0 = r[:, 0:1] + 1e-12
        rn = r / r0

        seg = rn[:, lag_min:lag_max + 1]
        peak_idx = np.argmax(seg, axis=1)
        peak_val = seg[np.arange(T), peak_idx].astype(np.float32)
        lag = (peak_idx + lag_min).astype(np.float32)
        f0 = (sr / np.maximum(lag, 1.0)).astype(np.float32)
        log_f0 = np.log(np.maximum(f0, 1.0)).astype(np.float32)
        v_clip = np.clip(peak_val, 1e-4, 1.0 - 1e-4)
        hnr = 10.0 * np.log10(v_clip / (1.0 - v_clip))
        return np.stack([f0, log_f0, peak_val, hnr.astype(np.float32)], axis=1)

    def _modulation_features(self, log_mel):
        """Temporal modulation energy at several rates, pooled over mel-band groups.

        Low-level analogue of Chi/Ru/Shamma (2005) STM analysis along the time axis.
        """
        T, Nm = log_mel.shape
        # Pool mel bands into n_mod_bands groups by averaging
        step = Nm // self.n_mod_bands
        pooled = log_mel[:, :step * self.n_mod_bands].reshape(T, self.n_mod_bands, step).mean(axis=2)
        # Zero-mean each band before modulation analysis
        pooled = pooled - pooled.mean(axis=0, keepdims=True)

        outs = []
        for g_cos, g_sin in self.mod_filters:
            c = fftconvolve(pooled, g_cos[:, None], mode='same', axes=0)
            s = fftconvolve(pooled, g_sin[:, None], mode='same', axes=0)
            env = np.sqrt(c * c + s * s)
            outs.append(env.astype(np.float32))
        return np.concatenate(outs, axis=1)

    # -------- main ----------
    def __call__(self, windows):
        x = np.asarray(windows, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("expected (T, win)")
        T, W = x.shape

        # Pre-emphasis (Rabiner & Juang 1993)
        xp = np.empty_like(x)
        xp[:, 0] = x[:, 0]
        xp[:, 1:] = x[:, 1:] - self.preemph * x[:, :-1]
        xw = xp * self.window

        # STFT magnitude / power
        X = np.fft.rfft(xw, n=self.n_fft, axis=1)
        mag = np.abs(X).astype(np.float32)
        power = (mag * mag).astype(np.float32)

        # Log-mel cochleogram
        mel = power @ self.mel_fb.T
        log_mel = np.log(mel + 1e-6).astype(np.float32)

        # Temporal derivatives on log-mel
        d_mel = self._delta(log_mel, M=2)
        pr_mel = self._peak_rate(log_mel)

        # MFCCs + deltas
        mfcc = (log_mel @ self.dct_mat.T).astype(np.float32)
        d_mfcc = self._delta(mfcc, M=2)
        dd_mfcc = self._delta(d_mfcc, M=2)

        # Spectral descriptors / contrast
        spec = self._spectral_descriptors(power)
        contrast = self._spectral_contrast(power)

        # Zero-crossing rate (on raw, windowed audio)
        zcr = self._zcr(x)

        # Energy / loudness
        rms_lin = np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-20).astype(np.float32)
        rms_log = (20.0 * np.log10(rms_lin + 1e-6)).astype(np.float32)
        a_power = (power * self.a_weights[None, :]).sum(axis=1, keepdims=True)
        loudness_a = (10.0 * np.log10(a_power + 1e-10)).astype(np.float32)

        # Pitch / voicing
        pitch = self._pitch(x)

        # Chroma (log)
        chroma = (power @ self.chroma_fb.T).astype(np.float32)
        chroma = np.log(chroma + 1e-6).astype(np.float32)

        # Broadband envelope peak rate
        bb_env = np.log(power.sum(axis=1) + 1e-6).astype(np.float32)
        bb_pr = np.empty_like(bb_env)
        bb_pr[0] = 0.0
        bb_pr[1:] = np.maximum(bb_env[1:] - bb_env[:-1], 0.0)
        bb_pr = bb_pr[:, None]

        # Temporal modulation features
        mod_feats = self._modulation_features(log_mel)

        feats = np.concatenate([
            log_mel,          # 64
            d_mel,            # 64
            pr_mel,           # 64
            mfcc,             # 13
            d_mfcc,           # 13
            dd_mfcc,          # 13
            spec,             # 10
            contrast,         # 7
            zcr,              # 1
            rms_lin,          # 1
            rms_log,          # 1
            loudness_a,       # 1
            pitch,            # 4
            chroma,           # 12
            bb_pr,            # 1
            mod_feats,        # 32
        ], axis=1).astype(np.float32)

        if not np.all(np.isfinite(feats)):
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feats
