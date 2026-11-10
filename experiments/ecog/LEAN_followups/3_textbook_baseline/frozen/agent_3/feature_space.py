"""A textbook interpretable acoustic feature space for an ECoG high-gamma
encoding model.  Closed-form signal processing only, with fixed parameters
drawn from standard practice in auditory neuroscience, speech science and
audio signal processing.

Feature groups (details in FEATURES.md):
  * Log-mel spectrogram (80 bands, Slaney normalisation) + delta + delta-delta
  * MFCC (20 coefficients) + delta + delta-delta
  * Fundamental frequency and voicing (autocorrelation-based)
  * Spectral shape descriptors (centroid, spread, skew, kurtosis, flatness,
    entropy, 50/85/95% rolloff, log-magnitude slope)
  * Spectral flux
  * Broadband energy / loudness (raw and A-weighted RMS, peak, crest factor)
  * Zero-crossing rate
  * Temporal modulation filterbank on log-mel bands (4 rates)
  * Spectral modulation energy along the mel axis (5 scales)
  * Broadband onset envelope
"""
import numpy as np


class FeatureSpace:
    win = 2048  # analysis window in samples (~46.4 ms at 44.1 kHz)

    # ---------- construction ----------
    def __init__(self, device='cpu'):
        self.device = device
        self.sr = 44100
        self.hop = 1470                           # fixed by harness, 30 Hz frames
        self.frame_rate = self.sr / self.hop      # 30 Hz

        self.n_fft = self.win
        self.n_bins = self.n_fft // 2 + 1

        # Mel filterbank parameters (Davis & Mermelstein 1980; Slaney 1998)
        self.n_mel = 80
        self.fmin = 20.0
        self.fmax = 8000.0

        # MFCC parameters (Davis & Mermelstein 1980)
        self.n_mfcc = 20

        # Autocorrelation pitch search range (adult speech / music F0)
        self.f0_min = 60.0
        self.f0_max = 500.0
        self.tau_min = int(self.sr / self.f0_max)   # 88 samples
        self.tau_max = int(self.sr / self.f0_min)   # 735 samples

        # Temporal modulation rates (Hz) - Chi, Ru & Shamma 2005
        self.mod_rates = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)

        # Spectral modulation scales (cycles per octave)
        self.mod_scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)

        # Pre-computed filters / matrices
        self.hann = np.hanning(self.win).astype(np.float32)
        self.mel_fb = self._build_mel_fb()
        self.dct = self._build_dct(self.n_mel, self.n_mfcc)
        self.freqs = np.arange(self.n_bins, dtype=np.float32) * (self.sr / self.n_fft)
        self.aweight = self._build_aweight()
        self.bands_per_oct = self.n_mel / np.log2(self.fmax / self.fmin)

    # ---------- builders ----------
    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_fb(self):
        mmin = self._hz_to_mel(self.fmin)
        mmax = self._hz_to_mel(self.fmax)
        mel_pts = np.linspace(mmin, mmax, self.n_mel + 2)
        hz_pts = self._mel_to_hz(mel_pts)
        bin_pts = hz_pts * (self.n_fft / self.sr)
        fb = np.zeros((self.n_mel, self.n_bins), dtype=np.float64)
        freqs = np.arange(self.n_bins, dtype=np.float64)
        for m in range(1, self.n_mel + 1):
            l, c, r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
            left = (freqs - l) / max(c - l, 1e-9)
            right = (r - freqs) / max(r - c, 1e-9)
            fb[m - 1] = np.maximum(0.0, np.minimum(left, right))
        enorm = 2.0 / (hz_pts[2:] - hz_pts[:-2])
        fb = (fb * enorm[:, None]).astype(np.float32)
        return fb

    @staticmethod
    def _build_dct(n_in, n_out):
        # Orthonormal DCT-II
        n = np.arange(n_in)
        k = np.arange(n_out)[:, None]
        d = np.cos(np.pi / n_in * (n + 0.5) * k)
        d[0] *= 1.0 / np.sqrt(2.0)
        d *= np.sqrt(2.0 / n_in)
        return d.astype(np.float32)

    def _build_aweight(self):
        # IEC 61672-1 A-weighting response (linear amplitude factor)
        f = np.maximum(self.freqs.astype(np.float64), 1e-6)
        ra = (12194.0 ** 2 * f ** 4) / (
            (f ** 2 + 20.6 ** 2)
            * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
            * (f ** 2 + 12194.0 ** 2)
        )
        A = 2.0 + 20.0 * np.log10(np.maximum(ra, 1e-20))
        return (10.0 ** (A / 20.0)).astype(np.float32)

    # ---------- helpers ----------
    @staticmethod
    def _delta(arr, N=2):
        # HTK-style delta (Furui 1986)
        T = arr.shape[0]
        pad = np.pad(arr, ((N, N), (0, 0)), mode='edge')
        out = np.zeros_like(arr, dtype=np.float32)
        denom = 2.0 * sum(n * n for n in range(1, N + 1))
        for n in range(1, N + 1):
            out += n * (pad[N + n: N + n + T] - pad[N - n: N - n + T])
        return (out / denom).astype(np.float32)

    def _pitch(self, x):
        # Autocorrelation-based F0 (Rabiner 1977; de Cheveigne & Kawahara 2002)
        T, W = x.shape
        n = 1
        while n < 2 * W:
            n *= 2
        X = np.fft.rfft(x, n=n, axis=1)
        acf = np.fft.irfft(X.real ** 2 + X.imag ** 2, n=n, axis=1)[:, :W]
        r0 = acf[:, :1]
        r0_safe = np.where(np.abs(r0) > 1e-12, r0, 1.0)
        acfn = acf / r0_safe
        seg = acfn[:, self.tau_min:self.tau_max]
        idx = seg.argmax(axis=1)
        peak = seg[np.arange(T), idx]
        lag = (self.tau_min + idx).astype(np.float32)
        f0 = (self.sr / np.maximum(lag, 1.0)).astype(np.float32)
        logf0 = np.log(np.maximum(f0, 1e-6)).astype(np.float32)
        voicing = np.clip(peak, 0.0, 1.0).astype(np.float32)
        # Harmonic-to-noise ratio in dB
        hnr = 10.0 * np.log10(
            np.clip(peak, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - peak, 1e-6, 1.0)
        ).astype(np.float32)
        voiced = (voicing > 0.3).astype(np.float32)
        f0v = (f0 * voiced).astype(np.float32)
        logf0v = (logf0 * voiced).astype(np.float32)
        return np.stack(
            [f0 / 100.0, logf0, voicing, hnr, f0v / 100.0, logf0v], axis=1
        ).astype(np.float32)

    def _spectral_shape(self, power):
        # Peeters 2004 spectral descriptors
        tot = power.sum(axis=1) + 1e-12
        pp = power / tot[:, None]
        f = self.freqs
        centroid = (pp * f[None]).sum(axis=1)
        var = (pp * (f[None] - centroid[:, None]) ** 2).sum(axis=1)
        spread = np.sqrt(np.maximum(var, 0.0))
        sdev = np.maximum(spread, 1e-6)
        skew = (pp * ((f[None] - centroid[:, None]) / sdev[:, None]) ** 3).sum(axis=1)
        kurt = (pp * ((f[None] - centroid[:, None]) / sdev[:, None]) ** 4).sum(axis=1)
        # Spectral flatness (Johnston 1988)
        logp = np.log(np.maximum(power, 1e-20))
        geo = np.exp(logp.mean(axis=1))
        arith = power.mean(axis=1) + 1e-20
        flatness = geo / arith
        entropy = -(pp * np.log(np.maximum(pp, 1e-20))).sum(axis=1)
        cs = np.cumsum(power, axis=1)
        tot_col = cs[:, -1:]
        tot_col_safe = np.where(tot_col > 1e-12, tot_col, 1.0)
        cum = cs / tot_col_safe
        r50 = f[(cum >= 0.5).argmax(axis=1)]
        r85 = f[(cum >= 0.85).argmax(axis=1)]
        r95 = f[(cum >= 0.95).argmax(axis=1)]
        # Spectral slope via OLS of log |X| vs log f (skip DC bin)
        logf = np.log(np.maximum(f[1:], 1e-6))
        logm = np.log(np.maximum(np.sqrt(power[:, 1:]), 1e-12))
        lf = logf - logf.mean()
        slope = (logm * lf[None]).sum(axis=1) / max((lf ** 2).sum(), 1e-12)
        return np.stack(
            [
                centroid / 1000.0,
                spread / 1000.0,
                skew,
                kurt,
                flatness,
                entropy,
                r50 / 1000.0,
                r85 / 1000.0,
                r95 / 1000.0,
                slope.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def _temporal_modulation(self, logmel):
        # Complex Morlet wavelet along the time axis of log-mel, per band
        # (Chi/Shamma 2005 cortical model, STM energy)
        T, M = logmel.shape
        N = 1
        while N < T + 32:
            N *= 2
        x = logmel - logmel.mean(axis=0, keepdims=True)
        X = np.fft.fft(x, n=N, axis=0)
        freqs = np.fft.fftfreq(N, d=1.0 / self.frame_rate)
        outs = []
        for fc in self.mod_rates:
            bw = max(float(fc) / 2.0, 0.25)
            W = np.exp(-0.5 * ((freqs - fc) / bw) ** 2) * 2.0
            W[freqs < 0] = 0.0
            W[freqs == 0] = 0.0
            Y = np.fft.ifft(X * W[:, None], n=N, axis=0)[:T]
            env = np.abs(Y).astype(np.float32)
            outs.append(env)
        return np.concatenate(outs, axis=1).astype(np.float32)

    def _spectral_modulation(self, logmel):
        # Modulation along the mel-frequency axis; Gaussian band at each scale
        T, M = logmel.shape
        N = 1
        while N < M + 8:
            N *= 2
        x = logmel - logmel.mean(axis=1, keepdims=True)
        X = np.fft.rfft(x, n=N, axis=1)
        mag = np.abs(X).astype(np.float32)
        cyc_per_band = np.arange(mag.shape[1], dtype=np.float32) / N
        cyc_per_oct = cyc_per_band * self.bands_per_oct
        feats = []
        for s in self.mod_scales:
            bw = max(float(s) / 2.0, 0.1)
            Wk = np.exp(-0.5 * ((cyc_per_oct - s) / bw) ** 2).astype(np.float32)
            feats.append((mag * Wk[None, :]).sum(axis=1))
        return np.stack(feats, axis=1).astype(np.float32)

    # ---------- main call ----------
    def __call__(self, windows):
        x = np.ascontiguousarray(windows, dtype=np.float32)
        T, W = x.shape

        # DC removal per frame
        x_dc = x - x.mean(axis=1, keepdims=True)

        # Pre-emphasis (standard speech preprocessing, alpha=0.97)
        pre = np.empty_like(x_dc)
        pre[:, 0] = x_dc[:, 0]
        pre[:, 1:] = x_dc[:, 1:] - 0.97 * x_dc[:, :-1]

        # Windowed FFT (Hann) on pre-emphasized signal
        xw = pre * self.hann
        X = np.fft.rfft(xw, axis=1)
        power = (X.real * X.real + X.imag * X.imag).astype(np.float32)

        # Log-mel spectrogram
        mel = power @ self.mel_fb.T
        logmel = np.log(mel + 1e-6).astype(np.float32)

        # MFCC (DCT-II of log-mel)
        mfcc = (logmel @ self.dct.T).astype(np.float32)

        # Deltas and delta-deltas
        dlogmel = self._delta(logmel)
        ddlogmel = self._delta(dlogmel)
        dmfcc = self._delta(mfcc)
        ddmfcc = self._delta(dmfcc)

        # Pitch / voicing - run on windowed DC-removed signal (no pre-emphasis)
        pitch = self._pitch(x_dc * self.hann)

        # Spectral shape descriptors
        shape = self._spectral_shape(power)

        # Spectral flux (positive differences across log-mel frames)
        flux = np.zeros((T, 1), dtype=np.float32)
        if T > 1:
            diff = np.maximum(np.diff(logmel, axis=0), 0.0)
            flux[1:, 0] = np.linalg.norm(diff, axis=1)

        # Energy / loudness
        rms = np.sqrt(np.mean(x_dc * x_dc, axis=1) + 1e-12).astype(np.float32)
        log_rms = np.log(rms + 1e-6).astype(np.float32)
        a_power = (power * (self.aweight ** 2)[None, :]).sum(axis=1)
        a_rms = np.sqrt(a_power / self.n_bins + 1e-12).astype(np.float32)
        log_a_rms = np.log(a_rms + 1e-6).astype(np.float32)
        peak = np.max(np.abs(x_dc), axis=1).astype(np.float32)
        crest = (peak / np.maximum(rms, 1e-12)).astype(np.float32)
        energy = np.stack([rms, log_rms, a_rms, log_a_rms, peak, crest], axis=1)

        # Zero-crossing rate
        sgn = np.sign(x_dc).astype(np.int8)
        zcr = (np.abs(np.diff(sgn, axis=1)) > 0).sum(axis=1).astype(np.float32) / max(W - 1, 1)
        zcr = zcr[:, None]

        # Temporal modulation (per-band modulation filterbank)
        tmod = self._temporal_modulation(logmel)

        # Spectral modulation (scales along mel axis, per frame)
        smod = self._spectral_modulation(logmel)

        # Broadband onset envelope (half-wave rectified log-RMS derivative)
        onset_env = np.zeros((T, 1), dtype=np.float32)
        if T > 1:
            de = np.diff(log_rms)
            onset_env[1:, 0] = np.maximum(de, 0.0)

        feats = np.concatenate(
            [
                logmel, dlogmel, ddlogmel,
                mfcc, dmfcc, ddmfcc,
                pitch,
                shape,
                flux,
                energy,
                zcr,
                tmod,
                smod,
                onset_env,
            ],
            axis=1,
        ).astype(np.float32)

        if not np.all(np.isfinite(feats)):
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feats
