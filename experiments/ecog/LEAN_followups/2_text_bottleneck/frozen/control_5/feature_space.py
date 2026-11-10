"""Interpretable, closed-form acoustic feature space for predicting ECoG high-gamma (30 Hz frames).

All parameters are fixed; nothing is learned or estimated except per-segment normalisation constants
(level floor, pitch reference) computed in closed form from the segment itself. See FEATURES.md.
"""
import numpy as np

SR = 44100
FPS = 30.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_fb(n_bands, fmin, fmax, nfft, sr=SR):
    """Triangular mel filterbank (n_bands, nfft//2+1), each filter at least ~2 FFT bins wide, area-normalised."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_bands + 2))
    df = sr / nfft
    fb = np.zeros((n_bands, len(freqs)))
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        lo, hi = min(lo, c - df), max(hi, c + df)
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        fb[b] = np.clip(np.minimum(up, dn), 0, None)
        fb[b] /= fb[b].sum()
    return fb, edges[1:-1]


def _pool_matrix(n_in, n_out):
    """Average adjacent bands: (n_out, n_in) with overlapping-free contiguous groups."""
    P = np.zeros((n_out, n_in))
    bounds = np.linspace(0, n_in, n_out + 1).round().astype(int)
    for i in range(n_out):
        P[i, bounds[i]:bounds[i + 1]] = 1.0 / (bounds[i + 1] - bounds[i])
    return P


def _ema(x, tau_s):
    """Causal exponential moving average along axis 0 with time constant tau_s (seconds), started at x[0]."""
    from scipy.signal import lfilter, lfilter_zi
    a = np.exp(-1.0 / (tau_s * FPS))
    x2 = x.reshape(len(x), -1)
    zi = lfilter_zi([1 - a], [1, -a])[:, None] * x2[:1]
    y, _ = lfilter([1 - a], [1, -a], x2, axis=0, zi=zi)
    return y.reshape(x.shape)


def _lag(x, k=1):
    """x delayed by k frames (first frames repeat x[0])."""
    return np.concatenate([np.repeat(x[:1], k, axis=0), x[:-k]], axis=0)


class FeatureSpace:
    win = 4096

    # main spectrogram
    N_MEL = 64
    FMIN, FMAX = 60.0, 10000.0
    MAIN_WIN = 2048            # Hann analysis window (46 ms), centred in the 4096-sample window
    # fine-time analysis
    SUB_WIN, SUB_HOP, N_SUB = 512, 245, 7   # 7 sub-frames (11.6 ms) spanning ~the 33 ms hop
    N_FINE = 16
    # coarse bands for flux / context / modulation
    N_COARSE = 16
    N_MOD_REG = 8
    CONTEXT_TAUS = (0.2, 0.6, 2.0)
    # pitch
    F0_MIN, F0_MAX = 70.0, 400.0
    N_F0_ABS, N_F0_REL, N_F0_DELTA = 12, 8, 5
    # spectro-temporal modulation (causal gammatone-envelope Gabors on the log-mel spectrogram)
    MOD_RATES = (2.0, 4.0, 8.0)             # Hz
    MOD_SCALES = (0.5, 1.0, 2.0)            # cycles / octave

    def __init__(self, device='cpu'):
        self.device = device
        self.fb_main, self.cf_main = _mel_fb(self.N_MEL, self.FMIN, self.FMAX, self.MAIN_WIN)
        self.fb_fine, _ = _mel_fb(self.N_FINE, 100.0, self.FMAX, self.SUB_WIN)
        self.pool_coarse = _pool_matrix(self.N_MEL, self.N_COARSE)
        self.pool_mod = _pool_matrix(self.N_MEL, self.N_MOD_REG)
        self.hann_main = np.hanning(self.MAIN_WIN + 2)[1:-1]
        self.hann_sub = np.hanning(self.SUB_WIN + 2)[1:-1]
        self.freqs_main = np.fft.rfftfreq(self.MAIN_WIN, 1.0 / SR)
        # A-weighting (power) on the main grid for a loudness proxy
        f2 = self.freqs_main ** 2
        ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
        self.aweight = (ra / ra.max()) ** 2
        # pitch: autocorrelation of the Hann window for bias correction
        nfft = 2 * self.MAIN_WIN
        w_acf = np.fft.irfft(np.abs(np.fft.rfft(self.hann_main, nfft)) ** 2, nfft)[:self.MAIN_WIN]
        self.win_acf = w_acf / w_acf[0]
        self.lag_lo = int(np.floor(SR / self.F0_MAX))
        self.lag_hi = int(np.ceil(SR / self.F0_MIN))
        self.oct_per_band = np.log2(self.FMAX / self.FMIN) / self.N_MEL  # approx. (mel is not exactly log)
        self._build_mod_kernels()

    # ------------------------------------------------------------------ helpers
    def _build_mod_kernels(self):
        """Causal temporal kernels (gammatone envelope, order 3) and spectral Gabor kernels."""
        self.t_kernels = []
        for r in self.MOD_RATES:
            n_cyc_len = 1.5 / r                         # ~1.5 cycles of support
            L = int(np.ceil(n_cyc_len * FPS * 2)) + 1
            t = np.arange(L) / FPS
            b = r / 1.5                                  # bandwidth ~ r/1.5
            env = t ** 2 * np.exp(-2 * np.pi * b * t)
            env /= env.sum()
            self.t_kernels.append(env * np.exp(2j * np.pi * r * t))
        # spectral axis, in bands; cf of mel bands is approximately log-spaced above ~700 Hz
        bands_per_oct = 1.0 / self.oct_per_band
        self.f_kernels = []
        for s in self.MOD_SCALES:
            sig = 0.5 / s * bands_per_oct                # Gaussian sd = half a period
            half = int(np.ceil(2.5 * sig))
            k = np.arange(-half, half + 1)
            g = np.exp(-0.5 * (k / sig) ** 2)
            g /= g.sum()
            self.f_kernels.append(g * np.exp(2j * np.pi * s * k / bands_per_oct))

    @staticmethod
    def _conv_time(x, k):
        """Causal convolution along axis 0 of real x (T, F) with complex kernel k (L,)."""
        from scipy.signal import lfilter
        return lfilter(k, [1.0], x, axis=0)

    @staticmethod
    def _conv_freq(x, k):
        from scipy.ndimage import convolve1d
        if np.iscomplexobj(x):
            return convolve1d(x.real, k.real, axis=1, mode='nearest') - convolve1d(x.imag, k.imag, axis=1, mode='nearest') \
                + 1j * (convolve1d(x.real, k.imag, axis=1, mode='nearest') + convolve1d(x.imag, k.real, axis=1, mode='nearest'))
        return convolve1d(x, k.real, axis=1, mode='nearest') + 1j * convolve1d(x, k.imag, axis=1, mode='nearest')

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        x = np.asarray(windows, dtype=np.float64)
        T, W = x.shape
        c = W // 2
        feats = []

        # ---- 1. main power spectrum and log-mel spectrogram
        seg = x[:, c - self.MAIN_WIN // 2: c + self.MAIN_WIN // 2]
        seg = seg - seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg * self.hann_main, axis=1)
        P = X.real ** 2 + X.imag ** 2                                   # (T, 1025)
        mel = P @ self.fb_main.T                                         # (T, 64)
        floor = 1e-5 * mel.mean() + 1e-20                                # ~50 dB below segment mean level
        lmel = np.log(mel + floor)
        feats.append(lmel)                                               # 64

        # ---- 2. loudness / level
        tot = P.sum(axis=1)
        tot_floor = 1e-5 * tot.mean() + 1e-20
        log_rms = np.log(tot + tot_floor)
        aw = P @ self.aweight
        log_a = np.log(aw + 1e-5 * aw.mean() + 1e-20)
        feats.append(np.stack([log_rms, log_a], 1))                      # 2

        # ---- 3. spectral flux (positive log-mel change), coarse bands + broadband + peakRate
        lco = lmel @ self.pool_coarse.T                                  # (T, 16)
        dco = lco - _lag(lco)
        flux_pos = np.maximum(dco, 0)
        flux_neg = np.maximum(-dco, 0)
        bb_env = _ema(log_a, 0.03)                                        # light smoothing
        peak_rate = np.maximum(bb_env - _lag(bb_env), 0)
        feats.append(flux_pos)                                           # 16
        feats.append(np.stack([flux_pos.mean(1), flux_neg.mean(1), peak_rate], 1))  # 3

        # ---- 4. fine-time (11.6 ms) envelope and onsets within the frame hop
        starts = c - (self.N_SUB * self.SUB_HOP) // 2 - self.SUB_WIN // 2 + self.SUB_HOP // 2 \
            + np.arange(self.N_SUB) * self.SUB_HOP
        sub = np.stack([x[:, s:s + self.SUB_WIN] for s in starts], 1)    # (T, 7, 512)
        sub = sub - sub.mean(axis=2, keepdims=True)
        S = np.fft.rfft(sub * self.hann_sub, axis=2)
        Ps = (S.real ** 2 + S.imag ** 2) @ self.fb_fine.T                 # (T, 7, 16)
        lfs = np.log(Ps + 1e-5 * Ps.mean() + 1e-20)
        fine_on = np.maximum(np.diff(lfs, axis=1), 0).max(axis=1)        # (T, 16) max within-frame onset
        fine_env = lfs.mean(axis=2)                                      # (T, 7) broadband envelope
        feats.append(fine_on)                                            # 16
        feats.append(fine_env)                                           # 7
        feats.append(np.maximum(np.diff(fine_env, axis=1), 0).max(1, keepdims=True))  # 1

        # ---- 5. slow context (causal EMAs of coarse log-mel) -> adaptation / longer history
        for tau in self.CONTEXT_TAUS:
            feats.append(_ema(lco, tau))                                 # 3 x 16
        feats.append(_ema(flux_pos.mean(1, keepdims=True), 0.5))         # 1 (recent onset density)

        # ---- 6. spectral shape
        Pn = P / (tot[:, None] + tot_floor)
        f = self.freqs_main
        cent = Pn @ f
        spread = np.sqrt(np.maximum(Pn @ f ** 2 - cent ** 2, 0))
        mn = mel + floor
        flat = np.mean(np.log(mn), 1) - np.log(np.mean(mn, 1))           # spectral flatness (log)
        mp = mn / mn.sum(1, keepdims=True)
        ent = -(mp * np.log(mp)).sum(1)
        cP = np.cumsum(Pn, 1)
        roll = f[np.minimum((cP < 0.85).sum(1), len(f) - 1)]
        feats.append(np.stack([np.log(cent + 50), np.log(spread + 50), flat, ent, np.log(roll + 50)], 1))  # 5

        # ---- 7. pitch / periodicity (compressed-spectrum autocorrelation)
        nfft = 2 * self.MAIN_WIN
        Xp = np.fft.rfft(seg * self.hann_main, nfft, axis=1)
        A = np.abs(Xp) ** 1.0                                            # generalised ACF, exponent 0.5 on power
        fr = np.fft.rfftfreq(nfft, 1.0 / SR)
        A[:, (fr < 50) | (fr > 5000)] = 0
        r = np.fft.irfft(A ** 2, nfft, axis=1)[:, :self.lag_hi + 2]
        r = r / (r[:, :1] + 1e-12 * (r[:, 0].mean() + 1e-30))
        r = r / self.win_acf[None, :self.lag_hi + 2]
        rr = r[:, self.lag_lo:self.lag_hi + 1]
        k = rr.argmax(1)
        kk = np.clip(k, 1, rr.shape[1] - 2)
        y0, y1, y2 = rr[np.arange(T), kk - 1], rr[np.arange(T), kk], rr[np.arange(T), kk + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1), 0)
        lagf = self.lag_lo + kk + np.clip(off, -1, 1)
        voic = np.clip(rr.max(1), 0, 1)
        # soft voicing: sigmoid around 0.5 so unvoiced frames are ~0 but never exactly constant
        v = 1.0 / (1.0 + np.exp(-(voic - 0.5) / 0.07))
        log_f0 = np.log2(SR / lagf)
        # absolute pitch
        ctr = np.linspace(np.log2(self.F0_MIN), np.log2(self.F0_MAX), self.N_F0_ABS)
        sd = ctr[1] - ctr[0]
        f0_abs = np.exp(-0.5 * ((log_f0[:, None] - ctr[None]) / sd) ** 2) * v[:, None]
        # relative pitch (segment-level reference over confidently voiced frames)
        wv = v ** 2
        mu = (wv * log_f0).sum() / (wv.sum() + 1e-9) if wv.sum() > 1 else np.log2(150.0)
        sg = np.sqrt((wv * (log_f0 - mu) ** 2).sum() / (wv.sum() + 1e-9)) if wv.sum() > 1 else 0.3
        sg = max(sg, 0.1)
        z = (log_f0 - mu) / sg
        ctr_r = np.linspace(-2, 2, self.N_F0_REL)
        f0_rel = np.exp(-0.5 * ((z[:, None] - ctr_r[None]) / (ctr_r[1] - ctr_r[0])) ** 2) * v[:, None]
        # pitch change (octaves per frame), only meaningful when consecutive frames voiced
        dlf = log_f0 - _lag(log_f0)
        vv = v * _lag(v)
        ctr_d = np.array([-0.1, -0.04, 0.0, 0.04, 0.1])
        f0_d = np.exp(-0.5 * ((dlf[:, None] - ctr_d[None]) / 0.03) ** 2) * vv[:, None]
        feats += [f0_abs, f0_rel, f0_d, np.stack([voic, v], 1)]          # 12 + 8 + 5 + 2

        # ---- 8. spectro-temporal modulation energy (causal, on 64-band log-mel)
        Z = lmel - lmel.mean(0, keepdims=True)
        mods = []
        tf = [self._conv_time(Z, k) for k in self.t_kernels]               # complex (T, 64) per rate
        for i in range(len(self.MOD_RATES)):
            mods.append(np.abs(tf[i]))                                     # pure temporal rate
            for fk in self.f_kernels:
                up = self._conv_freq(tf[i], fk)                            # downward/upward sweeps
                dn = self._conv_freq(tf[i], np.conj(fk))
                mods.append(np.abs(up))
                mods.append(np.abs(dn))
        for fk in self.f_kernels:                                          # pure spectral scale (rate 0)
            sm = _ema(Z, 0.1)
            mods.append(np.abs(self._conv_freq(sm, fk)))
        for m in mods:                                                     # pool to 8 frequency regions
            feats.append(np.log(m @ self.pool_mod.T + 1e-3))               # 24 x 8 = 192

        out = np.concatenate(feats, axis=1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # guard against exactly constant columns (would break per-segment z-scoring)
        const = out.std(0) < 1e-7
        if const.any():
            out[:, const] += 1e-4 * np.sin(np.arange(T) * 0.7)[:, None]
        return out.astype(np.float32)
