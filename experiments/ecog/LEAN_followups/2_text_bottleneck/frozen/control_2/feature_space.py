"""Interpretable acoustic feature space for predicting ECoG high-gamma (30 Hz frames).

All parameters are fixed; everything is closed-form signal processing (no fitting).
See FEATURES.md for a description of every group.
"""
import numpy as np
from scipy.signal import lfilter

SR = 44100
HOP = 1470                 # samples per 30 Hz frame
NSUB = 6                   # sub-frames per frame
SUB = HOP // NSUB          # 245 samples = 5.56 ms (180 Hz sub-frame rate)
SUB_SR = SR / SUB
EPS = 1e-6                 # fixed power floor before log compression


def _erb_number(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def _erb_inv(e):
    return (10 ** (e / 21.4) - 1) / 0.00437


def _erb_filterbank(n_bands, fmin, fmax, nfft, sr):
    """Triangular filters with centres equally spaced on the ERB-rate scale."""
    edges = _erb_inv(np.linspace(_erb_number(fmin), _erb_number(fmax), n_bands + 2))
    freqs = np.arange(nfft // 2 + 1) * sr / nfft
    df = sr / nfft
    fb = np.zeros((n_bands, freqs.size))
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        lo, hi = min(lo, c - 1.5 * df), max(hi, c + 1.5 * df)   # at least ~3 bins wide
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[b] = np.clip(np.minimum(up, down), 0, None)
        fb[b] /= fb[b].sum()
    return fb.astype(np.float32), edges[1:-1]


def _shift(x, s):
    """y[t] = x[t + s] with edge replication (s > 0 looks ahead, s < 0 looks back)."""
    idx = np.clip(np.arange(x.shape[0]) + s, 0, x.shape[0] - 1)
    return x[idx]


def _pool_frames(x, T, how='mean'):
    x = x[:T * NSUB].reshape(T, NSUB, *x.shape[1:])
    return x.mean(1) if how == 'mean' else (x.sum(1) if how == 'sum' else x.max(1))


class FeatureSpace:
    win = 4096

    # spectrogram
    N_BANDS = 64
    FMIN, FMAX = 80.0, 12000.0
    SPEC_WIN = 1323            # 30 ms Hann
    SPEC_NFFT = 2048
    # pitch
    PITCH_WIN = 2048           # 46 ms Hann
    PITCH_NFFT = 4096
    F0_MIN, F0_MAX, N_PITCH = 60.0, 500.0, 24
    # spectro-temporal modulation (Chi, Ru & Shamma 2005 style)
    RATES = (2.0, 4.0, 8.0, 16.0, 32.0)          # Hz
    SCALES = (0.25, 0.5, 1.0, 2.0)               # cycles / octave
    CH_PER_OCT = 10.0                            # approx. ERB channels per octave above 1 kHz
    STM_FPOOL = 4                                # channels pooled -> 16 frequency regions
    # slow context
    TAUS = (0.2, 0.5, 1.5)                       # s, causal exponential integration
    # frame shifts (feature at frame t + s); harness adds 0..3 frame delays + 100 ms lag
    CORE_SHIFTS = (2, -2)                        # -> effective audio lags 33-267 ms
    COARSE_SHIFT = -6                            # -> effective lags 300-400 ms
    STM_SHIFT = 2

    def __init__(self, device='cpu'):
        self.device = device
        self.fb, self.fc = _erb_filterbank(self.N_BANDS, self.FMIN, self.FMAX, self.SPEC_NFFT, SR)
        self.spec_window = np.hanning(self.SPEC_WIN).astype(np.float32)
        self.pitch_window = np.hanning(self.PITCH_WIN).astype(np.float32)
        wac = np.fft.irfft(np.abs(np.fft.rfft(self.pitch_window, self.PITCH_NFFT)) ** 2)
        self.lag_min = int(np.floor(SR / self.F0_MAX))
        self.lag_max = int(np.ceil(SR / self.F0_MIN))
        lags = np.arange(self.lag_min, self.lag_max + 1)
        self.wac = (wac[lags] / wac[0]).astype(np.float32)
        bin_edges = np.geomspace(self.F0_MIN, self.F0_MAX, self.N_PITCH + 1)
        f_of_lag = SR / lags
        self.pitch_bin = np.clip(np.searchsorted(bin_edges, f_of_lag) - 1, 0, self.N_PITCH - 1)
        self.lags = lags

    # ------------------------------------------------------------------ helpers
    def _signal(self, windows):
        """Rebuild the continuous waveform from the overlapping frame windows."""
        T, win = windows.shape
        a = win // 2 - HOP // 2
        x = np.concatenate([windows[0, :a], windows[:, a:a + HOP].reshape(-1), windows[-1, a + HOP:]])
        return x.astype(np.float32), win // 2  # frame i centre at win//2 + i*HOP

    def _subframes(self, x, c0, n, length):
        """(n, length) matrix of windows centred on c0 + j*SUB (zero padded at edges)."""
        pad = length
        xp = np.pad(x, (pad, pad))
        starts = c0 + np.arange(n) * SUB - length // 2 + pad
        sw = np.lib.stride_tricks.sliding_window_view(xp, length)
        return sw[starts]

    def _spectrogram(self, x, c0, n):
        out = np.empty((n, self.N_BANDS), np.float32)
        for s in range(0, n, 4096):
            fr = self._subframes(x, c0 + s * SUB, min(4096, n - s), self.SPEC_WIN) * self.spec_window
            pw = np.abs(np.fft.rfft(fr, self.SPEC_NFFT, axis=1)) ** 2 / self.SPEC_WIN
            out[s:s + fr.shape[0]] = pw.astype(np.float32) @ self.fb.T
        return out

    def _nacf(self, x, c0, n):
        out = np.empty((n, self.lags.size), np.float32)
        for s in range(0, n, 2048):
            fr = self._subframes(x, c0 + s * SUB, min(2048, n - s), self.PITCH_WIN)
            fr = (fr - fr.mean(1, keepdims=True)) * self.pitch_window
            r = np.fft.irfft(np.abs(np.fft.rfft(fr, self.PITCH_NFFT, axis=1)) ** 2, axis=1)
            r0 = r[:, :1] + 1e-3
            out[s:s + fr.shape[0]] = (r[:, self.lags] / r0) / self.wac
        return out

    def _stm(self, L):
        """Directional spectro-temporal modulation energy |S * h_rate,scale|."""
        n, F = L.shape
        nt = int(2 ** np.ceil(np.log2(n + int(4 * SUB_SR / min(self.RATES)) + 1)))
        nf = 2 * F
        Sf = np.fft.fft2(L, s=(nt, nf))
        wt = np.fft.fftfreq(nt)
        wf = np.fft.fftfreq(nf)
        pos_t = (wt > 0).astype(np.float32) * 2
        t = np.arange(nt) / SUB_SR
        ch = np.concatenate([np.arange(F), np.arange(F - nf, 0)]) / self.CH_PER_OCT   # octaves
        out = []
        for r in self.RATES:
            h = (r * t) ** 2 * np.exp(-3.5 * r * t) * np.sin(2 * np.pi * r * t)
            h /= np.abs(h).sum()
            Ht = np.fft.fft(h) * pos_t                      # analytic (positive temporal freqs)
            for sc in self.SCALES:
                g = (np.pi * sc * ch) ** 2
                hs = (1 - 2 * g) * np.exp(-g)
                Hs = np.fft.fft(hs).real
                for direction in (1, -1):
                    Hsd = Hs * (direction * wf > 0) * 2
                    R = np.fft.ifft2(Sf * Ht[:, None] * Hsd[None, :])[:n, :F]
                    A = np.abs(R).astype(np.float32)
                    out.append(A.reshape(n, F // self.STM_FPOOL, self.STM_FPOOL).mean(2))
        return np.stack(out, 1)  # (n, n_filters, n_regions)

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        windows = np.asarray(windows, dtype=np.float32)
        T = windows.shape[0]
        x, ctr0 = self._signal(windows)
        n = T * NSUB
        c0 = ctr0 - (HOP // 2 - SUB // 2)            # first sub-frame centre of frame 0

        # --- auditory spectrogram (sub-frame rate) ---
        P = self._spectrogram(x, c0, n)                       # (n, 64) band power
        L = np.log10(P + EPS)
        Pb = P.sum(1)
        Lb = np.log10(Pb + EPS)

        mel = np.log10(_pool_frames(P, T) + EPS)             # (T, 64)
        D = L - np.concatenate([L[:2], L[:-2]])               # 11 ms log-energy change
        onset = _pool_frames(np.maximum(D, 0), T, 'sum')      # (T, 64)
        offset = _pool_frames(np.maximum(-D, 0), T, 'sum').reshape(T, 32, 2).mean(2)

        # --- broadband descriptors ---
        loud = np.log10(_pool_frames(Pb, T) + EPS)
        Db = Lb - np.concatenate([Lb[:2], Lb[:-2]])
        bb_on = _pool_frames(np.maximum(Db, 0), T, 'sum')
        bb_off = _pool_frames(np.maximum(-Db, 0), T, 'sum')
        Lbs = np.convolve(np.pad(Lb, 2, mode='edge'), np.ones(5) / 5, 'valid')
        dLbs = np.diff(Lbs, prepend=Lbs[0]) * SUB_SR
        peak_rate = _pool_frames(np.maximum(dLbs, 0), T, 'max')
        loud_range = _pool_frames(Lb, T, 'max') - _pool_frames(Lb, T, 'mean')
        Pf = _pool_frames(P, T) + EPS
        w = Pf / Pf.sum(1, keepdims=True)
        chan = np.arange(self.N_BANDS, dtype=np.float32)
        centroid = w @ chan
        spread = np.sqrt(np.maximum(w @ chan ** 2 - centroid ** 2, 0))
        flatness = np.log(Pf).mean(1) - np.log(Pf.mean(1))
        hf_ratio = np.log10(Pf[:, 40:].sum(1)) - np.log10(Pf[:, :40].sum(1))   # > ~2.3 kHz vs below
        broadband = np.stack([loud, bb_on, bb_off, peak_rate, loud_range,
                              centroid, spread, flatness, hf_ratio], 1)

        # --- pitch / periodicity ---
        N = self._nacf(x, c0, n)                              # (n, lags)
        harm = np.clip(N.max(1), 0, 1)
        f0 = SR / self.lags[N.argmax(1)]
        pb = np.zeros((n, self.N_PITCH), np.float32)
        for b in range(self.N_PITCH):
            pb[:, b] = np.clip(N[:, self.pitch_bin == b].max(1), 0, 1)
        pitch_bins = _pool_frames(pb, T)
        harmonicity = _pool_frames(harm, T)
        logf0 = _pool_frames(harm * np.log2(f0 / 150.0), T) / (harmonicity + 1e-3)
        voiced_logf0 = harmonicity * logf0
        df0 = np.diff(logf0, prepend=logf0[0]) * np.minimum(harmonicity, np.r_[harmonicity[0], harmonicity[:-1]])
        pitch = np.concatenate([pitch_bins, np.stack([harmonicity, voiced_logf0, df0], 1)], 1)

        core = np.concatenate([mel, onset, offset, broadband, pitch], 1)

        # --- coarse long-lag copy ---
        coarse = np.concatenate([mel.reshape(T, 16, 4).mean(2), onset.reshape(T, 16, 4).sum(2),
                                 loud[:, None], bb_on[:, None]], 1)

        # --- spectro-temporal modulations ---
        stm = _pool_frames(self._stm(L), T).reshape(T, -1)

        # --- slow context (causal leaky integration of coarse spectrum + loudness) ---
        ctx_in = np.concatenate([np.log10(_pool_frames(P, T).reshape(T, 16, 4).sum(2) + EPS),
                                 loud[:, None]], 1)
        ctx = []
        for tau in self.TAUS:
            a = np.exp(-1.0 / (tau * 30.0))
            zi = ctx_in[0] * a
            ctx.append(lfilter([1 - a], [1, -a], ctx_in, axis=0, zi=zi[None, :])[0])
        ctx = np.concatenate(ctx, 1)

        feats = [_shift(core, s) for s in self.CORE_SHIFTS]
        feats += [_shift(coarse, self.COARSE_SHIFT), _shift(stm, self.STM_SHIFT), _shift(ctx, self.STM_SHIFT)]
        out = np.concatenate(feats, 1).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
