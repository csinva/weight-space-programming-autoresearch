"""Interpretable acoustic feature space for predicting ECoG high-gamma (30 Hz frames).

All processing is closed-form signal processing with fixed parameters (no learning).
Within a segment the overlapping windows are stitched back into the continuous waveform,
which is then analysed at a finer time resolution (6 sub-frames per 30 Hz frame) and
pooled back to the frame grid. See FEATURES.md for a description of every group.
"""
import numpy as np

SR = 44100
HOP = 1470                 # samples between 30 Hz frames
NSUB = 6                   # fine sub-frames per frame
FHOP = HOP / NSUB          # 245 samples = 5.56 ms
FRATE = SR / FHOP          # 180 Hz fine frame rate
NCH = 64                   # ERB-spaced channels
FMIN, FMAX = 50.0, 15000.0
LOG_FLOOR = 1e-8           # power floor (re. unit-RMS segment), i.e. -80 dB
RATES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)          # temporal modulation, Hz
SCALES = (1 / 24, 1 / 12, 1 / 6, 1 / 3)           # spectral modulation, cycles/channel
NREG = 16                  # frequency regions for modulation pooling
F0_MIN, F0_MAX = 60.0, 480.0
EMA_TAUS = (0.25, 0.75, 2.5)                      # s, causal context time constants
LEAD = 2                   # frames of look-ahead for anticipation copies


def erb_num(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def erb_inv(e):
    return (10 ** (e / 21.4) - 1) / 0.00437


def _fbank(nfft, cfs_erb, spacing):
    f = np.fft.rfftfreq(nfft, 1 / SR)
    e = erb_num(f)
    W = np.maximum(0.0, 1 - np.abs(e[None, :] - cfs_erb[:, None]) / spacing)
    for c in range(len(cfs_erb)):
        if W[c].sum() == 0:
            W[c, np.argmin(np.abs(e - cfs_erb[c]))] = 1.0
    return W / W.sum(1, keepdims=True)


def _spectral_mats(n, scale):
    """Complex Gabor along the channel axis (reflect boundaries), zero DC response."""
    sig = 3.0 / (2 * np.pi * scale)
    K = int(np.ceil(3 * sig))
    k = np.arange(-K, K + 1)
    gauss = np.exp(-k ** 2 / (2 * sig ** 2))
    g = gauss * np.exp(2j * np.pi * scale * k)
    g = g - gauss * (g.sum() / gauss.sum())
    g = g / np.abs(g).sum() * 2
    M = np.zeros((n, n), complex)
    for c in range(n):
        idx = c + k
        idx = np.abs(idx)
        idx = np.where(idx > n - 1, 2 * (n - 1) - idx, idx)
        idx = np.clip(idx, 0, n - 1)
        np.add.at(M[c], idx, g)
    return M


def _smooth_mat(n, sig):
    k = np.arange(n)
    M = np.exp(-(k[:, None] - k[None, :]) ** 2 / (2 * sig ** 2))
    return M / M.sum(1, keepdims=True)


def _frame_pool(a, T):
    """(..., NSUB*T) -> (..., T) mean."""
    return a[..., :T * NSUB].reshape(*a.shape[:-1], T, NSUB).mean(-1)


def _ema(F, tau_frames):
    """Causal exponential moving average along axis 0."""
    from scipy.signal import lfilter
    a = np.exp(-1.0 / tau_frames)
    zi = (a * F[0])[None, :]
    out, _ = lfilter([1 - a], [1, -a], F, axis=0, zi=zi)
    return out


def _shift(F, s):
    """out[i] = F[i+s] with edge replication."""
    T = F.shape[0]
    idx = np.clip(np.arange(T) + s, 0, T - 1)
    return F[idx]


class FeatureSpace:
    win = 4096

    def __init__(self, device='cpu'):
        self.device = device
        e0, e1 = erb_num(FMIN), erb_num(FMAX)
        self.cf_erb = np.linspace(e0, e1, NCH)
        self.cfs = erb_inv(self.cf_erb)
        sp = self.cf_erb[1] - self.cf_erb[0]
        self.W_lo = _fbank(2048, self.cf_erb, sp)       # channels below 1 kHz
        self.W_hi = _fbank(1024, self.cf_erb, sp)       # channels at/above 1 kHz
        self.lo = self.cfs < 1000
        self.spec_mats = [_spectral_mats(NCH, s) for s in SCALES]
        self.smooth2 = _smooth_mat(NCH, 1.0)

    # ------------------------------------------------------------------ helpers
    def _stft_power(self, xp, off0, J, N):
        w = np.hanning(N + 1)[:-1].astype(np.float32)
        norm = (w ** 2).sum()
        starts = np.round(off0 - N / 2 + np.arange(J) * FHOP).astype(np.int64)
        out = np.empty((J, N // 2 + 1), np.float32)
        for a in range(0, J, 4096):
            b = min(J, a + 4096)
            idx = starts[a:b, None] + np.arange(N)[None, :]
            X = np.fft.rfft(xp[idx] * w, axis=1)
            out[a:b] = (X.real ** 2 + X.imag ** 2) / norm
        return out

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        windows = np.asarray(windows, dtype=np.float32)
        T, win = windows.shape
        # ---- stitch continuous waveform
        L_sig = (T - 1) * HOP + win
        x = np.zeros(L_sig, np.float32)
        for i in range(T):
            x[i * HOP:i * HOP + win] = windows[i]
        x = x.astype(np.float64)
        x -= x.mean()
        rms = np.sqrt(np.mean(x ** 2))
        x = x / (rms + 1e-12) if rms > 0 else x
        x = x.astype(np.float32)
        P = 4096
        xp = np.pad(x, P)
        center0 = P + win / 2.0                       # center of frame 0 in xp
        J = T * NSUB
        off0 = center0 - (NSUB / 2 - 0.5) * FHOP      # centre of fine frame 0

        # ---- ERB spectrogram at 180 Hz
        p_lo = self._stft_power(xp, off0, J, 2048)
        p_hi = self._stft_power(xp, off0, J, 1024)
        Pb = np.where(self.lo[:, None], self.W_lo @ p_lo.T, self.W_hi @ p_hi.T)  # (NCH, J)
        Pb = Pb.astype(np.float64)
        Ldb = 10 * np.log10(Pb + LOG_FLOOR)                                     # (NCH, J)

        feats = {}
        # 1. log spectrogram (64)
        spec = _frame_pool(Ldb, T).T
        feats['logspec'] = spec
        # 2. compressed (power-law) spectrogram, 32 bands
        pc = _frame_pool(Pb, T)
        pc = pc.reshape(32, 2, T).mean(1) ** (1 / 3)
        feats['powspec'] = pc.T

        # 3. onsets / offsets (32 bands) from spectrally smoothed log spectrogram
        Ls = (self.smooth2 @ Ldb)
        d = np.zeros_like(Ls)
        d[:, 2:] = (Ls[:, 2:] - Ls[:, :-2]) / 2
        Ls32 = d.reshape(32, 2, J).mean(1)
        on = _frame_pool(np.maximum(Ls32, 0), T).T * NSUB
        off = _frame_pool(np.maximum(-Ls32, 0), T).T * NSUB
        feats['onset'] = on
        feats['offset'] = off
        # broadband
        bb = 10 * np.log10(Pb.mean(0) + LOG_FLOOR)                   # broadband level (J,)
        dbb = np.zeros_like(bb)
        dbb[2:] = (bb[2:] - bb[:-2]) / 2
        flux = np.maximum(d, 0).mean(0)
        nflux = np.maximum(-d, 0).mean(0)
        bbf = bb[:T * NSUB].reshape(T, NSUB)
        feats['envelope'] = np.stack([
            bbf.mean(1), bbf.max(1), bbf.min(1), bbf.std(1),
            _frame_pool(np.maximum(dbb, 0), T), _frame_pool(np.maximum(-dbb, 0), T),
            _frame_pool(flux, T), _frame_pool(nflux, T),
            np.maximum(dbb[:T * NSUB].reshape(T, NSUB), 0).max(1),
        ], 1)
        # 4. sub-frame envelope: 8 broad bands x 6 sub-frames + broadband x 6
        b8 = 10 * np.log10(Pb.reshape(8, 8, J).mean(1) + LOG_FLOOR)
        sub = b8[:, :T * NSUB].reshape(8, T, NSUB).transpose(1, 0, 2).reshape(T, 48)
        feats['subframe_env'] = np.concatenate([sub, bbf], 1)

        # 5. spectro-temporal modulation energy
        feats['modulation'] = self._modulation(Ldb, T)

        # 6. spectral shape statistics
        feats['spectral_stats'] = self._spectral_stats(Pb, Ldb, x, T, win)

        # 7. pitch / periodicity
        pitch, core_pitch = self._pitch(xp, center0, T)
        feats['pitch'] = pitch

        # 8. temporal context of a compact core set
        core = np.concatenate([
            spec.reshape(T, 32, 2).mean(2),               # 32 log bands
            on.reshape(T, 16, 2).sum(2),                  # 16 onset bands
            feats['envelope'][:, [0, 4, 5, 6]],           # level, rise, fall, flux
            core_pitch,                                   # voicing + 12 rel-pitch bins
        ], 1)
        ctx = [_shift(core, LEAD)]
        for tau in EMA_TAUS:
            ctx.append(_ema(core, tau * SR / HOP))
        feats['context'] = np.concatenate(ctx, 1)

        F = np.concatenate([feats[k] for k in feats], 1)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        # guard against constant columns (e.g. silent segment): tiny deterministic dither
        sd = F.std(0)
        flat = sd <= 1e-6 * (np.abs(F).mean(0) + 1)
        if flat.any():
            rng = np.random.default_rng(0)
            F[:, flat] += 1e-3 * rng.standard_normal((T, int(flat.sum())))
        return F.astype(np.float32)

    # ------------------------------------------------------------------
    def _modulation(self, Ldb, T):
        J = Ldb.shape[1]
        Lc = Ldb - Ldb.mean(1, keepdims=True)
        maxsig = 3.0 / (2 * np.pi * min(RATES)) * FRATE
        nfft = int(2 ** np.ceil(np.log2(J + 6 * maxsig)))
        padl = (nfft - J) // 2
        Lpad = np.pad(Lc, ((0, 0), (padl, nfft - J - padl)), mode='reflect' if padl < J else 'constant')
        S = np.fft.fft(Lpad, axis=1)
        fr = np.fft.fftfreq(nfft, 1 / FRATE)
        reg = lambda a: a.reshape(NREG, NCH // NREG, -1).mean(1)          # (NREG, J)
        out = []
        # temporal low-pass (frame-scale) copy for pure spectral modulation
        lp = np.exp(-fr ** 2 / (2 * 4.0 ** 2))
        Llp = np.real(np.fft.ifft(S * lp, axis=1))[:, padl:padl + J]
        for M in self.spec_mats:
            out.append(reg(np.abs(M @ Llp)))                              # scale only
        for r in RATES:
            sf = r / 3.0
            H = np.exp(-(fr - r) ** 2 / (2 * sf ** 2))
            H[0] = 0
            Y = np.fft.ifft(S * H, axis=1)[:, padl:padl + J]               # analytic, (NCH, J)
            out.append(reg(np.abs(Y)))                                    # rate only
            for M in self.spec_mats:
                up = M @ Y                                                # spectral phase +
                dn = np.conj(M) @ Y
                out.append(reg(np.abs(up)))
                out.append(reg(np.abs(dn)))
        A = np.stack(out, 0)                                              # (K, NREG, J)
        A = _frame_pool(A, T)                                             # (K, NREG, T)
        A = np.log1p(A)
        return A.reshape(-1, T).T

    def _spectral_stats(self, Pb, Ldb, x, T, win):
        J = Pb.shape[1]
        p = Pb / (Pb.sum(0, keepdims=True) + 1e-20)
        e = self.cf_erb[:, None]
        cent = (p * e).sum(0)
        spread = np.sqrt((p * (e - cent) ** 2).sum(0))
        flat = np.exp(np.mean(np.log(Pb + LOG_FLOOR), 0)) / (Pb.mean(0) + LOG_FLOOR)
        ent = -(p * np.log(p + 1e-20)).sum(0) / np.log(NCH)
        cum = np.cumsum(p, 0)
        roll = (cum < 0.85).sum(0).astype(float)
        ec = e[:, 0] - e.mean()
        slope = (ec[:, None] * (Ldb - Ldb.mean(0))).sum(0) / (ec ** 2).sum()
        lo = 10 * np.log10(Pb[self.cfs < 500].mean(0) + LOG_FLOOR)
        mid = 10 * np.log10(Pb[(self.cfs >= 500) & (self.cfs < 4000)].mean(0) + LOG_FLOOR)
        hi = 10 * np.log10(Pb[self.cfs >= 4000].mean(0) + LOG_FLOOR)
        st = np.stack([cent, spread, np.log(flat + 1e-12), ent, roll, slope, mid - lo, hi - mid], 0)
        st = _frame_pool(st, T)
        # zero-crossing rate per frame (1470-sample block around each frame centre)
        c0 = win // 2 - HOP // 2
        blk = np.lib.stride_tricks.sliding_window_view(x, HOP)[c0 + np.arange(T) * HOP]
        zcr = (np.abs(np.diff(np.signbit(blk), axis=1)).sum(1) / HOP)
        return np.concatenate([st.T, np.log(zcr + 1e-4)[:, None]], 1)

    def _pitch(self, xp, center0, T):
        N, NF = 2048, 4096
        w = np.hanning(N + 1)[:-1].astype(np.float32)
        starts = np.round(center0 - N / 2 + np.arange(T) * HOP).astype(np.int64)
        f = np.fft.rfftfreq(NF, 1 / SR)
        band = ((f >= 50) & (f <= 4000)).astype(np.float32)
        lag_min = int(np.floor(SR / F0_MAX)) - 2
        lag_max = int(np.ceil(SR / F0_MIN)) + 2
        # window autocorrelation for bias correction
        wa = np.fft.irfft(np.abs(np.fft.rfft(w, NF)) ** 2, NF)[:lag_max + 1]
        wa = wa / wa[0]
        R = np.empty((T, lag_max + 1), np.float32)
        for a in range(0, T, 2048):
            b = min(T, a + 2048)
            idx = starts[a:b, None] + np.arange(N)[None, :]
            X = np.fft.rfft(xp[idx] * w, n=NF, axis=1)
            mag = (np.abs(X) ** 1.0) * band           # magnitude-spectrum (compressed) ACF
            r = np.fft.irfft(mag, NF, axis=1)[:, :lag_max + 1]
            R[a:b] = r / (r[:, :1] + 1e-12)
        R = R / np.maximum(wa, 0.2)[None, :]
        lags = np.arange(lag_max + 1)
        valid = (lags >= SR / F0_MAX) & (lags <= SR / F0_MIN)
        Rv = np.where(valid[None, :], R, -1.0)
        best = Rv.argmax(1)
        voicing = np.clip(Rv.max(1), 0, 1.5)
        # parabolic interpolation of the peak lag
        y0, y1, y2 = R[np.arange(T), best - 1], R[np.arange(T), best], R[np.arange(T), np.minimum(best + 1, lag_max)]
        den = (y0 - 2 * y1 + y2)
        delta = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1), 0)
        f0 = SR / np.maximum(best + np.clip(delta, -0.5, 0.5), 1)
        lf0 = np.log2(f0)
        # salience: 36 log-spaced F0 bins (12/oct, 60-480 Hz), max of normalised ACF in each bin
        edges = F0_MIN * 2 ** (np.arange(37) / 12)
        sal = np.zeros((T, 36))
        lagf = SR / np.maximum(lags, 1)
        for k in range(36):
            m = (lagf >= edges[k]) & (lagf < edges[k + 1])
            sal[:, k] = R[:, m].max(1) if m.any() else 0
        sal = np.clip(sal, -0.5, 1.5)
        # relative pitch: re. median log-F0 of clearly voiced frames in the segment
        vmask = voicing > 0.35
        ref = np.median(lf0[vmask]) if vmask.sum() >= 10 else np.log2(150.0)
        rel = lf0 - ref
        vw = np.clip((voicing - 0.2) / 0.25, 0, 1)                   # soft voicing weight
        cen = np.linspace(-1, 1, 12)
        relb = np.exp(-(rel[:, None] - cen[None, :]) ** 2 / (2 * (1 / 11) ** 2)) * vw[:, None]
        # pitch change between consecutive frames (voiced both), 6 bins (octaves per frame)
        dl = np.zeros(T)
        dl[1:] = lf0[1:] - lf0[:-1]
        vw2 = vw * np.concatenate([[0], vw[:-1]])
        dcen = np.array([-0.12, -0.06, -0.02, 0.02, 0.06, 0.12])
        db = np.exp(-(dl[:, None] - dcen[None, :]) ** 2 / (2 * 0.03 ** 2)) * vw2[:, None]
        summary = np.stack([voicing, vw, vw * (lf0 - np.log2(150.0)), vw * rel], 1)
        pitch = np.concatenate([summary, sal, relb, db], 1)
        core = np.concatenate([voicing[:, None], relb], 1)
        return pitch, core
