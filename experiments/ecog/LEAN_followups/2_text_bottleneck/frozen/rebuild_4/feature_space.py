"""Closed-form acoustic feature space (win = 4096, ~3.8k dims).

Groups (see FEATURES.md):
  A. cochleagrams (1024/2048/4096 sub-windows) + temporal residuals, tonic LPs, differences
  B. spectrotemporal modulation (2-D Gabor on 20-ch and 40-ch cochleagrams)
  C. spectral / pitch / voicing descriptors (+ dyn, + temporal-Gabor envelopes)
  D. interactions of key scalars (z-scored, expanded [z]+, [-z]+, z^3)
  E. within-window Morlet scattering + instantaneous frequency / coherence

No fitting or learning: all parameters are fixed constants.
"""
import numpy as np
from scipy.signal import lfilter, fftconvolve

SR = 44100.0
FPS = 30.0
EPS = 1e-10          # power floor (audio normalised to unit segment RMS)
DYN_TAUS = (4.5, 15.0, 60.0)


# ----------------------------------------------------------------------------- temporal helpers
def lp(x, tau):
    """Causal one-pole lowpass over frames (axis 0), alpha = 1/(1+tau), initialised at x[0]."""
    a = 1.0 / (1.0 + tau)
    x = np.asarray(x, np.float64)
    zi = (1.0 - a) * x[:1]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, axis=0, zi=zi)
    return y


def lp_acausal(x, tau):
    """Forward-backward one-pole lowpass (zero phase)."""
    return lp(lp(x, tau)[::-1], tau)[::-1]


def dyn(x):
    """[x - LP_tau for tau in (4.5, 15, 60)] concatenated."""
    return np.concatenate([x - lp(x, t) for t in DYN_TAUS], axis=1)


def with_dyn(x):
    return np.concatenate([x, dyn(x)], axis=1)


def shift(x, k):
    """y[t] = x[t-k] (k>0 past, k<0 future), edge-padded."""
    T = x.shape[0]
    idx = np.clip(np.arange(T) - k, 0, T - 1)
    return x[idx]


def zs(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def gabor_env_t(x, rate, sigma=None):
    """Magnitude of complex temporal Gabor (centre `rate` Hz) along frames, zero-mean kernel."""
    if sigma is None:
        sigma = max(1.0 / (np.pi * rate), 1.0 / FPS)       # seconds
    half = int(np.ceil(3 * sigma * FPS))
    t = np.arange(-half, half + 1) / FPS
    g = np.exp(-0.5 * (t / sigma) ** 2)
    k = g * np.exp(2j * np.pi * rate * t)
    k = k - g * (k.sum() / g.sum())
    k = k / np.abs(k).sum()
    xp = np.pad(x, ((half, half), (0, 0)), mode='reflect') if x.shape[0] > half else \
        np.pad(x, ((half, half), (0, 0)), mode='edge')
    y = fftconvolve(xp, k[:, None], mode='valid', axes=0)
    return np.abs(y)


# ----------------------------------------------------------------------------- filterbanks
def erb_space(lo, hi, n):
    e = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    ei = lambda E: (10 ** (E / 21.4) - 1) / 0.00437
    return ei(np.linspace(e(lo), e(hi), n))


def gammatone_weights(nfft, n_ch, lo=50.0, hi=8000.0):
    f = np.fft.rfftfreq(nfft, 1 / SR)
    fc = erb_space(lo, hi, n_ch)
    b = 1.019 * 24.7 * (4.37 * fc / 1000 + 1)
    W = (1 + ((f[None, :] - fc[:, None]) / b[:, None]) ** 2) ** -4   # 4th-order gammatone power
    W /= W.sum(1, keepdims=True)
    return W.T.astype(np.float64), fc


def gabor2d(rate, scale, n_ch_kernel_sigma=None):
    """Complex 2-D Gabor over (frames, channels). rate in Hz, scale in cycles/channel."""
    st = 1.0 / (np.pi * rate) if rate > 0 else 0.2
    st = max(st, 1.0 / FPS)
    sc = 1.0 / (np.pi * abs(scale)) if scale != 0 else 3.0
    sc = max(sc, 0.75)
    ht = int(np.ceil(3 * st * FPS))
    hc = int(np.ceil(3 * sc))
    t = np.arange(-ht, ht + 1)[:, None] / FPS
    c = np.arange(-hc, hc + 1)[None, :].astype(float)
    g = np.exp(-0.5 * (t / st) ** 2 - 0.5 * (c / sc) ** 2)
    k = g * np.exp(2j * np.pi * (rate * t + scale * c))
    k = k - g * (k.sum() / g.sum())
    k = k / np.abs(k).sum()
    return k


def gabor2d_apply(L, k):
    ht, hc = k.shape[0] // 2, k.shape[1] // 2
    mt = 'reflect' if L.shape[0] > ht else 'edge'
    Lp = np.pad(L, ((ht, ht), (0, 0)), mode=mt)
    Lp = np.pad(Lp, ((0, 0), (hc, hc)), mode='reflect' if L.shape[1] > hc else 'edge')
    return fftconvolve(Lp, k, mode='valid')


# ----------------------------------------------------------------------------- feature space
class FeatureSpace:
    win = 4096

    def __init__(self, device='cpu'):
        self.device = device
        self.sizes = (1024, 2048, 4096)
        self.hann = {n: np.hanning(n + 2)[1:-1] for n in self.sizes}
        self.W20 = {n: gammatone_weights(n, 20)[0] for n in self.sizes}
        self.W40, _ = gammatone_weights(4096, 40)
        self.f4096 = np.fft.rfftfreq(4096, 1 / SR)

        # 2-D Gabors on 20-ch cochleagram: 17 rate/scale pairs
        pairs = [(0.0, 0.125), (0.0, 0.25), (0.0, 0.5)]
        for r in (1.0, 3.0, 6.0, 12.0):
            pairs += [(r, 0.0), (r, 0.25), (r, -0.25)]
        pairs += [(4.0, 0.5), (4.0, -0.5)]
        self.g20 = [gabor2d(r, s) for r, s in pairs]
        # 40-ch cochleagram: 6 rates x 5 scales
        self.rates40 = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
        self.scales40 = (0.0, 0.25, -0.25, 0.5, -0.5)
        self.g40 = [[gabor2d(r, s) for s in self.scales40] for r in self.rates40]

        # pitch grid
        self.f0_fine = 75.0 * 2 ** (np.arange(63) / 24.0)
        self.f0_11 = 75.0 * 2 ** (np.linspace(0, np.log2(6.0), 11))

        # scattering filterbank: Q=4, 36 filters (9 octaves from 40 Hz)
        self.sc_fc = 40.0 * 2 ** (np.arange(36) / 4.0)
        self.sc_ch = np.arange(2, 36, 4)          # 9 channels (one per octave) for 2nd order
        self.mod_fc = 32.0 * (2000.0 / 32.0) ** (np.arange(8) / 7.0)

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        x = np.asarray(windows, np.float32)
        T = x.shape[0]
        if T == 0:
            return np.zeros((0, self.n_out()), np.float32)
        rms = float(np.sqrt(np.mean(x[:, 1313:2783].astype(np.float64) ** 2)))
        x = x / (rms if rms > 1e-12 else 1.0)

        feats = []
        # ---------------------------------------------------- spectra
        P = {}
        for n in self.sizes:
            s = (4096 - n) // 2
            X = np.fft.rfft(x[:, s:s + n] * self.hann[n].astype(np.float32), axis=1)
            P[n] = (X.real.astype(np.float64) ** 2 + X.imag.astype(np.float64) ** 2) / n
        P4 = P[4096]

        # ---------------------------------------------------- A. cochleagrams
        L = {n: np.log(P[n] @ self.W20[n] + EPS) for n in self.sizes}
        C = {n: np.cbrt(P[n] @ self.W20[n]) for n in self.sizes}
        Lall = np.concatenate([L[n] for n in self.sizes], 1)        # 60
        Call = np.concatenate([C[n] for n in self.sizes], 1)
        feats.append(Lall)
        for tau in (1.5, 4.5, 15.0, 60.0, 150.0):
            feats.append(Lall - lp(Lall, tau))
        for tau in (60.0, 150.0):
            feats.append(lp(Lall, tau))
            feats.append(lp(Call, tau))
        feats.append(lp_acausal(Lall, 60.0))
        for k in (3, 6, 12):
            for d in (Lall - shift(Lall, k), shift(Lall, -k) - Lall):
                feats += [d, np.maximum(d, 0), np.maximum(-d, 0)]

        # ---------------------------------------------------- B. spectrotemporal modulation
        L20 = L[4096]
        for k in self.g20:
            feats.append(np.abs(gabor2d_apply(L20, k)))                # 17 x 20
        L40 = np.log(P4 @ self.W40 + EPS)
        thirds = (slice(0, 13), slice(13, 27), slice(27, 40))
        cos_, sin_, mag_ = [], [], []
        for ri in range(len(self.rates40)):
            for si in range(len(self.scales40)):
                y = gabor2d_apply(L40, self.g40[ri][si])
                m = np.abs(y)
                for th in thirds:
                    cos_.append(y.real[:, th].mean(1))
                    sin_.append(y.imag[:, th].mean(1))
                    mag_.append(m[:, th].mean(1))
        cos_, sin_, mag_ = (np.stack(v, 1) for v in (cos_, sin_, mag_))   # (T, 90) each, order r,s,third
        feats += [cos_, sin_, mag_, dyn(mag_), mag_ - lp_acausal(mag_, 60.0)]
        M = mag_.reshape(T, len(self.rates40), len(self.scales40), 3)
        feats.append(np.sqrt(M[:, :-1] * M[:, 1:]).reshape(T, -1))       # cross-rate products, 75

        # ---------------------------------------------------- C. descriptors
        D = self.descriptors(x, P4, L20, L40, P)
        scal, gated, inter = D
        feats.append(with_dyn(np.concatenate([scal, gated], 1)))
        zsc = zs(scal)
        for r in (2.0, 4.0, 6.0, 10.0):
            feats.append(gabor_env_t(zsc, r))

        # ---------------------------------------------------- D. interactions
        feats.append(self.interactions(inter))

        # ---------------------------------------------------- E. scattering
        S1, S2, IFC = self.scattering(x)
        feats += [with_dyn(S1), S2, with_dyn(IFC)]

        out = np.concatenate([np.asarray(f, np.float64).reshape(T, -1) for f in feats], 1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1e6, 1e6).astype(np.float32)
        # keep every column non-constant (fixed deterministic jitter on degenerate columns)
        sd = out.std(0)
        bad = sd < 1e-7
        if bad.any():
            rng = np.random.default_rng(0)
            out[:, bad] += (1e-6 * rng.standard_normal((T, int(bad.sum())))).astype(np.float32)
        return out

    def n_out(self):
        return 3769

    # ------------------------------------------------------------------ descriptors
    def descriptors(self, x, P4, L20, L40, P):
        T = P4.shape[0]
        f = self.f4096
        df = f[1]
        LP4 = np.log(P4 + EPS)

        # --- pitch salience: 8-harmonic log comb on max-smoothed log spectrum
        Ls = np.maximum(np.maximum(LP4, np.roll(LP4, 1, 1)), np.roll(LP4, -1, 1))
        h = np.arange(1, 9)
        w = 1.0 / np.sqrt(h)
        w = w / w.sum()

        def interp(bins):
            b0 = np.floor(bins).astype(int)
            fr = bins - b0
            return Ls[:, b0] * (1 - fr) + Ls[:, b0 + 1] * fr

        sal = np.zeros((T, len(self.f0_fine)))
        for i, f0 in enumerate(self.f0_fine):
            hb = h * f0 / df
            on = interp(hb)
            off = 0.5 * (interp((h - 0.5) * f0 / df) + interp((h + 0.5) * f0 / df))
            sal[:, i] = (on - off) @ w
        sal11 = np.stack([sal[:, max(0, j - 3):j + 4].max(1)
                          for j in np.round(np.log2(self.f0_11 / 75.0) * 24).astype(int)], 1)
        bi = sal.argmax(1)
        strength = sal[np.arange(T), bi]
        bc = np.clip(bi, 1, len(self.f0_fine) - 2)
        ym, y0, yp = sal[np.arange(T), bc - 1], sal[np.arange(T), bc], sal[np.arange(T), bc + 1]
        den = ym - 2 * y0 + yp
        off = np.where(np.abs(den) > 1e-9, 0.5 * (ym - yp) / np.where(np.abs(den) > 1e-9, den, 1), 0)
        logf0 = np.log2(75.0) + (bc + np.clip(off, -1, 1)) / 24.0

        # --- ACF HNR (Boersma: divide by window autocorrelation)
        hw = self.hann[4096]
        Xa = np.fft.rfft(x.astype(np.float64) * hw, n=8192, axis=1)
        r = np.fft.irfft(np.abs(Xa) ** 2, axis=1)[:, :600]
        rw = np.fft.irfft(np.abs(np.fft.rfft(hw, 8192)) ** 2)[:600]
        rn = (r / rw) / (r[:, :1] / rw[0] + EPS)
        rmax = np.clip(rn[:, 98:589].max(1), 1e-3, 0.999)
        hnr = 10 * np.log10(rmax / (1 - rmax))
        gate = np.clip((rmax - 0.4) / 0.4, 0, 1)

        # --- cepstral peak prominence
        Ldb = 10 * np.log10(P4 + EPS)
        cep = np.fft.irfft(Ldb, axis=1)
        q = np.arange(22, 2048)
        A = np.stack([np.ones_like(q, dtype=float), q.astype(float)], 1)
        coef = cep[:, q] @ np.linalg.pinv(A).T                   # fixed design matrix
        qi = 98 + cep[:, 98:589].argmax(1)
        cpk = cep[np.arange(T), qi]
        cpp = cpk - (coef[:, 0] + coef[:, 1] * qi)

        # --- spectral shape
        band = (f >= 50) & (f <= 11025)
        fb = f[band]
        pb = P4[:, band] + EPS
        pn = pb / pb.sum(1, keepdims=True)
        mu = pn @ fb
        var = pn @ (fb ** 2) - mu ** 2
        sd = np.sqrt(np.maximum(var, 1.0))
        dev = (fb[None, :] - mu[:, None]) / sd[:, None]
        skew = (pn * dev ** 3).sum(1)
        kurt = (pn * dev ** 4).sum(1)
        b8 = (f >= 50) & (f <= 8000)
        p8 = P4[:, b8] + EPS
        flat = np.log(p8).mean(1) - np.log(p8.mean(1))
        cs = np.cumsum(pb, 1)
        roll = fb[np.minimum((cs < 0.85 * cs[:, -1:]).sum(1), len(fb) - 1)]
        loud = np.log(p8.sum(1))
        shape = np.stack([np.log(mu), np.log(sd), np.arcsinh(skew), np.log(kurt + 1e-3),
                          flat, np.log(roll), loud], 1)

        # --- octave spectral contrast (8 octave bands 62.5 Hz-16 kHz)
        oc = []
        for k in range(8):
            m = (f >= 62.5 * 2 ** k) & (f < 62.5 * 2 ** (k + 1))
            s = np.sort(LP4[:, m], 1)
            n = max(1, int(round(0.2 * s.shape[1])))
            oc.append(s[:, -n:].mean(1) - s[:, :n].mean(1))
        # --- channel-group contrast (8 groups of 5 channels on 40-ch cochleagram)
        G = L40.reshape(T, 8, 5)
        gc = G.max(2) - G.min(2)
        contrast = np.concatenate([np.stack(oc, 1), gc], 1)

        # --- softmax formants on cepstrally smoothed envelope
        cn = np.fft.irfft(LP4, axis=1)
        cn[:, 30:-29] = 0
        Lenv = np.fft.rfft(cn, axis=1).real
        form = []
        for lo, hi in ((250, 1000), (800, 2800), (1800, 3600)):
            m = (f >= lo) & (f <= hi)
            z = 1.5 * Lenv[:, m]
            zmax = z.max(1, keepdims=True)
            e = np.exp(z - zmax)
            wts = e / e.sum(1, keepdims=True)
            form.append(np.log(wts @ f[m]))
            form.append((zmax[:, 0] + np.log(e.mean(1))) / 1.5 - Lenv[:, b8].mean(1))
        form = np.stack(form, 1)

        # --- spectral flux (frame-to-frame, 20-ch cochleagram)
        d = L20 - shift(L20, 1)
        flux = np.stack([np.maximum(d, 0).mean(1), np.abs(d).mean(1)], 1)

        scal = np.concatenate([sal11, logf0[:, None], strength[:, None], hnr[:, None], cpp[:, None],
                               gate[:, None], shape, contrast, form, flux], 1)          # 47
        gated = gate[:, None] * np.log1p((P4 @ self.W20[4096]) / EPS)                     # 20

        # --- phonetic band contrasts for interactions
        def bandlog(lo, hi):
            m = (f >= lo) & (f < hi)
            return np.log(P4[:, m].sum(1) + EPS)
        b_low, b_f1, b_f2, b_fr = bandlog(50, 300), bandlog(300, 1000), bandlog(1000, 3000), bandlog(3000, 8000)
        bc4 = np.stack([b_fr - b_f1, b_f2 - b_f1, b_low - b_f1, b_fr - np.logaddexp(b_f1, b_f2)], 1)
        voiced_f0 = gate * (logf0 - np.log2(150.0))
        other = np.stack([strength, voiced_f0, loud, flux[:, 0], form[:, 0], form[:, 2]], 1)
        key_rest = np.stack([hnr, cpp, np.log(mu), flat], 1)
        return scal, gated, (bc4, other, key_rest)

    # ------------------------------------------------------------------ interactions
    def interactions(self, inter):
        bc4, other, key_rest = inter
        key = zs(np.concatenate([bc4, other, key_rest], 1))          # 14
        prods = (zs(bc4)[:, :, None] * zs(other)[:, None, :]).reshape(len(key), -1)   # 24
        z = zs(np.concatenate([key, dyn(key), prods], 1))             # 80
        return np.concatenate([np.maximum(z, 0), np.maximum(-z, 0), z ** 3], 1)   # 240

    # ------------------------------------------------------------------ scattering
    def scattering(self, x):
        T, N = x.shape
        df = SR / N
        hw = self.hann[4096]
        S1 = np.zeros((T, len(self.sc_fc)))
        S2 = np.zeros((T, len(self.sc_ch), len(self.mod_fc)))
        IFC = np.zeros((T, len(self.sc_ch), 2))
        nb = N // 2 + 1
        ch_set = {int(c): j for j, c in enumerate(self.sc_ch)}
        for t0 in range(0, T, 256):
            xc = x[t0:t0 + 256].astype(np.float64)
            X = np.fft.rfft(xc, axis=1)
            for k, fc in enumerate(self.sc_fc):
                sig = fc / (4 * 2.355)                 # Q = 4 (FWHM = fc/4)
                second = k in ch_set
                Msub = 1024 if second else int(max(16, 2 ** np.ceil(np.log2(8 * sig / df + 1))))
                cb = int(round(fc / df))
                bins = np.arange(cb - Msub // 2, cb + Msub // 2)
                valid = (bins >= 0) & (bins < nb)
                psi = np.exp(-0.5 * ((bins * df - fc) / sig) ** 2) * valid
                Z = np.zeros((xc.shape[0], Msub), complex)
                Z[:, (bins - cb) % Msub] = X[:, np.clip(bins, 0, nb - 1)] * psi
                z = np.fft.ifft(Z, axis=1) * (Msub / N)
                U = np.abs(z)
                wsub = np.interp(np.arange(Msub) * N / Msub, np.arange(N), hw)
                wsub = wsub / wsub.sum()
                s1 = U @ wsub
                S1[t0:t0 + 256, k] = np.log(s1 + 1e-8)
                if second:
                    j = ch_set[k]
                    fs_sub = SR * Msub / N
                    Uc = (U - s1[:, None]) * (wsub * Msub)
                    UF = np.fft.fft(Uc, axis=1)
                    fm = np.fft.fftfreq(Msub, 1 / fs_sub)
                    for m, mf in enumerate(self.mod_fc):
                        ms = mf / 2.355            # Q = 1
                        phi = np.exp(-0.5 * ((fm - mf) / ms) ** 2) * (fm > 0)
                        v = np.abs(np.fft.ifft(UF * phi, axis=1)).mean(1)
                        S2[t0:t0 + 256, j, m] = np.log((v + 1e-8) / (s1 + 1e-8))
                    zz = z[:, 1:] * np.conj(z[:, :-1]) * wsub[1:]
                    sz = zz.sum(1)
                    ifo = np.angle(sz) * fs_sub / (2 * np.pi) + cb * df - fc
                    IFC[t0:t0 + 256, j, 0] = ifo / sig
                    coh = np.abs(sz) / ((np.abs(zz)).sum(1) + 1e-12)
                    IFC[t0:t0 + 256, j, 1] = -np.log(np.maximum(1.0 - coh, 0) + 1e-9)
        return S1, S2.reshape(T, -1), IFC.reshape(T, -1)
