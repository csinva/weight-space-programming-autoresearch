"""Closed-form acoustic feature space (win=4096, 30 Hz frames).

Groups (see FEATURES.md): cochleagrams, spectrotemporal modulation, spectral/voice
descriptors, z-scored interactions, within-window Morlet scattering.
No learned or fitted parameters: every filter / constant is fixed in advance.
Within-segment operations (causal/acausal lowpasses over frames, z-scoring,
relative log floors) use only the segment being processed.
"""
import numpy as np
import torch
from scipy.signal import lfilter, fftconvolve

SR = 44100
FPS = 30.0
DYN_TAUS = (4.5, 15.0, 60.0)


# ----------------------------------------------------------------------------- helpers
def erb_space(lo, hi, n):
    e = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    ie = lambda x: (10 ** (x / 21.4) - 1) / 0.00437
    return ie(np.linspace(e(lo), e(hi), n))


def gammatone_weights(nfft, n_ch, lo=50.0, hi=8000.0):
    """Power response of 4th-order gammatone filters sampled at rfft bins, rows sum to 1."""
    f = np.fft.rfftfreq(nfft, 1 / SR)
    fc = erb_space(lo, hi, n_ch)
    b = 1.019 * (24.7 + 0.108 * fc)
    W = (1 + ((f[None] - fc[:, None]) / b[:, None]) ** 2) ** -4
    W /= W.sum(1, keepdims=True)
    return W.astype(np.float32), fc


def lp(x, tau):
    """Causal one-pole lowpass over frames, alpha = 1/(1+tau), initialised at x[0]."""
    a = 1.0 / (1.0 + tau)
    zi = (1 - a) * x[:1]
    y, _ = lfilter([a], [1, -(1 - a)], x, axis=0, zi=zi)
    return y


def lp_acausal(x, tau):
    return lp(lp(x, tau)[::-1], tau)[::-1]


def dyn(x):
    """[x, x-LP4.5, x-LP15, x-LP60]"""
    return np.concatenate([x] + [x - lp(x, t) for t in DYN_TAUS], 1)


def dyn_res(x):
    return np.concatenate([x - lp(x, t) for t in DYN_TAUS], 1)


def zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def shift(x, k):
    """y[t] = x[t-k] (k>0 past, k<0 future) with edge replication."""
    T = len(x)
    idx = np.clip(np.arange(T) - k, 0, T - 1)
    return x[idx]


def slog(e):
    """log with a floor relative to the segment's mean energy (scale invariant)."""
    return np.log(e + 1e-6 * e.mean() + 1e-30)


def gabor_1d(freq, sigma, zero_mean=True):
    """Complex Gabor; freq in cycles/sample, sigma in samples."""
    h = int(np.ceil(3 * sigma))
    n = np.arange(-h, h + 1)
    env = np.exp(-0.5 * (n / sigma) ** 2)
    g = env * np.exp(2j * np.pi * freq * n)
    if zero_mean and freq != 0:
        g = g - env * (g.sum() / env.sum())
    return g / env.sum()


def tconv(x, k):
    """Centred convolution along time (axis 0) with reflect padding. x (T,D) real, k complex."""
    h = (len(k) - 1) // 2
    T = len(x)
    if h >= T:
        xp = np.pad(x, ((h, h), (0, 0)), mode='edge')
    else:
        xp = np.pad(x, ((h, h), (0, 0)), mode='reflect')
    y = fftconvolve(xp, k[:, None], mode='valid', axes=0)
    return y[:T]


def chan_matrix(n_ch, scale, sigma):
    """Channel-axis Gabor as a (n_ch, n_ch) matrix (same-size, truncated edges)."""
    g = gabor_1d(scale, sigma, zero_mean=True)
    h = (len(g) - 1) // 2
    M = np.zeros((n_ch, n_ch), complex)
    for i in range(n_ch):
        for j in range(n_ch):
            d = j - i
            if -h <= d <= h:
                M[i, j] = g[d + h]
    return M


def tgabor(x, rate_hz, cycles=0.5):
    """Complex temporal Gabor at rate_hz on frame series x (T,D)."""
    if rate_hz == 0:
        return tconv(x, gabor_1d(0, 3.0).astype(complex))
    sigma = FPS * cycles / rate_hz
    return tconv(x, gabor_1d(rate_hz / FPS, sigma))


def stm_filter(C, rate, scale):
    """Separable 2-D Gabor on cochleagram C (T, n_ch); scale in cycles/channel (signed)."""
    n_ch = C.shape[1]
    sc = 0.5 / abs(scale) if scale != 0 else 2.0
    M = chan_matrix(n_ch, scale, sc)
    Y = C @ M.T                                      # channel filtering (complex)
    if rate == 0:
        k = gabor_1d(0, 3.0)
    else:
        k = gabor_1d(rate / FPS, FPS * 0.5 / rate)
    return tconv(Y.real, k) + 1j * tconv(Y.imag, k)


# ----------------------------------------------------------------------------- feature space
class FeatureSpace:
    win = 4096

    def __init__(self, device='cpu'):
        self.device = device
        self.subwins = (1024, 2048, 4096)
        self.gt = {n: gammatone_weights(n, 20) for n in self.subwins}
        self.gt40, self.fc40 = gammatone_weights(4096, 40)
        self.fc20 = self.gt[4096][1]
        self.freqs = np.fft.rfftfreq(4096, 1 / SR)

        # pitch: harmonic log-comb on a fine F0 grid, pooled to 11 F0s (75-450 Hz)
        self.f0_fine = np.geomspace(75, 450, 121)
        self.f0_11 = np.geomspace(75, 450, 11)
        H = np.arange(1, 9)
        wh = 1 / np.sqrt(H); wh /= wh.sum()
        nb = len(self.freqs); df = SR / 4096
        M = np.zeros((nb, len(self.f0_fine)))

        def add(col, f, w):
            p = f / df; k = int(np.floor(p)); a = p - k
            M[k, col] += w * (1 - a); M[k + 1, col] += w * a
        for j, f0 in enumerate(self.f0_fine):
            for h, w in zip(H, wh):
                add(j, h * f0, w)
                add(j, (h - 0.5) * f0, -0.5 * w)
                add(j, (h + 0.5) * f0, -0.5 * w)
        self.comb = M.astype(np.float32)
        lf = np.log(self.f0_fine); edges = np.log(self.f0_11)
        self.pool_idx = np.argmin(np.abs(lf[:, None] - edges[None]), 1)

        # ACF lag range for 75-450 Hz, window autocorrelation for normalisation
        self.lag_lo, self.lag_hi = int(SR / 450), int(np.ceil(SR / 75))
        hw = np.hanning(4096)
        wac = np.fft.irfft(np.abs(np.fft.rfft(hw, 8192)) ** 2)[:4096]
        self.wac = (wac / wac[0]).astype(np.float32)

        # cepstral regression line over the pitch quefrency range
        q = np.arange(self.lag_lo, self.lag_hi + 1)
        A = np.stack([q, np.ones_like(q)], 1).astype(float)
        self.cpp_pinv = np.linalg.pinv(A)
        self.cpp_q = q

        # lifter for smooth spectral envelope (formants)
        self.lifter = np.zeros(4096); self.lifter[:40] = 1; self.lifter[-39:] = 1

        # scattering filterbanks
        self.sc_fc = 125 * 2 ** (np.arange(9) * 0.75)
        self.sc_fm = np.array([25, 50, 100, 200, 400.0])

    # ------------------------------------------------------------------ cochleagrams
    def _spectra(self, w):
        out = {}
        for n in self.subwins:
            s = (4096 - n) // 2
            hw = np.hanning(n).astype(np.float32)
            X = np.fft.rfft(w[:, s:s + n] * hw, axis=1)
            P = (X.real ** 2 + X.imag ** 2).astype(np.float32) / (hw ** 2).sum()
            out[n] = P
        return out

    def cochleagrams(self, P):
        E = np.concatenate([P[n] @ self.gt[n][0].T for n in self.subwins], 1)   # (T, 60)
        L = slog(E)
        C3 = np.cbrt(E / E.mean())
        f = [L]
        f += [L - lp(L, t) for t in (1.5, 4.5, 15, 60, 150)]
        f += [lp(L, 60), lp(L, 150), lp(C3, 60), lp(C3, 150)]
        f += [lp_acausal(L, 60), lp_acausal(C3, 60)]
        for k in (3, 6, 12):
            dp = L - shift(L, k)          # past difference
            df = shift(L, -k) - L         # future difference
            for d in (dp, df):
                f += [d, np.maximum(d, 0), np.maximum(-d, 0)]
        return np.concatenate(f, 1), L

    # ------------------------------------------------------------------ STM
    def stm(self, L20, P4096):
        feats = []
        # 17 rate/scale pairs on the 20-channel 4096 cochleagram, per-channel quadrature energy
        pairs = [(0, 0.25), (0, 0.5)] + [(r, s) for r in (3, 6, 12) for s in (0, .25, -.25, .5, -.5)]
        for r, s in pairs:
            Y = stm_filter(L20, r, s)
            feats.append(slog(np.abs(Y) ** 2))
        A = np.concatenate(feats, 1)

        # 40-channel cochleagram
        C40 = slog(P4096 @ self.gt40.T)
        thirds = [slice(0, 13), slice(13, 26), slice(26, 40)]
        rates = (0.5, 1, 2, 4, 8, 12)
        scales = (0, .25, -.25, .5, -.5)
        cs, mags = [], {}
        for r in rates:
            for s in scales:
                Y = stm_filter(C40, r, s)
                m = np.abs(Y)
                for th in thirds:
                    cs += [Y.real[:, th].mean(1), Y.imag[:, th].mean(1)]
                mags[(r, s)] = np.stack([m[:, th].mean(1) for th in thirds], 1)
        CS = np.stack(cs, 1)
        MG = np.concatenate([mags[(r, s)] for r in rates for s in scales], 1)
        MGn = MG / (MG.mean(0) + 1e-8)
        prods = []
        for i in range(len(rates) - 1):
            for s in scales:
                a = mags[(rates[i], s)]; b = mags[(rates[i + 1], s)]
                prods.append((a / (a.mean(0) + 1e-8)) * (b / (b.mean(0) + 1e-8)))
        PR = np.concatenate(prods, 1)
        B = np.concatenate([CS, MGn, dyn_res(MGn), MGn - lp_acausal(MGn, 60), PR], 1)
        return np.concatenate([A, B], 1)

    # ------------------------------------------------------------------ descriptors
    def descriptors(self, w, P, L20, E20):
        T = len(P)
        f = self.freqs
        LP = slog(P)
        # pitch salience
        sal_f = LP @ self.comb                                    # (T, 121)
        sal11 = np.stack([sal_f[:, self.pool_idx == j].max(1) for j in range(11)], 1)
        best = np.argmax(sal_f, 1)
        bestf0 = np.log2(self.f0_fine[best])
        strength = sal_f.max(1)
        # ACF / HNR
        r = np.fft.irfft(P, axis=1)[:, :4096]
        r = r / (r[:, :1] + 1e-30)
        rr = r[:, self.lag_lo:self.lag_hi + 1] / self.wac[self.lag_lo:self.lag_hi + 1]
        rmax = np.clip(rr.max(1), 1e-3, 0.999)
        hnr = np.log(rmax / (1 - rmax))
        voicing = 1 / (1 + np.exp(-10 * (rmax - 0.5)))
        # cepstral peak prominence
        cep = np.fft.irfft(LP, axis=1)
        cq = cep[:, self.cpp_q]
        coef = cq @ self.cpp_pinv.T                               # (T, 2)
        pk = np.argmax(cq, 1)
        cpp = cq[np.arange(T), pk] - (coef[:, 0] * self.cpp_q[pk] + coef[:, 1])
        # spectral moments, flatness, rolloff (50-8000 Hz)
        band = (f >= 50) & (f <= 8000)
        Pb = P[:, band] + 1e-6 * P.mean() + 1e-30
        fk = f[band] / 1000.0
        p = Pb / Pb.sum(1, keepdims=True)
        mu = p @ fk
        var = p @ fk ** 2 - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-8))
        dev = (fk[None] - mu[:, None]) / sd[:, None]
        skew = (p * dev ** 3).sum(1)
        kurt = np.log((p * dev ** 4).sum(1) + 1e-8)
        flat = np.log(Pb).mean(1) - np.log(Pb.mean(1))
        cdf = np.cumsum(p, 1)
        roll = fk[np.minimum((cdf < 0.85).sum(1), len(fk) - 1)]
        # octave spectral contrast
        oc_edges = [50, 200, 400, 800, 1600, 3200, 6400, 11025]
        octc = []
        for lo, hi in zip(oc_edges[:-1], oc_edges[1:]):
            s = np.sort(P[:, (f >= lo) & (f < hi)], 1)
            k = max(1, int(round(0.2 * s.shape[1])))
            octc.append(np.log(s[:, -k:].mean(1) + 1e-6 * P.mean()) - np.log(s[:, :k].mean(1) + 1e-6 * P.mean()))
        octc = np.stack(octc, 1)
        chc = np.stack([L20[:, g * 5:(g + 1) * 5].max(1) - L20[:, g * 5:(g + 1) * 5].min(1) for g in range(4)], 1)
        # softmax formants on liftered log spectrum
        Ls = np.fft.rfft(cep * self.lifter, axis=1).real
        forms = []
        for lo, hi in ((250, 900), (800, 2500), (2000, 3500)):
            m = (f >= lo) & (f <= hi)
            z = 2.0 * Ls[:, m]
            z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
            forms.append(z @ (f[m] / 1000.0))
        forms = np.stack(forms, 1)
        # flux, loudness
        dl = L20 - shift(L20, 1)
        flux = np.maximum(dl, 0).mean(1)
        loud = slog(E20.sum(1))
        # phonetic band contrasts (gammatone groups: <300, 300-1k, 1k-3k, >3k Hz)
        fc = self.fc20
        grp = [fc < 300, (fc >= 300) & (fc < 1000), (fc >= 1000) & (fc < 3000), fc >= 3000]
        Bk = [slog(E20[:, g].sum(1)) for g in grp]
        phon = np.stack([Bk[1] - Bk[0], Bk[2] - Bk[1], Bk[3] - Bk[2], Bk[3] - loud], 1)

        S = np.concatenate([sal11, np.stack([bestf0, strength, hnr, cpp, mu, sd, skew, kurt,
                                             flat, roll], 1),
                            octc, chc, forms, np.stack([flux, loud, voicing], 1), phon], 1)
        gated = voicing[:, None] * L20
        Sz = zscore(S)
        env = np.concatenate([np.abs(tgabor(Sz, r)) for r in (2, 4, 6, 10)], 1)
        out = np.concatenate([dyn(S), dyn(gated), env], 1)
        key = dict(loud=loud, flux=flux, f0=bestf0, strength=strength, hnr=hnr, cpp=cpp,
                   F1=forms[:, 0], F2=forms[:, 1], F3=forms[:, 2], centroid=mu, flat=flat,
                   voicing=voicing)
        return out, key, phon

    # ------------------------------------------------------------------ interactions
    def interactions(self, key, phon):
        K = np.concatenate([np.stack(list(key.values()), 1), phon], 1)     # (T, 16)
        D = dyn(K)
        partners = [key[k] for k in ('f0', 'strength', 'loud', 'flux', 'F1', 'F2', 'F3')]
        pz = zscore(np.stack(partners, 1))
        cz = zscore(phon)
        prods = [cz[:, i:i + 1] * pz for i in range(cz.shape[1])]
        kz = {k: zscore(v[:, None]) for k, v in key.items()}
        pairs = [('strength', 'loud'), ('flux', 'loud'), ('voicing', 'F1'), ('voicing', 'F2'),
                 ('voicing', 'F3'), ('strength', 'flux'), ('f0', 'loud'), ('hnr', 'loud'),
                 ('cpp', 'loud')]
        prods += [kz[a] * kz[b] for a, b in pairs]
        prods = np.concatenate(prods, 1)
        z = zscore(np.concatenate([D, prods], 1))
        zc = np.clip(z, -4, 4)
        return np.concatenate([np.maximum(z, 0), np.maximum(-z, 0), zc ** 3 / 8], 1)

    # ------------------------------------------------------------------ scattering
    def scattering(self, w):
        T = len(w)
        dev = self.device
        x = torch.from_numpy(w).to(dev)
        hw = torch.hann_window(4096, periodic=False, dtype=torch.float32, device=dev)
        X = torch.fft.fft(x * hw, dim=1)
        f = torch.fft.fftfreq(4096, 1 / SR, device=dev).float()
        fpos = torch.clamp(f, min=0) * (f > 0)
        s1, s2, ifd, coh = [], [], [], []
        ref = None
        for fc in self.sc_fc:
            sig = fc / 4 / 2.355
            psi = 2 * torch.exp(-0.5 * ((fpos - fc) / sig) ** 2) * (f > 0)
            xc = torch.fft.ifft(X * psi, dim=1)
            U = xc.abs()
            mU = U.mean(1)
            if ref is None:
                ref = 1e-4 * float(mU.mean()) + 1e-20
            s1.append(torch.log(mU + ref))
            Uf = torch.fft.fft(U, dim=1)
            for fm in self.sc_fm:
                sm = fm / 2.355
                psi2 = 2 * torch.exp(-0.5 * ((fpos - fm) / sm) ** 2) * (f > 0)
                V = torch.fft.ifft(Uf * psi2, dim=1).abs().mean(1)
                s2.append(torch.log(V + 0.01 * mU + ref) - torch.log(mU + ref))
            lag = int(round(SR / (fc / 4)))
            z = (xc[:, lag:] * xc[:, :-lag].conj()).sum(1)
            z = z * np.exp(-2j * np.pi * fc * lag / SR)
            pw = (U[:, lag:] * U[:, :-lag]).sum(1) + 1e-30
            ifd.append(torch.angle(z) / (2 * np.pi))
            coh.append(z.abs() / pw)
        S = torch.stack(s1 + s2 + ifd + coh, 1).cpu().numpy().astype(np.float64)
        return dyn(S)

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        w = np.asarray(windows, dtype=np.float32)
        T = len(w)
        P = self._spectra(w)
        coch, L = self.cochleagrams(P)
        L20 = L[:, 40:60]
        E20 = P[4096] @ self.gt[4096][0].T
        stm = self.stm(L20.astype(np.float64), P[4096])
        desc, key, phon = self.descriptors(w, P[4096].astype(np.float64), L20, E20.astype(np.float64))
        inter = self.interactions(key, phon)
        scat = self.scattering(w)
        out = np.concatenate([coch, stm, desc, inter, scat], 1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1e6, 1e6).astype(np.float32)
