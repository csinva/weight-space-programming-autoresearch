"""Closed-form acoustic feature space (no learned or data-fitted parameters).

Groups (see FEATURES.md):
  A. multi-resolution gammatone cochleagrams + temporal dynamics
  B. spectrotemporal modulation (2-D Gabor) energies
  C. spectral / pitch / voice descriptors with dyn + temporal-Gabor envelopes
  D. nonlinear interactions of key scalars
  E. within-window Morlet scattering + instantaneous frequency / coherence
"""
import numpy as np
from scipy.signal import lfilter
from scipy.ndimage import maximum_filter1d

SR = 44100.0
FPS = 30.0
DYN_TAUS = (4.5, 15.0, 60.0)
EPS = 1e-8


# ----------------------------------------------------------------------------- helpers
def lp(x, tau):
    """Causal one-pole lowpass over frames (axis 0), alpha = 1/(1+tau), initialised at x[0]."""
    a = 1.0 / (1.0 + tau)
    x = np.asarray(x, np.float64)
    zi = (1.0 - a) * x[:1]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, axis=0, zi=zi)
    return y


def lp_acausal(x, tau):
    """Zero-phase (forward-backward) version of lp."""
    return lp(lp(x, tau)[::-1], tau)[::-1]


def dyn(x):
    return np.concatenate([x - lp(x, t) for t in DYN_TAUS], axis=1)


def with_dyn(x):
    return np.concatenate([x, dyn(x)], axis=1)


def zscore(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def conv_time(x, k):
    """Centred (acausal) convolution along axis 0 with kernel k (odd length, may be complex);
    edge-replicate padding.  x: (T, C)."""
    T = x.shape[0]
    h = len(k) // 2
    xp = np.concatenate([np.repeat(x[:1], h, 0), x, np.repeat(x[-1:], h, 0)], 0)
    n = xp.shape[0] + len(k) - 1
    nf = 1 << int(np.ceil(np.log2(n)))
    if np.iscomplexobj(k):
        y = np.fft.ifft(np.fft.fft(xp, nf, axis=0) * np.fft.fft(k, nf)[:, None], axis=0)
    else:
        y = np.fft.irfft(np.fft.rfft(xp, nf, axis=0) * np.fft.rfft(k, nf)[:, None], nf, axis=0)
    return y[2 * h:2 * h + T]


def erb_space(lo, hi, n):
    erb = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    ierb = lambda e: (10 ** (e / 21.4) - 1) / 0.00437
    return ierb(np.linspace(erb(lo), erb(hi), n))


def gammatone_weights(nfft, cfs):
    """Fourth-order gammatone power responses sampled at rfft bin frequencies, unit-sum rows."""
    f = np.arange(nfft // 2 + 1) * SR / nfft
    b = 1.019 * 24.7 * (4.37 * cfs / 1000.0 + 1.0)
    W = (1.0 + ((f[None, :] - cfs[:, None]) / b[:, None]) ** 2) ** -4
    W /= W.sum(1, keepdims=True)
    return W.T.astype(np.float32)  # (F, C)


def gabor_time(rate, sigma_s):
    """Complex temporal Gabor (frames); rate in Hz, sigma in seconds; DC removed for rate>0."""
    sig = sigma_s * FPS
    h = int(np.ceil(3 * sig))
    t = np.arange(-h, h + 1)
    g = np.exp(-0.5 * (t / sig) ** 2)
    if rate == 0:
        return (g / g.sum()).astype(np.complex128)
    c = np.exp(2j * np.pi * rate * t / FPS)
    k = g * c
    k = k - g * (k.sum() / g.sum())          # zero-mean
    return k / np.abs(k).sum()


def gabor_chan_matrix(C, scale, sigma_c):
    """(C, C) complex matrix applying a centred channel-axis Gabor (cycles/channel)."""
    c = np.arange(C)
    d = c[None, :] - c[:, None]                 # [out, in]
    g = np.exp(-0.5 * (d / sigma_c) ** 2) * np.exp(2j * np.pi * scale * d)
    if scale != 0:
        g = g - np.exp(-0.5 * (d / sigma_c) ** 2) * (g.sum(1, keepdims=True) /
                                                    np.exp(-0.5 * (d / sigma_c) ** 2).sum(1, keepdims=True))
    g /= np.abs(g).sum(1, keepdims=True)
    return g.T                                  # x (T, C_in) @ M -> (T, C_out)


# ----------------------------------------------------------------------------- feature space
class FeatureSpace:
    win = 4096

    def __init__(self, device='cpu'):
        self.device = device
        self.cf20 = erb_space(50.0, 8000.0, 20)
        self.cf40 = erb_space(50.0, 8000.0, 40)
        self.subwins = (1024, 2048, 4096)
        self.W20 = {n: gammatone_weights(n, self.cf20) for n in self.subwins}
        self.W40 = gammatone_weights(4096, self.cf40)
        self.hann = {n: np.hanning(n + 2)[1:-1].astype(np.float32) for n in self.subwins}
        self.freqs = np.arange(2049) * SR / 4096
        # pitch candidates
        self.f0_11 = 75.0 * 6.0 ** (np.arange(11) / 10.0)
        self.f0_fine = 75.0 * 6.0 ** (np.arange(61) / 60.0)
        # scattering filters (first order Q=4 on 9 channels, second order Q=1)
        self.sc_fc = 125.0 * 2.0 ** (0.75 * np.arange(9))
        self.sc_fm = 25.0 * 2.0 ** np.arange(6)

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        x = np.asarray(windows, np.float32)
        T, N = x.shape
        # global (segment-level) level normalisation so that fixed floors are meaningful
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
        x = x / rms

        spec = {}
        for n in self.subwins:
            o = (N - n) // 2
            fr = x[:, o:o + n] * self.hann[n]
            P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 / (self.hann[n] ** 2).sum()
            spec[n] = P.astype(np.float32)
        P4 = spec[4096]

        coch_pow = {n: spec[n] @ self.W20[n] for n in self.subwins}            # (T,20)
        coch_log = {n: np.log(coch_pow[n] + EPS) for n in self.subwins}

        feats = []
        feats += self.cochleagram_feats(coch_pow, coch_log)
        feats += self.stm_feats(coch_log[4096], np.log(P4 @ self.W40 + EPS))
        scal, key, contrasts, gated = self.descriptors(x, P4, coch_log[4096])
        feats += self.descriptor_feats(scal, gated)
        feats += self.interaction_feats(key, contrasts)
        feats += self.scattering_feats(x)

        out = np.concatenate([np.asarray(f, np.float64).reshape(T, -1) for f in feats], axis=1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1e6, 1e6).astype(np.float32)

    # ------------------------------------------------------------------ A. cochleagrams
    def cochleagram_feats(self, cpow, clog):
        E = np.concatenate([clog[n] for n in self.subwins], 1)                 # (T,60)
        C3 = np.concatenate([np.cbrt(cpow[n]) for n in self.subwins], 1)
        out = [E]
        out += [E - lp(E, t) for t in (1.5, 4.5, 15.0, 60.0, 150.0)]
        out += [lp(E, 60.0), lp(E, 150.0), lp(C3, 60.0), lp(C3, 150.0)]
        out += [lp_acausal(E, 60.0)]
        T = E.shape[0]
        for k in (3, 6, 12):
            past = E - np.concatenate([np.repeat(E[:1], k, 0), E[:-k]], 0)[:T]
            fut = np.concatenate([E[k:], np.repeat(E[-1:], k, 0)], 0)[:T] - E
            for d in (past, fut):
                out += [d, np.maximum(d, 0), np.maximum(-d, 0)]
        return out

    # ------------------------------------------------------------------ B. spectrotemporal modulation
    def stm_feats(self, c20, c40):
        out = []
        # (1) 17 rate/scale quadrature energies on the 20-channel 4096 cochleagram
        pairs = [(0.0, 0.25), (0.0, 0.5)]
        pairs += [(r, s) for r in (1.0, 3.0, 6.0, 12.0) for s in (0.0, 0.25, -0.25)]
        pairs += [(r, 0.5) for r in (3.0, 6.0, 12.0)]
        X = c20 - c20.mean(0)
        tcache = {}
        for r, s in pairs:
            if r not in tcache:
                kt = gabor_time(r, 0.6 / r if r > 0 else 0.25)
                tcache[r] = conv_time(X, kt)
            M = gabor_chan_matrix(20, s, 0.6 / abs(s) if s != 0 else 2.0)
            out.append(np.abs(tcache[r] @ M))
        # (2) 40-channel cochleagram: rates x scales, cos/sin/mag averaged over frequency thirds
        X = c40 - c40.mean(0)
        thirds = np.array_split(np.arange(40), 3)
        rates = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
        scales = (0.0, 0.25, -0.25, 0.5)
        cs, mags = [], {}
        for r in rates:
            Y = conv_time(X, gabor_time(r, 0.6 / r))
            for s in scales:
                Z = Y @ gabor_chan_matrix(40, s, 0.6 / abs(s) if s != 0 else 2.0)
                for ti, th in enumerate(thirds):
                    z = Z[:, th]
                    m = np.abs(z).mean(1)
                    cs += [z.real.mean(1), z.imag.mean(1), m]
                    mags[(r, s, ti)] = m
        B = np.stack(cs, 1)                                                    # (T,216)
        out.append(with_dyn(B))
        Mg = np.stack([mags[k] for k in sorted(mags)], 1)                      # (T,72)
        out.append(Mg - lp_acausal(Mg, 15.0))
        prods = []
        for i in range(len(rates) - 1):
            for s in scales:
                for ti in range(3):
                    prods.append(np.sqrt(mags[(rates[i], s, ti)] * mags[(rates[i + 1], s, ti)]))
        out.append(np.stack(prods, 1))                                         # (T,60)
        return out

    # ------------------------------------------------------------------ C. descriptors
    def descriptors(self, x, P, c20):
        T = P.shape[0]
        f = self.freqs
        L = np.log(P + EPS).astype(np.float64)

        # cepstrum, spectral envelope
        ceps = np.fft.irfft(L, 4096, axis=1)
        lift = np.zeros(4096); lift[:80] = 1; lift[-79:] = 1
        env = np.fft.rfft(ceps * lift, axis=1).real                           # (T,2049)

        # pitch salience: 8-harmonic log comb on envelope-whitened log spectrum
        Lw = maximum_filter1d(L - env, 3, axis=1)
        binf = lambda fr: np.interp(fr, f, np.arange(len(f)))
        hw = 1.0 / np.sqrt(np.arange(1, 9)); hw /= hw.sum()

        def salience(f0s):
            S = np.zeros((T, len(f0s)))
            for j, f0 in enumerate(f0s):
                for h in range(1, 9):
                    on = binf(h * f0); off = binf((h - 0.5) * f0)
                    S[:, j] += hw[h - 1] * (_lin(Lw, on) - _lin(Lw, off))
            return S
        sal11 = salience(self.f0_11)
        salf = salience(self.f0_fine)
        bi = salf.argmax(1)
        bestf0 = np.log2(self.f0_fine[bi])
        strength = salf.max(1)

        # ACF HNR / voicing
        xw = x * np.hanning(x.shape[1] + 2)[1:-1].astype(np.float32)
        acf = np.fft.irfft(np.abs(np.fft.rfft(xw, 8192, axis=1)) ** 2, 8192, axis=1)[:, :700]
        wac = np.fft.irfft(np.abs(np.fft.rfft(np.hanning(x.shape[1] + 2)[1:-1], 8192)) ** 2, 8192)[:700]
        r = acf / (acf[:, :1] + 1e-12) / (wac / wac[0])[None]
        lo, hi = int(SR / 450), int(SR / 75)
        rmax = np.clip(r[:, lo:hi].max(1), 1e-3, 0.999)
        hnr = 10 * np.log10(rmax / (1 - rmax))
        voicing = np.clip(r[:, lo:hi].max(1), 0, 1)

        # cepstral peak prominence
        q = ceps[:, int(0.0022 * SR):int(0.0133 * SR)]
        cpp = q.max(1) - q.mean(1)

        # spectral moments etc. (50 Hz - 10 kHz)
        band = (f >= 50) & (f <= 10000)
        Pb = P[:, band].astype(np.float64) + EPS
        fb = f[band] / 1000.0
        w = Pb / Pb.sum(1, keepdims=True)
        cen = (w * fb).sum(1)
        spr = np.sqrt((w * (fb - cen[:, None]) ** 2).sum(1)) + 1e-9
        skw = (w * (fb - cen[:, None]) ** 3).sum(1) / spr ** 3
        krt = np.log((w * (fb - cen[:, None]) ** 4).sum(1) / spr ** 4)
        flat = np.log(Pb).mean(1) - np.log(Pb.mean(1))
        roll = fb[np.minimum((np.cumsum(w, 1) < 0.85).sum(1), len(fb) - 1)]

        # octave spectral contrast (7 octaves 62.5-8000 Hz)
        octc = []
        for k in range(7):
            sel = (f >= 62.5 * 2 ** k) & (f < 62.5 * 2 ** (k + 1))
            Ls = np.sort(L[:, sel], 1)
            m = max(1, int(round(0.2 * Ls.shape[1])))
            octc.append(Ls[:, -m:].mean(1) - Ls[:, :m].mean(1))
        octc = np.stack(octc, 1)
        # channel-group contrast (5 groups of 4 cochlear channels)
        g = c20.reshape(T, 5, 4)
        chc = g.max(2) - g.min(2)

        # softmax formants on the cepstral envelope
        forms = []
        for lo_f, hi_f in ((250, 900), (850, 2500), (2300, 3500)):
            sel = (f >= lo_f) & (f <= hi_f)
            e = 2.0 * env[:, sel]
            e = np.exp(e - e.max(1, keepdims=True))
            forms.append(np.log((e * f[sel]).sum(1) / e.sum(1)))
        forms = np.stack(forms, 1)

        # spectral flux on the log cochleagram
        dc = np.diff(c20, axis=0, prepend=c20[:1])
        flux = np.maximum(dc, 0).mean(1)

        loud = np.log(np.exp(c20).sum(1) + EPS)
        scal = np.column_stack([sal11, bestf0, strength, voicing, hnr, cpp, loud, cen, spr, skw, krt,
                                flat, roll, octc, chc, forms, flux])           # (T,39)
        gated = voicing[:, None] * c20                                         # (T,20)

        cf = self.cf20
        lowb, midb, highb = cf < 500, (cf >= 500) & (cf < 2000), cf >= 2000
        bl = lambda s: np.log(np.exp(c20[:, s]).sum(1) + EPS)
        Bl, Bm, Bh = bl(lowb), bl(midb), bl(highb)
        contrasts = np.column_stack([Bm - Bl, Bh - Bm, Bh - Bl])
        key = {'f0': bestf0, 'strength': strength, 'loud': loud, 'flux': flux,
               'F1': forms[:, 0], 'F2': forms[:, 1], 'F3': forms[:, 2], 'hnr': hnr,
               'voicing': voicing, 'cen': cen, 'flat': flat}
        return scal, key, contrasts, gated

    def descriptor_feats(self, scal, gated):
        out = [with_dyn(np.concatenate([scal, gated], 1))]                     # (T,236)
        Z = zscore(scal)
        for rate in (2.0, 4.0, 6.0, 10.0):
            out.append(np.abs(conv_time(Z, gabor_time(rate, 0.5 / rate))))     # (T,39) each
        return out

    # ------------------------------------------------------------------ D. interactions
    def interaction_feats(self, key, contrasts):
        names = ['f0', 'strength', 'voicing', 'loud', 'flux', 'F1', 'F2', 'F3', 'hnr', 'cen', 'flat']
        K = np.column_stack([key[n] for n in names] + [contrasts])            # (T,14)
        Kz = zscore(K)
        prods = []
        for c in range(3):
            cz = zscore(contrasts[:, c])
            for n in ['f0', 'strength', 'loud', 'flux', 'F1', 'F2', 'F3', 'hnr', 'cen']:
                prods.append(cz * zscore(key[n]))
        Z = np.concatenate([K, dyn(K), np.stack(prods, 1)], 1)                 # (T,83)
        Z = np.clip(zscore(Z), -8, 8)
        return [np.maximum(Z, 0), np.maximum(-Z, 0), Z ** 3]

    # ------------------------------------------------------------------ E. scattering
    def scattering_feats(self, x, chunk=256):
        T, N = x.shape
        hw = np.hanning(N + 2)[1:-1].astype(np.float32)
        fr = np.fft.fftfreq(N, 1.0 / SR)
        # analytic Morlet-like (Gaussian) first-order filters, Q=4 (FWHM = fc/4)
        sig1 = (self.sc_fc / 4.0) / 2.3548
        H1 = np.exp(-0.5 * ((fr[None, :] - self.sc_fc[:, None]) / sig1[:, None]) ** 2)  # (9,N)
        H1[:, fr <= 0] = 0
        # second order, Q=1 (FWHM = fm), on envelopes; work on decimated band (0..nb bins)
        nb = 256
        fr2 = np.fft.fftfreq(N, 1.0 / SR)[:nb]
        sig2 = (self.sc_fm / 1.0) / 2.3548
        H2 = np.exp(-0.5 * ((fr2[None, :] - self.sc_fm[:, None]) / sig2[:, None]) ** 2)  # (6,nb)
        S1 = np.zeros((T, 9)); S2 = np.zeros((T, 9, 6)); IF = np.zeros((T, 9)); CO = np.zeros((T, 9))
        for a in range(0, T, chunk):
            X = np.fft.fft(x[a:a + chunk] * hw, axis=1)                        # (B,N)
            Zc = np.fft.ifft(X[:, None, :] * H1[None] * 2, axis=2)             # (B,9,N) analytic
            env = np.abs(Zc)
            S1[a:a + chunk] = env.mean(2)
            Ef = np.fft.fft(env, axis=2)[:, :, :nb]                            # low band of envelope
            M = np.fft.ifft(Ef[:, :, None, :] * H2[None, None], axis=3)        # (B,9,6,nb)
            S2[a:a + chunk] = np.abs(M).mean(3) * nb / N
            pr = (np.conj(Zc[:, :, :-1]) * Zc[:, :, 1:]).sum(2)
            IF[a:a + chunk] = np.angle(pr) * SR / (2 * np.pi) / self.sc_fc[None] - 1.0
            CO[a:a + chunk] = np.abs(pr) / ((env[:, :, :-1] * env[:, :, 1:]).sum(2) + 1e-12)
        lS1 = np.log(S1 + 1e-6)
        lS2 = np.log(S2 + 1e-6) - lS1[:, :, None]
        return [with_dyn(lS1), lS2.reshape(T, -1), with_dyn(IF), CO]


def _lin(A, pos):
    """Linear interpolation of A[:, pos] for scalar fractional bin position."""
    i = int(np.floor(pos)); fr = pos - i
    i = min(max(i, 0), A.shape[1] - 2)
    return (1 - fr) * A[:, i] + fr * A[:, i + 1]
