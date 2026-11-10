"""Closed-form acoustic feature space (3,852 dims, win=4096).

Groups (see FEATURES.md):
  cochleagrams, spectrotemporal modulation, descriptors, interactions, within-window scattering.
No learned or fitted parameters; every constant below is fixed by hand.
"""
import numpy as np
import scipy.fft as sfft
from scipy.signal import lfilter, fftconvolve

SR = 44100
FPS = 30.0
EPS = 1e-8
DYN_TAUS = (4.5, 15.0, 60.0)


# ----------------------------------------------------------------------------------------------
# generic temporal operators (axis 0 = frames)
# ----------------------------------------------------------------------------------------------
def lp(x, tau):
    """Causal one-pole lowpass, alpha = 1/(1+tau), initialised at the first frame."""
    a = 1.0 / (1.0 + tau)
    zi = (1.0 - a) * x[:1]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, axis=0, zi=zi)
    return y


def lp_acausal(x, tau):
    """Forward-backward one-pole lowpass (zero phase)."""
    return lp(lp(x, tau)[::-1], tau)[::-1]


def dyn(x, taus=DYN_TAUS):
    return np.concatenate([x - lp(x, t) for t in taus], axis=1)


def shift(x, k):
    """y[t] = x[t-k] (k>0 past, k<0 future), edge-replicated."""
    if k == 0:
        return x
    y = np.empty_like(x)
    if k > 0:
        y[k:] = x[:-k]
        y[:k] = x[:1]
    else:
        k = -k
        y[:-k] = x[k:]
        y[-k:] = x[-1:]
    return y


def gabor_t(rate, sigma_s=None):
    """Complex temporal Gabor kernel over frames (rate in Hz); rate 0 -> real Gaussian lowpass."""
    if sigma_s is None:
        sigma_s = 0.5 / rate if rate > 0 else 0.25
    sig = sigma_s * FPS
    n = np.arange(-int(np.ceil(3 * sig)), int(np.ceil(3 * sig)) + 1)
    env = np.exp(-0.5 * (n / sig) ** 2)
    if rate == 0:
        return env / env.sum()
    g = env * np.exp(2j * np.pi * rate * n / FPS)
    g = g - env * (g.sum() / env.sum())          # zero DC
    return g / np.abs(g).sum()


def tconv(x, k):
    """'same' convolution along frames with edge padding. x (T, D), k (L,) possibly complex."""
    L = len(k)
    h = L // 2
    xp = np.concatenate([np.repeat(x[:1], h, 0), x, np.repeat(x[-1:], h, 0)], 0)
    return fftconvolve(xp, k[:, None], mode='valid', axes=0)


def gabor_c(scale, C):
    """Spectral (channel-axis) Gabor as a (C, C) matrix; scale in cycles/channel (signed)."""
    sig = 1.0 / (2 * abs(scale)) if scale != 0 else 3.0
    d = np.arange(C)[None, :] - np.arange(C)[:, None]       # in - out
    env = np.exp(-0.5 * (d / sig) ** 2)
    if scale == 0:
        M = env
    else:
        M = env * np.exp(2j * np.pi * scale * d)
        M = M - env * (M.sum(1, keepdims=True) / env.sum(1, keepdims=True))
    return M / np.abs(M).sum(1, keepdims=True)


def zscore(x):
    m = x.mean(0, keepdims=True)
    s = x.std(0, keepdims=True)
    return (x - m) / np.where(s > 1e-9, s, 1.0)


# ----------------------------------------------------------------------------------------------
# filterbanks
# ----------------------------------------------------------------------------------------------
def erb_space(lo, hi, n):
    e = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    ie = lambda E: (10 ** (E / 21.4) - 1) / 0.00437
    return ie(np.linspace(e(lo), e(hi), n))


def gammatone_weights(nfft, cf):
    f = np.arange(nfft // 2 + 1) * SR / nfft
    b = 1.019 * 24.7 * (4.37 * cf / 1000 + 1)
    W = (1 + ((f[None, :] - cf[:, None]) / b[:, None]) ** 2) ** -4.0     # 4th-order power response
    return (W / W.sum(1, keepdims=True)).astype(np.float32)


def interp_matrix(freqs, nfft):
    """Linear interpolation of a spectrum (nfft//2+1 bins) at arbitrary frequencies."""
    nb = nfft // 2 + 1
    pos = np.clip(np.asarray(freqs) * nfft / SR, 0, nb - 1.001)
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    M = np.zeros((len(pos), nb))
    M[np.arange(len(pos)), i0] += 1 - fr
    M[np.arange(len(pos)), i0 + 1] += fr
    return M


def comb_matrix(f0s, nfft, nh=8):
    """8-harmonic log-comb: mean over harmonics of L(hF0) - mean(L((h-.5)F0), L((h+.5)F0)), w_h=h^-.5."""
    rows = []
    w = np.arange(1, nh + 1) ** -0.5
    w = w / w.sum()
    for f0 in f0s:
        h = np.arange(1, nh + 1)
        M = (w[:, None] * (interp_matrix(h * f0, nfft)
                           - 0.5 * interp_matrix((h - 0.5) * f0, nfft)
                           - 0.5 * interp_matrix((h + 0.5) * f0, nfft))).sum(0)
        rows.append(M)
    return np.array(rows, dtype=np.float32)


class FeatureSpace:
    win = 4096

    def __init__(self, device='cpu'):
        self.device = device
        N = self.win
        self.subwins = (1024, 2048, 4096)
        self.hann = {n: np.hanning(n + 2)[1:-1].astype(np.float32) for n in self.subwins}
        self.cf20 = erb_space(50, 8000, 20)
        self.cf40 = erb_space(50, 8000, 40)
        self.gt20 = {n: gammatone_weights(n, self.cf20) for n in self.subwins}
        self.gt40 = gammatone_weights(N, self.cf40)
        self.freqs = np.arange(N // 2 + 1) * SR / N
        # pitch
        self.f0_11 = np.geomspace(75, 450, 11)
        self.f0_fine = np.geomspace(75, 450, 63)
        self.comb11 = comb_matrix(self.f0_11, N)
        self.comb_fine = comb_matrix(self.f0_fine, N)
        # cepstral peak prominence: line fit over quefrencies 1-22 ms, peak in 75-450 Hz periods
        self.q_lo, self.q_hi = int(SR / 450), int(SR / 75)
        q = np.arange(44, 1000)
        X = np.stack([np.ones_like(q), q], 1).astype(np.float64)
        self.cpp_q = q
        self.cpp_proj = np.linalg.pinv(X)     # closed-form regression operator (fixed)
        # ACF normalisation by the window autocorrelation
        w = self.hann[N].astype(np.float64)
        rw = sfft.irfft(np.abs(sfft.rfft(w, 2 * N)) ** 2)[:N]
        self.rw = rw / rw[0]
        # octave bands for spectral contrast
        self.oct_edges = [100 * 2 ** m for m in range(8)]
        # STM (20-channel 4096 cochleagram): 17 rate/scale pairs
        self.stm17 = ([(0, s) for s in (0.125, 0.25, 0.5)]
                      + [(r, 0) for r in (2, 4, 8, 12)]
                      + [(r, s) for r in (2, 4, 8, 12) for s in (0.25, -0.25)]
                      + [(4, 0.5), (4, -0.5)])
        assert len(self.stm17) == 17
        self.stm_rates = (0.5, 1, 2, 4, 8, 12)
        self.stm_scales = (0, 0.25, -0.25, 0.5, -0.5)
        # scattering filters (on the 4096 window)
        self._init_scattering()
        self.group_dims = {}

    # ------------------------------------------------------------------------------------------
    def _init_scattering(self):
        N = self.win
        f = np.arange(N // 2 + 1) * SR / N
        Q = 4
        bw = lambda xi, Q: xi * (2 ** (1 / (2 * Q)) - 2 ** (-1 / (2 * Q))) / 2.355
        self.sc_bands = [62.5 * 2 ** m for m in range(9)]          # 9 octave channels 62.5 Hz .. 32 kHz
        psi1 = []
        self.sc_cent = []
        for m in range(9):
            for q in range(Q):
                xi = 62.5 * 2 ** (m + (q + 0.5) / Q)
                xi = min(xi, 20000.0)
                psi1.append(np.exp(-0.5 * ((f - xi) / max(bw(xi, Q), 2 * SR / N)) ** 2))
            self.sc_cent.append(62.5 * 2 ** (m + 0.5))
        self.psi1 = np.array(psi1, dtype=np.float32)               # (36, N/2+1), analytic
        self.mod_f = (25.0, 50.0, 100.0, 200.0, 400.0)
        psi2 = [np.exp(-0.5 * ((f - xi) / bw(xi, 1)) ** 2) for xi in self.mod_f]
        self.psi2 = np.array(psi2, dtype=np.float32)               # Q=1
        self.sc_mid = slice(N // 4, 3 * N // 4)

    # ------------------------------------------------------------------------------------------
    def _spectra(self, w):
        N = self.win
        out = {}
        for n in self.subwins:
            a = (N - n) // 2
            x = w[:, a:a + n] * self.hann[n]
            X = sfft.rfft(x, axis=1, workers=-1)
            out[n] = (X.real ** 2 + X.imag ** 2) / float((self.hann[n] ** 2).sum())
        return out

    # ------------------------------------------------------------------------------------------
    def _cochleagram(self, P):
        logs, cubes = [], []
        for n in self.subwins:
            E = P[n] @ self.gt20[n].T
            logs.append(np.log(E + EPS))
            cubes.append(np.cbrt(E))
        L = np.concatenate(logs, 1).astype(np.float64)     # (T, 60)
        Cb = np.concatenate(cubes, 1).astype(np.float64)
        f = [L]
        f += [L - lp(L, t) for t in (1.5, 4.5, 15, 60, 150)]
        f += [lp(L, 60), lp(L, 150), lp(Cb, 60), lp(Cb, 150)]
        f += [lp_acausal(L, 60)]
        for k in (3, 6, 12):
            for d in (L - shift(L, k), shift(L, -k) - L):
                f += [d, np.maximum(d, 0), np.maximum(-d, 0)]
        return np.concatenate(f, 1), logs[-1].astype(np.float64)

    # ------------------------------------------------------------------------------------------
    def _stm(self, L20, L40):
        f = []
        # 17 quadrature energies, per channel, on the 20-channel 4096 cochleagram
        for r, s in self.stm17:
            M = gabor_c(s, L20.shape[1])
            y = tconv(L20 @ M.T, gabor_t(r))
            f.append(np.sqrt(np.abs(y) ** 2 + 1e-6))
        # 40-channel cochleagram: rates x scales, cos / sin / magnitude averaged over thirds
        thirds = [slice(0, 13), slice(13, 26), slice(26, 40)]
        C, S, A = [], [], []
        mags = {}
        for r in self.stm_rates:
            kt = gabor_t(r)
            for s in self.stm_scales:
                M = gabor_c(s, 40)
                y = tconv(L40 @ M.T, kt)
                a = np.abs(y)
                for j, th in enumerate(thirds):
                    C.append(y[:, th].real.mean(1))
                    S.append(y[:, th].imag.mean(1))
                    A.append(a[:, th].mean(1))
                    mags[(r, s, j)] = A[-1]
        C, S, A = (np.stack(v, 1) for v in (C, S, A))
        base = np.concatenate([C, S, A], 1)                        # 270
        f.append(base)
        f.append(dyn(A))                                           # 270
        f.append(base - lp_acausal(base, 15))                      # 270 acausal residuals
        prods = []
        R = self.stm_rates
        for i in range(len(R)):
            for k in range(i + 1, len(R)):
                for s in self.stm_scales:
                    for j in range(3):
                        prods.append(np.sqrt(mags[(R[i], s, j)] * mags[(R[k], s, j)]))
        f.append(np.stack(prods, 1))                               # 225 cross-rate energy products
        return np.concatenate(f, 1)

    # ------------------------------------------------------------------------------------------
    def _descriptors(self, w, P, L20):
        N = self.win
        P4 = P[N].astype(np.float64)
        T = P4.shape[0]
        Lspec = np.log(P4 + EPS * 1e-2)
        # pitch salience
        sal = Lspec @ self.comb11.T.astype(np.float64)                         # (T, 11)
        salf = Lspec @ self.comb_fine.T.astype(np.float64)
        ib = salf.argmax(1)
        strength = salf[np.arange(T), ib]
        bestf0 = np.log2(self.f0_fine[ib])
        # ACF HNR
        x = (w * self.hann[N]).astype(np.float64)
        Xz = sfft.rfft(x, 2 * N, axis=1, workers=-1)
        r = sfft.irfft(np.abs(Xz) ** 2, axis=1, workers=-1)[:, :N]
        r = r / np.maximum(r[:, :1], 1e-20) / np.maximum(self.rw, 1e-3)
        rmax = np.clip(r[:, self.q_lo:self.q_hi + 1].max(1), 1e-3, 0.999)
        hnr = 10 * np.log10(rmax / (1 - rmax))
        # cepstral peak prominence
        cep = sfft.irfft(Lspec, axis=1, workers=-1)
        cq = cep[:, self.cpp_q]
        ab = cq @ self.cpp_proj.T                                               # (T, 2)
        seg = cep[:, self.q_lo:self.q_hi + 1]
        iq = seg.argmax(1)
        cpp = seg[np.arange(T), iq] - (ab[:, 0] + ab[:, 1] * (iq + self.q_lo))
        # spectral moments, flatness, rolloff
        fr = self.freqs
        band = (fr >= 50) & (fr <= 11025)
        fb = fr[band] / 1000.0
        pb = P4[:, band] + 1e-12
        p = pb / pb.sum(1, keepdims=True)
        cen = p @ fb
        dev = fb[None, :] - cen[:, None]
        var = (p * dev ** 2).sum(1) + 1e-9
        spread = np.sqrt(var)
        skew = (p * dev ** 3).sum(1) / var ** 1.5
        kurt = np.log((p * dev ** 4).sum(1) / var ** 2 + 1e-9)
        b8 = (fr >= 50) & (fr <= 8000)
        lb = np.log(P4[:, b8] + EPS * 1e-2)
        flat = lb.mean(1) - np.log(P4[:, b8].mean(1) + EPS * 1e-2)
        cs = np.cumsum(p, 1)
        roll = np.log2(fb[np.minimum((cs < 0.85).sum(1), len(fb) - 1)] * 1000)
        # octave spectral contrast
        octc = []
        for lo, hi in zip(self.oct_edges[:-1], self.oct_edges[1:]):
            sb = np.sort(Lspec[:, (fr >= lo) & (fr < hi)], 1)
            k = max(1, int(round(0.2 * sb.shape[1])))
            octc.append(sb[:, -k:].mean(1) - sb[:, :k].mean(1))
        octc = np.stack(octc, 1)                                                # (T, 7)
        # channel-group (phonetic band) contrasts on the 20-channel cochleagram
        edges = [0, 300, 1000, 2500, 5000, 1e9]
        gm = L20.mean(1)
        grp = np.stack([L20[:, (self.cf20 >= lo) & (self.cf20 < hi)].mean(1) - gm
                        for lo, hi in zip(edges[:-1], edges[1:])], 1)           # (T, 5)
        # softmax formants on a cepstrally smoothed envelope
        lif = cep.copy()
        lif[:, 40:-40] = 0
        env = sfft.rfft(lif, axis=1, workers=-1).real
        forms = []
        for lo, hi in ((200, 1000), (800, 2800), (1800, 3800)):
            m = (fr >= lo) & (fr <= hi)
            e = 2.0 * env[:, m]
            e = np.exp(e - e.max(1, keepdims=True))
            forms.append(np.log2((e @ fr[m]) / e.sum(1)))
        forms = np.stack(forms, 1)
        # flux, loudness
        flux = np.maximum(L20 - shift(L20, 1), 0).mean(1)
        loud = np.log(np.exp(L20).sum(1) + EPS)
        # voicing-gated cochleagram
        v = 1 / (1 + np.exp(-(strength - 3.0) / 0.75))
        gated = v[:, None] * np.maximum(L20 - np.log(EPS), 0)
        scal = np.column_stack([bestf0, strength, hnr, cpp, cen, spread, skew, kurt, flat, roll,
                                octc, grp, forms, flux, loud])                  # 27 scalars
        base = np.concatenate([sal, scal, gated], 1)                            # 58
        env_feats = [np.abs(tconv(scal, gabor_t(rr))) for rr in (2, 4, 6, 10)]
        D = np.concatenate([base, dyn(base)] + env_feats, 1)
        key = dict(f0=bestf0, strength=strength, hnr=hnr, cpp=cpp, cen=cen, flat=flat,
                   loud=loud, flux=flux, F1=forms[:, 0], F2=forms[:, 1], F3=forms[:, 2], grp=grp)
        return D, key

    # ------------------------------------------------------------------------------------------
    def _interactions(self, key):
        ks = np.column_stack([key[k] for k in ('loud', 'flux', 'f0', 'strength', 'hnr', 'cpp',
                                               'cen', 'flat', 'F1', 'F2', 'F3')]
                             + [key['grp']])                                    # 16
        z = zscore(ks)
        partners = np.column_stack([key[k] for k in ('f0', 'strength', 'loud', 'flux',
                                                     'F1', 'F2', 'F3')])
        zp, zg = zscore(partners), zscore(key['grp'])
        prods = (zg[:, :, None] * zp[:, None, :]).reshape(len(z), -1)           # 35
        zk = dict(zip(('loud', 'flux', 'f0', 'strength', 'hnr', 'cpp', 'cen', 'flat', 'F1', 'F2', 'F3'),
                      z.T))
        extra = np.column_stack([zk[a] * zk[b] for a, b in (
            ('strength', 'loud'), ('flux', 'loud'), ('F1', 'F2'), ('strength', 'flux'),
            ('hnr', 'loud'), ('f0', 'strength'), ('cen', 'loud'), ('flat', 'flux'))])  # 8
        X = zscore(np.concatenate([z, dyn(z), prods, extra], 1))                # 16+48+35+8 = 107
        X = np.clip(X, -6, 6)
        return np.concatenate([np.maximum(X, 0), np.maximum(-X, 0), X ** 3], 1)

    # ------------------------------------------------------------------------------------------
    def _scattering(self, w, chunk=96):
        N = self.win
        T = w.shape[0]
        out = np.empty((T, 9 + 45 + 18 + 4), np.float64)
        win = np.hanning(N + 2)[1:-1].astype(np.float32)
        mid = self.sc_mid
        cent = np.array(self.sc_cent)
        for a in range(0, T, chunk):
            x = w[a:a + chunk] * win
            B = x.shape[0]
            X = sfft.rfft(x, axis=1, workers=-1)                                # (B, F)
            Y = np.zeros((B, 36, N), np.complex64)
            Y[:, :, :N // 2 + 1] = X[:, None, :] * self.psi1[None]
            z = sfft.ifft(Y, axis=2, workers=-1) * 2                            # analytic
            z = z.reshape(B, 9, 4, N)
            amp = np.abs(z)
            U = amp.sum(2)                                                      # (B, 9, N)
            S1 = U[:, :, mid].mean(2) + 1e-6
            Uh = sfft.rfft(U * win, axis=2, workers=-1)                         # (B, 9, F)
            Y2 = np.zeros((B, 9, 5, N), np.complex64)
            Y2[..., :N // 2 + 1] = Uh[:, :, None, :] * self.psi2[None, None]
            S2 = np.abs(sfft.ifft(Y2, axis=3, workers=-1) * 2)[..., mid].mean(3)  # (B, 9, 5)
            zc = z.sum(2)                                                       # (B, 9, N)
            pr = (zc[:, :, 1:] * np.conj(zc[:, :, :-1]))[:, :, mid]
            ps = pr.sum(2)
            inst = np.angle(ps) * SR / (2 * np.pi)
            ifr = np.log2(np.maximum(np.abs(inst), 1.0) / cent)
            coh = np.abs(ps) / (np.abs(pr).sum(2) + 1e-12)
            # broadband: S0 (log energy), analytic-signal IF and phase coherence, mean channel coherence
            zb = zc.sum(1)
            pb = (zb[:, 1:] * np.conj(zb[:, :-1]))[:, mid]
            psb = pb.sum(1)
            s0 = np.log((x[:, mid].astype(np.float64) ** 2).mean(1) + 1e-8)
            ifb = np.log2(np.maximum(np.abs(np.angle(psb) * SR / (2 * np.pi)), 1.0))
            cohb = np.abs(psb) / (np.abs(pb).sum(1) + 1e-12)
            out[a:a + B] = np.concatenate([np.log(S1), np.log(S2 / S1[:, :, None] + 1e-4).reshape(B, -1),
                                           ifr, coh, np.column_stack([s0, ifb, cohb, coh.mean(1)])], 1)
        return out

    # ------------------------------------------------------------------------------------------
    def __call__(self, windows):
        w = np.asarray(windows, dtype=np.float32)
        T, N = w.shape
        assert N == self.win
        # level normalisation by segment RMS (central hop of each window)
        c = w[:, N // 2 - 735:N // 2 + 735]
        rms = float(np.sqrt(np.mean(c.astype(np.float64) ** 2)))
        w = w / (rms if rms > 1e-12 else 1.0)

        P = self._spectra(w)
        coch, L20 = self._cochleagram(P)
        L40 = np.log(P[N] @ self.gt40.T + EPS).astype(np.float64)
        stm = self._stm(L20, L40)
        desc, key = self._descriptors(w, P, L20)
        inter = self._interactions(key)
        scat = self._scattering(w)
        groups = dict(cochleagram=coch, stm=stm, descriptors=desc, interactions=inter, scattering=scat)
        self.group_dims = {k: v.shape[1] for k, v in groups.items()}
        out = np.concatenate(list(groups.values()), 1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1e6, 1e6).astype(np.float32)
