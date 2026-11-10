"""Closed-form acoustic feature space for ECoG high-gamma encoding models.

All parameters are fixed; nothing is learned or fitted.  The only data-dependent
operations are within-segment normalisations (global RMS of the segment, and the
within-segment z-scoring of the interaction block), which the description allows.
See FEATURES.md for the full list of groups and dimensions.
"""
import numpy as np
import scipy.signal as sps
import torch
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 44100.0
FPS = 30.0
DYN_TAUS = (4.5, 15.0, 60.0)
EPS = 1e-6          # power floor relative to segment RMS^2 (~ -60 dB)


# ----------------------------------------------------------------------------- helpers
def lp_causal(x, tau):
    """Causal one-pole lowpass over frames (axis 0), alpha = 1/(1+tau), started at x[0]."""
    a = 1.0 / (1.0 + tau)
    zi = ((1.0 - a) * x[:1])
    return sps.lfilter([a], [1.0, a - 1.0], x, axis=0, zi=zi)[0]


def lp_acausal(x, tau):
    """Zero-phase (forward-backward) one-pole lowpass."""
    y = lp_causal(x, tau)
    return lp_causal(y[::-1], tau)[::-1]


def dyn(x, taus=DYN_TAUS):
    return np.concatenate([x - lp_causal(x, t) for t in taus], axis=1)


def shift(x, k):
    """y[t] = x[t-k] (k>0 past, k<0 future), edge-padded."""
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


def erb(f):
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


def erb_space(lo, hi, n):
    e = lambda f: 21.4 * np.log10(4.37e-3 * f + 1.0)
    ie = lambda E: (10 ** (E / 21.4) - 1.0) / 4.37e-3
    return ie(np.linspace(e(lo), e(hi), n))


def gammatone_weights(nfft, cfs):
    """Power response of 4th-order gammatone filters at rfft bin frequencies, rows sum to 1."""
    f = np.fft.rfftfreq(nfft, 1.0 / SR)
    b = 1.019 * erb(cfs)[:, None]
    w = (1.0 + ((f[None, :] - cfs[:, None]) / b) ** 2) ** -4
    w[w < 1e-5] = 0.0
    return w / w.sum(1, keepdims=True)


def gabor_t(rate, n_t=None):
    """Complex temporal Gabor (frames), zero-mean for rate>0."""
    if rate == 0:
        sig = 0.2 * FPS
    else:
        sig = 0.75 / rate * FPS
    L = int(np.ceil(3 * sig))
    t = np.arange(-L, L + 1)
    env = np.exp(-0.5 * (t / sig) ** 2)
    k = env * np.exp(2j * np.pi * rate / FPS * t)
    if rate > 0:
        k = k - env * (k.sum() / env.sum())
    return k / np.abs(k).sum()


def gabor_c(scale):
    """Complex spectral Gabor (channels), zero-mean for scale!=0."""
    sig = 3.0 if scale == 0 else 0.75 / abs(scale)
    L = int(np.ceil(2.5 * sig))
    c = np.arange(-L, L + 1)
    env = np.exp(-0.5 * (c / sig) ** 2)
    k = env * np.exp(2j * np.pi * scale * c)
    if scale != 0:
        k = k - env * (k.sum() / env.sum())
    return k / np.abs(k).sum()


def conv_axis(x, k, axis):
    """'same' convolution with reflect padding along axis (complex ok)."""
    L = (len(k) - 1) // 2
    pad = [(0, 0)] * x.ndim
    n = x.shape[axis]
    if L >= n:  # very short segments: fall back to edge padding
        pad[axis] = (L, L)
        xp = np.pad(x, pad, mode='edge')
    else:
        pad[axis] = (L, L)
        xp = np.pad(x, pad, mode='reflect')
    shp = [1] * x.ndim
    shp[axis] = len(k)
    return sps.fftconvolve(xp, k.reshape(shp), mode='valid', axes=axis)


def tgabor_env(x, freqs=(2.0, 4.0, 6.0, 10.0)):
    """Magnitude of complex temporal Gabor (sigma = 1/(pi f)) applied to each column."""
    out = []
    for f in freqs:
        sig = FPS / (np.pi * f)
        L = int(np.ceil(3 * sig))
        t = np.arange(-L, L + 1)
        env = np.exp(-0.5 * (t / sig) ** 2)
        k = env * np.exp(2j * np.pi * f / FPS * t)
        k = k - env * (k.sum() / env.sum())
        k = k / np.abs(k).sum()
        out.append(np.abs(conv_axis(x, k, 0)))
    return np.concatenate(out, axis=1)


# ----------------------------------------------------------------------------- feature space
class FeatureSpace:
    win = 4096

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        n = self.win
        # cochleagram filterbanks (20 ch) for 1024/2048/4096 sub-windows
        self.cf20 = erb_space(50.0, 8000.0, 20)
        self.subwins = (1024, 2048, 4096)
        self.W20 = {m: gammatone_weights(m, self.cf20) for m in self.subwins}
        self.hann = {m: np.hanning(m + 2)[1:-1] for m in self.subwins}
        # 40-channel cochleagram (4096)
        self.cf40 = erb_space(50.0, 8000.0, 40)
        self.W40 = gammatone_weights(4096, self.cf40)
        self.freqs = np.fft.rfftfreq(n, 1.0 / SR)
        # pitch: 11 F0s for salience and a fine grid for best F0
        self.f0_11 = 75.0 * 6.0 ** (np.arange(11) / 10.0)
        self.f0_fine = 75.0 * 6.0 ** (np.arange(61) / 60.0)
        # 2-D Gabor (rate Hz, scale cyc/channel) pairs on the 20-ch 4096 cochleagram
        self.stm17 = [(0, .25), (1, 0), (1, .25), (1, -.25),
                      (3, 0), (3, .25), (3, -.25), (3, .5), (3, -.5),
                      (6, 0), (6, .25), (6, -.25), (6, .5), (6, -.5),
                      (12, 0), (12, .25), (12, -.25)]
        self.stm_rates = (0.5, 1, 2, 4, 8, 12)
        self.stm_scales = (0, .25, -.25, .5, -.5)
        # octave bands for spectral contrast
        self.oct_edges = 100.0 * 2.0 ** np.arange(8)   # 100 .. 12800 Hz -> 7 bands
        # scattering: 9 first-order Morlets (Q=4), 5 second-order (Q=1)
        self.scat_fc = 150.0 * (7000.0 / 150.0) ** (np.arange(9) / 8.0)
        self.scat_fm = np.array([20.0, 40.0, 80.0, 160.0, 320.0])

    # ------------------------------------------------------------------ spectra
    def _spectra(self, x):
        """Power spectra of centred 1024/2048/4096 Hann sub-windows (float64)."""
        n = x.shape[1]
        out = {}
        for m in self.subwins:
            s = (n - m) // 2
            seg = x[:, s:s + m] * self.hann[m]
            P = np.abs(np.fft.rfft(seg, axis=1)) ** 2 / np.sum(self.hann[m] ** 2)
            out[m] = P
        return out

    # ------------------------------------------------------------------ cochleagram block
    def _cochleagram_block(self, E20):
        """E20: dict m -> (T,20) power.  Returns features and log cochleagram (T,60)."""
        Ecat = np.concatenate([E20[m] for m in self.subwins], axis=1)
        logE = np.log(Ecat + EPS)
        cub = np.cbrt(Ecat)
        f = [logE]
        f += [logE - lp_causal(logE, t) for t in (1.5, 4.5, 15.0, 60.0, 150.0)]
        f += [lp_causal(logE, 60.0), lp_causal(logE, 150.0),
              lp_causal(cub, 60.0), lp_causal(cub, 150.0)]
        f += [lp_acausal(logE, 60.0)]
        for k in (3, 6, 12):
            dp = logE - shift(logE, k)           # past difference
            df = shift(logE, -k) - logE          # future difference
            f += [dp, df, np.maximum(dp, 0), np.maximum(df, 0)]
        return np.concatenate(f, axis=1), logE

    # ------------------------------------------------------------------ STM blocks
    def _stm17(self, C):
        """2-D Gabor quadrature energy on 20-ch log cochleagram (T,20) -> (T, 2*17*20): energy and acausal residual."""
        C = C - C.mean(0, keepdims=True)
        cache = {}
        out = []
        for r, s in self.stm17:
            if r not in cache:
                cache[r] = conv_axis(C.astype(np.complex128), gabor_t(r), 0)
            z = conv_axis(cache[r], gabor_c(s), 1)
            out.append(np.abs(z))
        M = np.concatenate(out, axis=1)
        return np.concatenate([M, M - lp_acausal(M, 15.0)], axis=1)

    def _stm40(self, C):
        """40-ch: rates x scales, cos/sin/mag averaged over frequency thirds."""
        C = C - C.mean(0, keepdims=True)
        thirds = np.array_split(np.arange(C.shape[1]), 3)
        cs, mags = [], []
        for r in self.stm_rates:
            zt = conv_axis(C.astype(np.complex128), gabor_t(r), 0)
            for s in self.stm_scales:
                z = conv_axis(zt, gabor_c(s), 1)
                for th in thirds:
                    zz = z[:, th]
                    cs.append(zz.real.mean(1))
                    cs.append(zz.imag.mean(1))
                    mags.append(np.abs(zz).mean(1))
        cs = np.stack(cs, 1)                         # (T, 6*5*3*2)
        M = np.stack(mags, 1)                        # (T, 6*5*3), order rate,scale,third
        nR, nS = len(self.stm_rates), len(self.stm_scales)
        M4 = M.reshape(-1, nR, nS, 3)
        cross = np.sqrt(np.maximum(M4[:, :-1] * M4[:, 1:], 0)).reshape(M.shape[0], -1)
        resid = M - lp_acausal(M, 15.0)
        return np.concatenate([cs, M, dyn(M), resid, cross], axis=1)

    # ------------------------------------------------------------------ descriptors
    def _descriptors(self, P, logC40, logC20):
        T = P.shape[0]
        fr = self.freqs
        df = fr[1] - fr[0]
        L = np.log(P + EPS)
        sc = {}
        tot = P[:, (fr >= 50) & (fr <= 11025)].sum(1)
        sc['loud'] = np.log(tot + EPS)

        # --- pitch salience: 8-harmonic comb on log spectrum (peak-picked) minus inter-harmonics
        Lmax = maximum_filter1d(L, 5, axis=1)
        Lsm = uniform_filter1d(L, 5, axis=1)
        hw = 0.85 ** np.arange(8)

        def salience(f0s):
            out = np.empty((T, len(f0s)))
            for i, f0 in enumerate(f0s):
                h = np.arange(1, 9)
                ph = h * f0 / df
                pi = (h - 0.5) * f0 / df
                vp = self._interp(Lmax, ph)
                vi = self._interp(Lsm, pi)
                out[:, i] = (vp - vi) @ hw / hw.sum()
            return out
        sal11 = salience(self.f0_11)
        salf = salience(self.f0_fine)
        best = salf.argmax(1)
        sc['logf0'] = np.log2(self.f0_fine[best])
        sc['pstr'] = salf.max(1)

        # --- ACF / HNR
        r = np.fft.irfft(P, axis=1)[:, :self.win // 2]
        wac = np.fft.irfft(np.abs(np.fft.rfft(self.hann[4096], 2 * 4096)) ** 2)[:self.win // 2]
        r = r / (wac / wac[0])[None, :]
        r = r / (r[:, :1] + 1e-12)
        lo, hi = int(SR / 450), int(SR / 75)
        rmax = np.clip(r[:, lo:hi].max(1), 1e-3, 0.999)
        sc['hnr'] = 10 * np.log10(rmax / (1 - rmax))
        sc['voic'] = rmax
        voicing = np.clip((rmax - 0.3) / 0.5, 0, 1)

        # --- cepstral peak prominence
        cep = np.fft.irfft(L, axis=1)
        qlo, qhi = int(SR / 450), int(SR / 60)
        cq = cep[:, qlo:qhi]
        sc['cpp'] = cq.max(1) - cq.mean(1)

        # --- spectral moments / flatness / rolloff (50 Hz - 11 kHz)
        band = (fr >= 50) & (fr <= 11025)
        Pb, fb = P[:, band] + EPS * 1e-3, fr[band] / 1000.0
        w = Pb / Pb.sum(1, keepdims=True)
        mu = w @ fb
        dev = fb[None, :] - mu[:, None]
        var = (w * dev ** 2).sum(1)
        sd = np.sqrt(var + 1e-12)
        sc['cent'] = np.log(mu)
        sc['spread'] = np.log(sd)
        sc['skew'] = (w * dev ** 3).sum(1) / sd ** 3
        sc['kurt'] = np.log((w * dev ** 4).sum(1) / sd ** 4)
        sc['flat'] = np.log(Pb).mean(1) - np.log(Pb.mean(1))
        cum = np.cumsum(Pb, 1)
        idx = (cum < 0.85 * cum[:, -1:]).sum(1)
        sc['roll'] = np.log(fb[np.minimum(idx, len(fb) - 1)])

        # --- octave spectral contrast (top/bottom 20% of bins)
        octc = []
        for lo_, hi_ in zip(self.oct_edges[:-1], self.oct_edges[1:]):
            Lb = np.sort(L[:, (fr >= lo_) & (fr < hi_)], axis=1)
            q = max(1, int(round(0.2 * Lb.shape[1])))
            octc.append(Lb[:, -q:].mean(1) - Lb[:, :q].mean(1))
        octc = np.stack(octc, 1)

        # --- channel-group spectral contrast (40-ch cochleagram, 5 groups of 8)
        g = logC40.reshape(T, 5, 8)
        grpc = g.max(2) - g.mean(2)

        # --- softmax formants on cepstrally smoothed log spectrum
        cl = cep.copy()
        cl[:, 40:-39] = 0
        Ls = np.fft.rfft(cl, axis=1).real[:, :L.shape[1]]
        form = []
        for lo_, hi_ in ((200, 1000), (700, 2600), (1800, 3600)):
            m = (fr >= lo_) & (fr <= hi_)
            v = Ls[:, m] * 1.5
            v = v - v.max(1, keepdims=True)
            e = np.exp(v)
            p = e / e.sum(1, keepdims=True)
            form.append(np.log(p @ fr[m]))
            form.append((p * Ls[:, m]).sum(1))
        form = np.stack(form, 1)

        # --- spectral flux
        d40 = np.diff(logC40, axis=0, prepend=logC40[:1])
        sc['flux_c'] = np.maximum(d40, 0).mean(1)
        Ln = L[:, band]
        Ln = Ln - Ln.mean(1, keepdims=True)
        sc['flux_s'] = np.sqrt(np.mean(np.maximum(np.diff(Ln, axis=0, prepend=Ln[:1]), 0) ** 2, 1))

        # --- phonetic band contrasts (40-ch log cochleagram)
        cf = self.cf40
        bm = lambda a, b: logC40[:, (cf >= a) & (cf < b)].mean(1)
        low, f1, f2, hi_b, fric = bm(50, 500), bm(300, 1000), bm(1000, 2500), bm(2500, 5000), bm(4000, 8001)
        sc['c_fric_low'] = fric - low
        sc['c_f2_f1'] = f2 - f1
        sc['c_hi_mid'] = hi_b - 0.5 * (f1 + f2)
        sc['c_low_rest'] = low - logC40.mean(1)

        names = list(sc.keys())
        S = np.concatenate([np.stack([sc[k] for k in names], 1), sal11, octc, grpc, form], axis=1)
        gated = voicing[:, None] * np.maximum(logC20 - np.log(EPS), 0)
        base = np.concatenate([S, gated], axis=1)
        Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
        feats = np.concatenate([base, dyn(base), tgabor_env(Sz)], axis=1)

        # --- interactions
        key = np.stack([sc['loud'], sc['flux_c'], sc['pstr'], sc['logf0'], sc['hnr'],
                        sc['cent'], form[:, 0], form[:, 2]], 1)
        contr = np.stack([sc['c_fric_low'], sc['c_f2_f1'], sc['c_hi_mid'], sc['c_low_rest']], 1)
        partners = np.stack([sc['pstr'], sc['logf0'], sc['loud'], sc['flux_c'], form[:, 0], form[:, 2]], 1)
        z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-9)
        cz, pz = z(contr), z(partners)
        prods = (cz[:, :, None] * pz[:, None, :]).reshape(T, -1)
        I = z(np.concatenate([key, dyn(key), prods], axis=1))
        I = np.clip(I, -6, 6)
        inter = np.concatenate([np.maximum(I, 0), np.maximum(-I, 0), I ** 3 / 10.0], axis=1)
        return feats, inter

    @staticmethod
    def _interp(A, pos):
        pos = np.clip(pos, 0, A.shape[1] - 1.001)
        i0 = np.floor(pos).astype(int)
        fr = pos - i0
        return A[:, i0] * (1 - fr) + A[:, i0 + 1] * fr

    # ------------------------------------------------------------------ scattering
    def _scattering(self, x):
        n = self.win
        dev = self.device
        f = torch.fft.fftfreq(n, 1.0 / SR, dtype=torch.float64)
        fc = torch.tensor(self.scat_fc)
        sig1 = fc / (4.0 * 2.3548)
        psi1 = torch.exp(-0.5 * ((f[None] - fc[:, None]) / sig1[:, None]) ** 2)
        psi1[:, f <= 0] = 0
        psi1 = (2 * psi1).to(torch.complex64).to(dev)
        fm = torch.tensor(self.scat_fm)
        sig2 = fm / 2.3548
        psi2 = torch.exp(-0.5 * ((f[None] - fm[:, None]) / sig2[:, None]) ** 2)
        psi2[:, f <= 0] = 0
        psi2 = (2 * psi2).to(torch.complex64).to(dev)
        w = torch.tensor(np.hanning(n), dtype=torch.float32, device=dev)
        w = w / w.sum()
        out = []
        xt = torch.from_numpy(x.astype(np.float32))
        for s in range(0, x.shape[0], 256):
            xb = xt[s:s + 256].to(dev)
            X = torch.fft.fft(xb)
            u1 = torch.fft.ifft(X[:, None, :] * psi1[None])        # (B,9,n)
            a1 = u1.abs()
            S1 = (a1 * w).sum(-1)
            A = torch.fft.fft(a1)
            u2 = torch.fft.ifft(A[:, :, None, :] * psi2[None, None])  # (B,9,5,n)
            S2 = (u2.abs() * w).sum(-1)
            c = (w[1:] * u1[..., 1:] * u1[..., :-1].conj()).sum(-1)
            den = (w[1:] * a1[..., 1:] * a1[..., :-1]).sum(-1)
            coh = c.abs() / (den + 1e-12)
            ifr = torch.angle(c) * SR / (2 * np.pi)
            ifd = torch.log2(torch.clamp(ifr, min=1.0) / fc.to(dev).float())
            B = xb.shape[0]
            out.append(torch.cat([torch.log(S1 + 1e-4),
                                  torch.log((S2 + 1e-5) / (S1[..., None] + 1e-4)).reshape(B, -1),
                                  torch.clamp(ifd, -2, 2), coh], 1).cpu().numpy().astype(np.float64))
        return np.concatenate(out, 0)

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        x = np.asarray(windows, dtype=np.float64)
        T = x.shape[0]
        # within-segment level normalisation (centre 1470 samples of each window ~ the whole segment)
        c = x.shape[1] // 2
        rms = np.sqrt(np.mean(x[:, c - 735:c + 735] ** 2)) + 1e-12
        x = x / rms
        torch.set_num_threads(max(1, torch.get_num_threads()))

        Ps = self._spectra(x)
        E20 = {m: Ps[m] @ self.W20[m].T for m in self.subwins}
        coch, logE = self._cochleagram_block(E20)
        logC20 = logE[:, 40:60]                      # 4096 20-ch log cochleagram
        P = Ps[4096]
        logC40 = np.log(P @ self.W40.T + EPS)

        stm17 = self._stm17(logC20)
        stm40 = self._stm40(logC40)
        desc, inter = self._descriptors(P, logC40, logC20)
        scat = self._scattering(x)
        c40 = np.concatenate([logC40, dyn(logC40)], 1)

        F = np.concatenate([coch, stm17, stm40, desc, inter, scat, dyn(scat), c40], axis=1)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        return F.astype(np.float32)
