"""Closed-form acoustic feature space for predicting ECoG high-gamma (30 Hz frames).

All parameters are fixed; nothing is learned or fitted. Because neighbouring windows
(win=4096, hop=1470) overlap, the continuous segment waveform is reconstructed exactly and
all features are computed on it at fine time resolution, then pooled onto the 30 Hz frame
grid with a short Hann kernel (anti-aliasing). See FEATURES.md for the full list.
"""
import math
import numpy as np
import torch

SR = 44100
HOP = 1470                 # samples between output frames (1/30 s)
WIN = 4096
H1 = 245                   # fine hop (HOP / 6) -> 180 Hz
H2 = 490                   # medium hop (HOP / 3) -> 90 Hz
OFF = 88                   # start offset so fine frame centers hit frame centers (2048 = 88 + 8*245)


# ----------------------------------------------------------------------------- helpers
def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _tri_bank(centers_hz, n_fft):
    """Triangular filters at given centre frequencies (edges = neighbouring centres), area-normalised.
    Filters narrower than the bin spacing fall back to linear interpolation onto the nearest bins."""
    freqs = np.arange(n_fft // 2 + 1) * SR / n_fft
    df = SR / n_fft
    fb = np.zeros((len(centers_hz) - 2, len(freqs)))
    for b in range(1, len(centers_hz) - 1):
        lo, c, hi = centers_hz[b - 1], centers_hz[b], centers_hz[b + 1]
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        w = np.maximum(0.0, np.minimum(up, dn))
        if w.sum() < 1.0:          # too narrow: linear interpolation at the centre frequency
            w = np.maximum(0.0, 1.0 - np.abs(freqs - c) / df) + w
        fb[b - 1] = w / w.sum()
    return fb


def mel_bank(n_bands, fmin, fmax, n_fft):
    m = np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_bands + 2)
    return _tri_bank(_mel2hz(m), n_fft)


def log_bank(n_bands, fmin, fmax, n_fft):
    c = fmin * (fmax / fmin) ** (np.arange(n_bands + 2) / (n_bands + 1))
    return _tri_bank(c, n_fft)


def hann_kernel(half):
    n = torch.arange(-half, half + 1, dtype=torch.float64)
    w = 0.5 + 0.5 * torch.cos(math.pi * n / half)
    return (w / w.sum()).float()


def pool(feat, step, first, T, half):
    """feat: (N, D) on fine grid. Smooth along time with a Hann kernel (+-half samples, zero at ends)
    and sample at indices first + step*i (i < T). Indices beyond the edge are clamped."""
    k = hann_kernel(half).to(feat.device)
    x = feat.T.unsqueeze(1)                                         # (D, 1, N)
    x = torch.nn.functional.pad(x, (half, half), mode='replicate')
    y = torch.nn.functional.conv1d(x, k.view(1, 1, -1)).squeeze(1)  # (D, N)
    idx = (first + step * torch.arange(T, device=feat.device)).clamp(max=y.shape[1] - 1)
    return y[:, idx].T


def shift(F, s):
    """Row i of the output = row i+s of F (s>0: future frames), edge-replicated."""
    T = F.shape[0]
    idx = (torch.arange(T, device=F.device) + s).clamp(0, T - 1)
    return F[idx]


def safe_log(P, rel=1e-3):
    """log power with a per-band floor at -30 dB re the band's segment mean (gain invariant)."""
    eps = rel * P.mean(0, keepdim=True) + 1e-10 * P.mean() + 1e-20
    return torch.log(P + eps)


def ema(x, tau_frames):
    """Causal exponential moving average along dim 0 (closed-form IIR)."""
    a = math.exp(-1.0 / tau_frames)
    out = torch.empty_like(x)
    # vectorised via cumulative sums in float64 for stability on blocks
    xn = x.double()
    N = x.shape[0]
    B = 512
    acc = xn[0].clone()
    for s in range(0, N, B):
        blk = xn[s:s + B]
        n = blk.shape[0]
        p = a ** torch.arange(n, dtype=torch.float64, device=x.device)
        # y[t] = a^(t+1) acc + (1-a) sum_{u<=t} a^(t-u) x[u]
        c = torch.cumsum(blk / p.unsqueeze(1), 0) * p.unsqueeze(1)
        y = (a * p).unsqueeze(1) * acc + (1 - a) * c
        out[s:s + n] = y.float()
        acc = y[-1]
    return out


def soft_bins(v, centers, width, weight=None):
    """Gaussian soft one-hot encoding of scalar v (N,) onto bins (K,)."""
    d = (v.unsqueeze(1) - centers.unsqueeze(0)) / width
    out = torch.exp(-0.5 * d * d)
    if weight is not None:
        out = out * weight.unsqueeze(1)
    return out


# ----------------------------------------------------------------------------- feature space
class FeatureSpace:
    win = WIN

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        dev = self.device
        self.mel80 = torch.tensor(mel_bank(80, 40.0, 16000.0, 2048), dtype=torch.float32, device=dev)
        self.mel32 = torch.tensor(mel_bank(32, 80.0, 16000.0, 1024), dtype=torch.float32, device=dev)
        self.log64 = torch.tensor(log_bank(64, 100.0, 12800.0, 2048), dtype=torch.float32, device=dev)
        self.win2048 = torch.hann_window(2048, device=dev)
        self.win1024 = torch.hann_window(1024, device=dev)
        self.freqs2048 = torch.arange(1025, device=dev, dtype=torch.float32) * SR / 2048
        # pitch: lags for F0 in [55, 800] Hz; 36 log-spaced salience bins
        self.f0_min, self.f0_max = 55.0, 800.0
        self.sal_edges = np.geomspace(self.f0_min, self.f0_max, 37)
        self.stm_rates = [2.0, 4.0, 8.0, 16.0, 32.0]          # Hz
        self.stm_scales = [0.25, 0.5, 1.0, 2.0, 4.0]          # cycles / octave

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        w = np.asarray(windows, dtype=np.float32)
        T, win = w.shape
        with torch.no_grad():
            x = self._reconstruct(w)
            F = self._features(x, T)
        F = F.cpu().numpy().astype(np.float64)
        F[~np.isfinite(F)] = 0.0
        # tiny fixed dither so that no column is exactly constant (z-scoring safety)
        rng = np.random.default_rng(0)
        sc = np.abs(F).max(0, keepdims=True) + 1e-3
        F = F + 1e-6 * sc * rng.standard_normal(F.shape)
        return F.astype(np.float32)

    def _reconstruct(self, w):
        T, win = w.shape
        if win == WIN:
            x = np.concatenate([w[:, :HOP].ravel(), w[-1, HOP:]])
        else:  # generic fallback (not used with win=4096)
            x = np.zeros((T - 1) * HOP + win, np.float32)
            for i in range(T):
                x[i * HOP:i * HOP + win] = w[i]
            pad = (WIN - win) // 2
            x = np.pad(x, (pad, WIN - win - pad))
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device)

    def _stft_pow(self, x, n_fft, hop, window, win_length=None):
        S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window,
                       center=True, pad_mode='reflect', return_complex=True)
        return S.abs().pow(2).T  # (N, bins)

    def _features(self, x, T):
        dev = self.device
        feats = []
        xs = x[OFF:]
        xs = xs - xs.mean()

        # ---------------- spectra
        P2 = self._stft_pow(xs, 2048, H1, self.win2048)              # (N1, 1025) 46 ms, 180 Hz
        P1 = self._stft_pow(xs, 1024, H1, self.win1024)              # (N1, 513)  23 ms, 180 Hz
        N1 = P2.shape[0]

        # 1. log-mel spectrogram, 80 bands (energy pooled per frame, then log)
        M80 = P2 @ self.mel80.T
        M80f = pool(M80, 6, 8, T, 6)
        feats.append(safe_log(M80f))                                  # 80

        # 32-band log-mel at fine resolution (23 ms window)
        M32 = P1 @ self.mel32.T
        L32 = safe_log(M32)                                            # (N1, 32)

        # 2. look-ahead copies (short-latency responses): 32-band log-mel at +1, +2 frames
        L32f = safe_log(pool(M32, 6, 8, T, 6))
        feats.append(shift(L32f, 1))                                   # 32
        feats.append(shift(L32f, 2))                                   # 32

        # 3. spectral onsets / offsets (half-wave rectified temporal derivative of log-mel, 11 ms step)
        L32s = torch.nn.functional.avg_pool1d(L32.T.unsqueeze(0), 3, 1, 1, count_include_pad=False)[0].T
        d = torch.zeros_like(L32s)
        d[2:] = L32s[2:] - L32s[:-2]
        on = pool(d.clamp(min=0), 6, 8, T, 6)
        off = pool((-d).clamp(min=0), 6, 8, T, 6)
        feats += [on, off, shift(on, 1)]                               # 96

        # 4. broadband envelope dynamics
        E = M32.sum(1)
        LE = safe_log(E.unsqueeze(1))[:, 0]                            # fine log energy
        # loudness envelope low-passed (~10 Hz) and its derivative
        k10 = hann_kernel(9).to(dev)                                  # ~100 ms Hann at 180 Hz
        LEs = torch.nn.functional.conv1d(
            torch.nn.functional.pad(LE.view(1, 1, -1), (9, 9), mode='replicate'), k10.view(1, 1, -1)).view(-1)
        dE = torch.zeros_like(LEs)
        dE[1:] = (LEs[1:] - LEs[:-1]) * (SR / H1)
        dEp = dE.clamp(min=0)
        # peakRate: local maxima of the positive envelope derivative
        pk = torch.zeros_like(dEp)
        pk[1:-1] = dEp[1:-1] * ((dEp[1:-1] > dEp[:-2]) & (dEp[1:-1] >= dEp[2:])).float()
        pk_frames = pool(pk.unsqueeze(1), 6, 8, T, 6) * 13.0
        LEf = pool(LE.unsqueeze(1), 6, 8, T, 6)
        # within-frame max / min / std of fine log energy (+-1 frame)
        idx = 8 + 6 * torch.arange(T, device=dev)
        offs = torch.arange(-5, 6, device=dev)
        J = (idx.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, N1 - 1)
        LEw = LE[J]
        feats += [LEf, pool(dEp.unsqueeze(1), 6, 8, T, 6), pool((-dE).clamp(min=0).unsqueeze(1), 6, 8, T, 6),
                  pk_frames, shift(pk_frames, 1),
                  LEw.max(1, keepdim=True).values, LEw.min(1, keepdim=True).values,
                  LEw.std(1, keepdim=True)]                            # 8

        # 5. loudness level, soft-binned in dB re the segment's 99th percentile
        LdB = 10 * torch.log10(pool(E.unsqueeze(1), 6, 8, T, 6)[:, 0] + 1e-20)
        ref = torch.quantile(LdB.float(), 0.99)
        lv = (LdB - ref).clamp(-60, 5)
        feats.append(soft_bins(lv, torch.arange(-45.0, 1.0, 5.0, device=dev), 3.0))  # 10

        # 6. slow context: causal exponential averages of 32-band log-mel (tau 0.25 s, 1 s)
        L32c = L32f
        feats.append(ema(L32c, 0.25 * 30))                             # 32
        feats.append(ema(L32c, 1.0 * 30))                              # 32

        # 7. syllabic-rate envelope (2-8 Hz band-pass of log energy) amplitude and phase
        feats.append(self._syllabic(LE, T))                            # 3

        # 8. spectral shape descriptors (from 2048-pt spectrum, pooled)
        feats.append(self._shape(P2, xs, T))                           # 7

        # 9. pitch / periodicity
        feats.append(self._pitch(xs, T))                               # 80

        # 10. spectro-temporal modulation energy (Gabor filter bank on log-frequency spectrogram)
        feats.append(self._stm(xs, T))                                 # 480

        return torch.cat([f.float() for f in feats], 1)

    # ------------------------------------------------------------------ groups
    def _syllabic(self, LE, T):
        dev = LE.device
        N = LE.shape[0]
        n = 1 << int(math.ceil(math.log2(N + 2 * 360)))
        y = torch.nn.functional.pad(LE.view(1, 1, -1), (360, 360), mode='reflect').view(-1).double()
        y = y - y.mean()
        Y = torch.fft.fft(y, n=n)
        f = torch.fft.fftfreq(n, d=H1 / SR).to(dev)
        # smooth band-pass 2-8 Hz (raised-cosine edges), analytic (positive freqs only)
        af = f.abs()
        g = torch.zeros_like(af)
        g[(af >= 2.0) & (af <= 8.0)] = 1.0
        lo = (af > 1.0) & (af < 2.0); g[lo] = 0.5 - 0.5 * torch.cos(math.pi * (af[lo] - 1.0))
        hi = (af > 8.0) & (af < 12.0); g[hi] = 0.5 + 0.5 * torch.cos(math.pi * (af[hi] - 8.0) / 4.0)
        g = g * (f > 0).double() * 2.0
        z = torch.fft.ifft(Y * g)[360:360 + N]
        idx = (8 + 6 * torch.arange(T, device=dev)).clamp(max=N - 1)
        z = z[idx]
        amp = z.abs()
        ph = torch.angle(z)
        return torch.stack([amp, torch.cos(ph), torch.sin(ph)], 1).float()

    def _shape(self, P2, xs, T):
        dev = P2.device
        fr = self.freqs2048
        sel = (fr >= 50) & (fr <= 16000)
        P = P2[:, sel] + 1e-12 * P2.mean() + 1e-20
        f = fr[sel]
        tot = P.sum(1, keepdim=True)
        p = P / tot
        cen = (p * f).sum(1)
        bw = torch.sqrt((p * (f - cen.unsqueeze(1)) ** 2).sum(1))
        flat = torch.log(P).mean(1) - torch.log(P.mean(1))
        cs = torch.cumsum(p, 1)
        roll = f[(cs < 0.85).sum(1).clamp(max=len(f) - 1)]
        ent = -(p * torch.log(p + 1e-20)).sum(1) / math.log(P.shape[1])
        hl = torch.log(P[:, f >= 2000].sum(1) + 1e-20) - torch.log(P[:, f < 2000].sum(1) + 1e-20)
        S = torch.stack([torch.log(cen), torch.log(bw + 1.0), flat, torch.log(roll + 50.0), ent, hl], 1)
        S = pool(S, 6, 8, T, 6)
        # zero-crossing rate per output frame (+-1470 samples around the centre)
        sg = torch.signbit(xs).float()
        zc = (sg[1:] != sg[:-1]).float()
        cz = torch.cat([torch.zeros(1, device=dev), torch.cumsum(zc.double(), 0).float()])
        c = 1960 + HOP * torch.arange(T, device=dev)
        a = (c - 735).clamp(0, len(cz) - 1); b = (c + 735).clamp(0, len(cz) - 1)
        zcr = (cz[b] - cz[a]) / (b - a).clamp(min=1).float()
        return torch.cat([S, torch.log(zcr + 1e-4).unsqueeze(1)], 1)

    def _pitch(self, xs, T):
        """Normalised autocorrelation (Boersma 1993) on 46 ms frames at 90 Hz, from the
        power spectrum (low-passed at 5 kHz), via zero-padded FFT."""
        dev = xs.device
        nfft = 4096
        win = self.win2048
        S = torch.stft(xs, n_fft=nfft, hop_length=H2, win_length=2048, window=win, center=True,
                       pad_mode='reflect', return_complex=True)
        P = S.abs().pow(2)                                        # (2049, N2)
        fr = torch.arange(P.shape[0], device=dev) * SR / nfft
        P = P * (fr <= 5000).float().unsqueeze(1)
        r = torch.fft.irfft(P, n=nfft, dim=0)[:900].T            # (N2, 900) lags 0..899
        # window autocorrelation for normalisation
        wpad = torch.zeros(nfft, device=dev); wpad[:2048] = win
        rw = torch.fft.irfft(torch.fft.rfft(wpad).abs().pow(2), n=nfft)[:900]
        rn = r / (r[:, :1] + 1e-9 * r[:, 0].mean() + 1e-20) / (rw / rw[0]).unsqueeze(0)
        lag_min = int(SR / self.f0_max); lag_max = int(SR / self.f0_min)
        lags = torch.arange(lag_min, lag_max + 1, device=dev)
        rr = rn[:, lag_min:lag_max + 1]
        # salience bins: max normalised autocorrelation within each log-F0 bin
        sal = []
        for k in range(len(self.sal_edges) - 1):
            l_hi = int(math.ceil(SR / self.sal_edges[k])); l_lo = int(math.floor(SR / self.sal_edges[k + 1]))
            l_lo = max(l_lo, lag_min); l_hi = min(l_hi, lag_max)
            sal.append(rr[:, l_lo - lag_min:l_hi - lag_min + 1].max(1).values)
        sal = torch.stack(sal, 1).clamp(-1, 1)                     # (N2, 36)
        # best peak with a mild preference for shorter lags (reduces octave-down errors)
        score = rr * (1.0 - 0.1 * torch.log2(lags.float() / lag_min) / math.log2(lag_max / lag_min))
        best = score.argmax(1)
        rmax = rr.gather(1, best.unsqueeze(1))[:, 0].clamp(0, 1)
        # parabolic interpolation of the lag
        b = (best + lag_min).clamp(lag_min + 1, lag_max - 1)
        y0 = rn.gather(1, (b - 1).unsqueeze(1))[:, 0]; y1 = rn.gather(1, b.unsqueeze(1))[:, 0]
        y2 = rn.gather(1, (b + 1).unsqueeze(1))[:, 0]
        den = (y0 - 2 * y1 + y2)
        dl = torch.where(den.abs() > 1e-9, 0.5 * (y0 - y2) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        logf0 = torch.log2(SR / (b.float() + dl))
        # loudness gate: frames > 50 dB below the segment's 99th-pct energy are unvoiced
        e = r[:, 0]
        edb = 10 * torch.log10(e + 1e-20)
        gate = (edb > torch.quantile(edb, 0.99) - 50).float()
        voic = ((rmax - 0.45) / 0.45).clamp(0, 1) * gate              # voicing weight 0..1
        # absolute pitch: 20 soft bins, 70-450 Hz (log2)
        c_abs = torch.linspace(math.log2(70), math.log2(450), 20, device=dev)
        absb = soft_bins(logf0, c_abs, (c_abs[1] - c_abs[0]).item(), voic)
        # relative pitch: log F0 z-scored against a voicing-weighted Gaussian window (sigma 2 s)
        sig = 2.0 * SR / H2
        half = int(3 * sig)
        g = torch.exp(-0.5 * (torch.arange(-half, half + 1, device=dev).float() / sig) ** 2)
        def wsmooth(v):
            vp = torch.nn.functional.pad(v.view(1, 1, -1), (half, half))
            return torch.nn.functional.conv1d(vp, g.view(1, 1, -1)).view(-1)
        wsum = wsmooth(voic) + 1e-6
        mu = wsmooth(voic * logf0) / wsum
        var = wsmooth(voic * (logf0 - mu) ** 2) / wsum
        sd = torch.sqrt(var + 0.15 ** 2)                                # floor ~ 2 semitones
        rel = ((logf0 - mu) / sd).clamp(-3, 3)
        relb = soft_bins(rel, torch.linspace(-2.5, 2.5, 12, device=dev), 5.0 / 11, voic)
        # pitch change (semitones per 11 ms frame), both frames voiced
        dlf = torch.zeros_like(logf0); dlf[1:] = (logf0[1:] - logf0[:-1]) * 12
        vv = torch.zeros_like(voic); vv[1:] = torch.minimum(voic[1:], voic[:-1])
        dlf = dlf.clamp(-2, 2)
        chb = soft_bins(dlf, torch.linspace(-1.2, 1.2, 8, device=dev), 2.4 / 7, vv)
        G = torch.cat([sal, absb, relb, chb, voic.unsqueeze(1), rmax.unsqueeze(1),
                       (logf0 * voic).unsqueeze(1), (rel * voic).unsqueeze(1)], 1)   # 36+20+12+8+4 = 80
        return pool(G, 3, 4, T, 3)

    def _stm(self, xs, T):
        """Spectro-temporal modulation energy: |complex 2-D Gabor| on a 64-band log-frequency
        log-spectrogram (100-12800 Hz, 9.14 ch/oct) at 90 Hz, pooled in 8 frequency regions."""
        dev = xs.device
        P = self._stft_pow(xs, 2048, H2, self.win2048)          # (N2, 1025)
        L = safe_log(P @ self.log64.T)                           # (N2, 64)
        L = L - L.mean()
        N2, Fb = L.shape
        cpo = (Fb + 1) / 7.0                                       # channels per octave
        fs = SR / H2
        padT = int(1.0 * fs)
        nT = 1 << int(math.ceil(math.log2(N2 + 2 * padT)))
        nF = 128
        Lp = torch.zeros(nT, nF, device=dev)
        Lp[padT:padT + N2, :Fb] = L
        Lp[:padT, :Fb] = L[0]; Lp[padT + N2:, :Fb] = L[-1]
        # reflect spectrally into the padding to limit edge effects
        Lp[:, Fb:Fb + 32] = Lp[:, Fb - 32:Fb].flip(1)
        Lp[:, nF - 32:] = Lp[:, 1:33].flip(1)
        LF = torch.fft.fft2(Lp)
        t = torch.fft.fftfreq(nT, d=1.0 / nT, device=dev)          # integer offsets (circular)
        t = torch.where(t > nT / 2, t - nT, t) / fs                # seconds
        q = torch.fft.fftfreq(nF, d=1.0 / nF, device=dev) / cpo     # octaves
        tt, qq = t.view(-1, 1), q.view(1, -1)
        specs = []
        for rate in [0.0] + self.stm_rates:
            for scale in [0.0] + self.stm_scales:
                if rate == 0.0 and scale == 0.0:
                    continue
                for sgn in ([1.0, -1.0] if (rate > 0 and scale > 0) else [1.0]):
                    st = 0.6 / rate if rate > 0 else 0.03
                    ss = 0.6 / scale if scale > 0 else 0.4
                    env = torch.exp(-0.5 * (tt / st) ** 2 - 0.5 * (qq / ss) ** 2)
                    car = torch.exp(1j * 2 * math.pi * (rate * tt + sgn * scale * qq))
                    gk = env * car
                    gk = gk - env * (gk.sum() / env.sum())             # zero mean
                    gk = gk / env.sum()
                    specs.append(gk)
        outs = []
        idx = (padT + 4 + 3 * torch.arange(T, device=dev)).clamp(max=padT + N2 - 1)
        kern = hann_kernel(3).to(dev)
        for gk in specs:
            y = torch.fft.ifft2(LF * torch.fft.fft2(gk)).abs()[:, :Fb]   # (nT, 64)
            # temporal pooling (+-1 frame) then sample at frame centres
            yy = torch.nn.functional.conv1d(y.T.unsqueeze(1), kern.view(1, 1, -1), padding=3).squeeze(1).T
            yy = yy[idx]                                             # (T, 64)
            outs.append(yy.view(T, 8, 8).mean(2))                    # 8 regions
        return torch.log(torch.cat(outs, 1) + 1e-3)                  # 60 filters x 8 = 480
