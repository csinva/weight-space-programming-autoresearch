"""Interpretable acoustic feature space for predicting ECoG high-gamma (30 Hz frames).

All features are closed-form signal processing with fixed parameters (see FEATURES.md).
The overlapping windows are stitched back into the continuous segment waveform, features are
computed on a 150 Hz sub-frame grid (5 sub-frames per 30 Hz frame) and pooled to the frame rate.
"""
import numpy as np
import torch

SR = 44100
HOP = 1470                      # samples between 30 Hz frames
SUB = 5                         # sub-frames per frame
SHOP = HOP // SUB               # 294 samples -> 150 Hz sub-frame rate
SFR = SR / SHOP                 # 150.0
MARG = 3                        # extra frames computed on each side (leads / edges)


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _tri_bank(centers_hz, nfft):
    """Triangular filters with given center frequencies (edges = neighbouring centers)."""
    freqs = np.arange(nfft // 2 + 1) * SR / nfft
    fb = np.zeros((len(centers_hz) - 2, len(freqs)), np.float64)
    for k in range(1, len(centers_hz) - 1):
        lo, c, hi = centers_hz[k - 1], centers_hz[k], centers_hz[k + 1]
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        fb[k - 1] = np.maximum(0.0, np.minimum(up, dn))
        if fb[k - 1].sum() == 0:            # narrower than one bin: nearest bin
            fb[k - 1, np.argmin(np.abs(freqs - c))] = 1.0
    fb /= fb.sum(1, keepdims=True)
    return fb


def _pool(x, n_frames):
    """x: (N_sub, D) on sub-frame grid with frame f centered at sub-frame SUB*f + SUB//2.
    Returns mean over the SUB sub-frames of each frame -> (n_frames, D)."""
    return x[: n_frames * SUB].reshape(n_frames, SUB, -1).mean(1)


def _ema(x, tau_s, rate):
    """Causal exponential moving average along axis 0 (time)."""
    from scipy.signal import lfilter
    a = np.exp(-1.0 / (tau_s * rate))
    y, _ = lfilter([1 - a], [1, -a], x, axis=0, zi=a * x[:1])       # start at steady state
    return y


class FeatureSpace:
    win = 4096

    def __init__(self, device='cpu'):
        self.device = device
        # --- STFT windows ---
        self.n1 = 1024                                          # 23 ms: spectrogram
        self.n2 = 4096                                          # 93 ms: pitch (resolution)
        self.w1 = np.hanning(self.n1 + 1)[:-1].astype(np.float32)
        self.w2 = np.hanning(self.n2 + 1)[:-1].astype(np.float32)

        # --- mel filterbank: 80 bands, 50 Hz .. 12 kHz ---
        mel_pts = _mel2hz(np.linspace(_hz2mel(50.0), _hz2mel(12000.0), 82))
        self.mel_fb = _tri_bank(mel_pts, self.n1).astype(np.float32)          # (80, 513)

        # --- log-spaced (constant-Q) bank for spectro-temporal modulation: 64 bands, 100 Hz..8 kHz
        self.n_oct = np.log2(8000.0 / 100.0)
        log_pts = 100.0 * 2.0 ** np.linspace(-self.n_oct / 63, self.n_oct * 64 / 63, 66)
        self.n3 = 2048                                          # 46 ms: modulation spectrogram
        self.w3 = np.hanning(self.n3 + 1)[:-1].astype(np.float32)
        self.log_fb = _tri_bank(log_pts, self.n3).astype(np.float32)          # (64, 1025)
        self.ch_per_oct = 63.0 / self.n_oct

        # --- 16 coarse bands for onset / context features: groups of 5 mel bands
        self.coarse = 16

        # --- pitch: sub-harmonic summation on whitened spectrum (n2 window) ---
        f0s = 65.0 * 2.0 ** (np.arange(37) / 12.0)                          # 65..520 Hz, semitone
        self.f0s = f0s
        nb = self.n2 // 2 + 1
        bin_hz = SR / self.n2
        H = np.zeros((nb, len(f0s)), np.float64)
        for j, f0 in enumerate(f0s):
            for h in range(1, 11):
                fb_ = h * f0 / bin_hz
                if fb_ >= nb - 1 or h * f0 > 5000:
                    break
                i0 = int(np.floor(fb_)); fr = fb_ - i0
                w = 0.84 ** (h - 1)
                H[i0, j] += w * (1 - fr); H[i0 + 1, j] += w * fr
        H /= H.sum(0, keepdims=True)
        self.shs = H.astype(np.float32)
        self.pitch_maxbin = int(5200 / bin_hz) + 2

        # --- spectro-temporal modulation filters ---
        self.rates = [2.0, 4.0, 8.0, 16.0, 32.0]               # Hz
        self.scales = [0.25, 0.5, 1.0, 2.0, 4.0]               # cycles / octave
        self.n_stm_bands = 6

    # ------------------------------------------------------------------ helpers
    def _stitch(self, windows):
        T, W = windows.shape
        x = np.empty((T - 1) * HOP + W, np.float32)
        x[: (T - 1) * HOP] = windows[:-1, :HOP].reshape(-1)
        x[(T - 1) * HOP:] = windows[-1]
        return x

    def _stft_power(self, xp, centers, n, win):
        """power spectra of frames of length n centered at `centers` (indices into xp)."""
        starts = centers - n // 2
        idx = starts[:, None] + np.arange(n)[None]
        out = []
        for s in range(0, len(starts), 4096):
            fr = torch.from_numpy(xp[idx[s:s + 4096]] * win)
            sp = torch.fft.rfft(fr.to(self.device), dim=1)
            out.append((sp.real ** 2 + sp.imag ** 2).cpu().numpy())
        return np.concatenate(out, 0)

    @staticmethod
    def _gabor_t(rate):
        sig = 0.5 / rate                                        # seconds
        t = np.arange(-int(np.ceil(3 * sig * SFR)), int(np.ceil(3 * sig * SFR)) + 1) / SFR
        env = np.exp(-0.5 * (t / sig) ** 2)
        k = env * np.exp(2j * np.pi * rate * t)
        k -= env * (k.sum() / env.sum())                        # zero mean
        return k / np.abs(k).sum()

    def _gabor_f(self, scale):
        sig = min(0.5 / scale, 1.5)                             # octaves
        half = int(np.ceil(3 * sig * self.ch_per_oct))
        o = np.arange(-half, half + 1) / self.ch_per_oct
        env = np.exp(-0.5 * (o / sig) ** 2)
        k = env * np.exp(2j * np.pi * scale * o)
        k -= env * (k.sum() / env.sum())
        return k / np.abs(k).sum()

    # ------------------------------------------------------------------ main
    def __call__(self, windows):
        windows = np.asarray(windows, np.float32)
        T, W = windows.shape
        x = self._stitch(windows).astype(np.float64)
        rms = np.sqrt(np.mean(x ** 2)) + 1e-8                   # per-segment gain normalisation
        x = (x / rms).astype(np.float32)
        c0 = W // 2                                             # center of frame 0 in x

        # sub-frame grid: frames -MARG .. T-1+MARG, SUB sub-frames each
        nF = T + 2 * MARG
        nS = nF * SUB
        sub_c = c0 + (np.arange(nS) - MARG * SUB - SUB // 2) * SHOP
        pad = self.n2 // 2 + MARG * HOP + HOP
        xp = np.pad(x, pad, mode='reflect')
        sub_c = sub_c + pad

        eps = 1e-5                                              # power floor (signal has unit RMS)
        P1 = self._stft_power(xp, sub_c, self.n1, self.w1) / self.n1      # (nS, 513)
        P2 = self._stft_power(xp, sub_c, self.n2, self.w2) / self.n2      # (nS, 2049)
        P3 = self._stft_power(xp, sub_c, self.n3, self.w3) / self.n3      # (nS, 1025)

        feats = {}

        # ---------- 1. log-mel spectrogram (80), current frame and 1-frame lead
        melP = P1 @ self.mel_fb.T                                          # (nS, 80)
        melF = np.log(_pool(melP, nF) + eps)                               # pool power, then log
        cur = slice(MARG, MARG + T)
        lead1 = slice(MARG + 1, MARG + 1 + T)
        lead2 = slice(MARG + 2, MARG + 2 + T)
        feats['mel'] = melF[cur]
        feats['mel_lead1'] = melF[lead1]

        # ---------- 2. onsets / offsets in 16 coarse bands (half-wave rectified d/dt log-mel)
        lmel_s = np.log(melP + eps)
        cb = lmel_s.reshape(nS, self.coarse, -1).mean(2)                   # (nS, 16)
        d = np.diff(cb, axis=0, prepend=cb[:1])
        on = _pool(np.maximum(d, 0), nF) * SUB
        off = _pool(np.maximum(-d, 0), nF) * SUB
        feats['onset16'] = on[cur]
        feats['onset16_lead1'] = on[lead1]
        feats['offset16'] = off[cur]

        # ---------- 3. broadband envelope: loudness, peak-rate, sub-frame loudness trajectory
        bb = np.log(melP.sum(1) + eps)                                     # (nS,)
        bbF = np.log(_pool(melP.sum(1, keepdims=True), nF) + eps)[:, 0]
        k = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2); k /= k.sum()
        bb_s = np.convolve(bb, k, mode='same')
        dbb = np.diff(bb_s, prepend=bb_s[0])
        peak = _pool(np.maximum(dbb, 0)[:, None], nF)[:, 0] * SUB
        trough = _pool(np.maximum(-dbb, 0)[:, None], nF)[:, 0] * SUB
        subl = bb[: nF * SUB].reshape(nF, SUB)                              # 5 sub-frame levels
        bbcol = []
        for sl in (cur, lead1, lead2):
            bbcol += [bbF[sl], peak[sl], trough[sl]]
        feats['envelope'] = np.stack(bbcol, 1)                              # 9
        feats['subframe_level'] = subl[cur] - bbF[cur][:, None]             # 5 (relative shape)

        # ---------- 4. pitch / periodicity (sub-harmonic summation)
        mag = np.sqrt(P2[:, : self.pitch_maxbin])
        lm = np.log(mag + 1e-3)
        kw = 15
        ker = np.ones(kw) / kw
        lm_t = torch.from_numpy(lm.astype(np.float32))[:, None]
        smooth = torch.nn.functional.conv1d(
            torch.nn.functional.pad(lm_t, (kw // 2, kw // 2), mode='replicate'),
            torch.from_numpy(ker.astype(np.float32))[None, None])[:, 0].numpy()
        wh = np.maximum(lm - smooth, 0.0)                                   # spectral peaks
        shs_mat = np.zeros((wh.shape[1], len(self.f0s)), np.float32)
        nb = min(wh.shape[1], self.shs.shape[0])
        shs_mat[:nb] = self.shs[:nb]
        sal = wh @ shs_mat                                                  # (nS, 37)
        # suppress sub-octave errors: salience minus salience at f0/2 harmonics already captured;
        # normalise by mean peakiness to get periodicity strength
        base = wh.mean(1, keepdims=True) + 1e-3
        salr = sal / base                                                   # ~1 for aperiodic
        level_gate = 1.0 / (1.0 + np.exp(-(bb - np.log(1e-3)) * 1.0))      # silence -> 0
        salr = salr * level_gate[:, None]
        salF = _pool(salr, nF)
        voic = salr.max(1)
        f0i = salr.argmax(1)
        lf0 = np.log2(self.f0s[f0i] / 65.0)                                 # octaves above 65 Hz
        voicF = _pool(voic[:, None], nF)[:, 0]
        # pitch distribution (salience), 37 semitone bins
        feats['pitch_salience'] = salF[cur]
        # voicing, voicing-weighted log f0, lead copies
        vf = np.stack([voicF, _pool((lf0 * voic)[:, None], nF)[:, 0]], 1)
        feats['voicing'] = np.concatenate([vf[cur], vf[lead1]], 1)          # 4
        # pitch change: soft-binned delta log f0 between frames, gated by voicing of both
        lf0F = _pool((lf0 * voic)[:, None], nF)[:, 0] / (voicF + 1e-6)
        dlf = np.diff(lf0F, prepend=lf0F[0]) * 12.0                         # semitones / frame
        g = np.sqrt(np.maximum(voicF, 0) * np.maximum(np.roll(voicF, 1), 0))
        centers = np.array([-6, -3, -1.5, -0.5, 0, 0.5, 1.5, 3, 6], float)
        soft = np.exp(-0.5 * ((dlf[:, None] - centers[None]) / 0.75) ** 2)
        soft /= soft.sum(1, keepdims=True) + 1e-9
        feats['pitch_change'] = (soft * g[:, None])[cur]                    # 9

        # ---------- 5. spectro-temporal modulation (Gabor, constant-Q 64-band log spectrogram)
        Slog = np.log(P3 @ self.log_fb.T + eps).astype(np.float32)          # (nS, 64)
        Sl = torch.from_numpy(Slog.T.copy())[None].to(self.device)          # (1, 64, nS)
        stm_list, tm_list = [], []
        band_edges = np.linspace(0, 64, self.n_stm_bands + 1).astype(int)
        for r in self.rates:
            kt = self._gabor_t(r)
            kt_t = torch.from_numpy(np.stack([kt.real, kt.imag]).astype(np.float32))[:, None, None]
            ph = kt.shape[0] // 2
            Sp = torch.nn.functional.pad(Sl[:, None], (ph, ph, 0, 0), mode='reflect')  # (1,1,64,nS+)
            Y = torch.nn.functional.conv2d(Sp, kt_t.flip(-1).reshape(2, 1, 1, -1))[0]  # (2,64,nS)
            Yc = torch.complex(Y[0], Y[1])                                  # temporal analytic
            # pure temporal modulation per coarse band (16 bands)
            tm = Yc.abs().reshape(16, 4, -1).mean(1)
            tm_list.append(tm.T.cpu().numpy())
            for s in self.scales:
                kf = self._gabor_f(s)
                pf = kf.shape[0] // 2
                Yr = torch.nn.functional.pad(Y.permute(0, 2, 1), (pf, pf), mode='reflect')  # (2,nS,64+)
                kf_t = torch.from_numpy(np.stack([kf.real, kf.imag]).astype(np.float32))
                kf_t = kf_t.flip(-1).to(self.device)
                rr = torch.nn.functional.conv1d(Yr.reshape(-1, 1, Yr.shape[-1]), kf_t[:, None])
                rr = rr.reshape(2, -1, 2, 64)                               # (re/im Y, nS, re/im kf, 64)
                # (a+ib)(c+id) and (a+ib)(c-id): two sweep directions
                re_u = rr[0, :, 0] - rr[1, :, 1]; im_u = rr[0, :, 1] + rr[1, :, 0]
                re_d = rr[0, :, 0] + rr[1, :, 1]; im_d = rr[1, :, 0] - rr[0, :, 1]
                for re, im in ((re_u, im_u), (re_d, im_d)):
                    m = torch.sqrt(re ** 2 + im ** 2).cpu().numpy()        # (nS, 64)
                    stm_list.append(np.stack([m[:, a:b].mean(1) for a, b in
                                              zip(band_edges[:-1], band_edges[1:])], 1))
        stm = np.sqrt(_pool(np.concatenate(stm_list, 1), nF))               # 5*5*2*6 = 300
        tmod = np.sqrt(_pool(np.concatenate(tm_list, 1), nF))               # 5*16 = 80
        feats['stm'] = stm[cur]
        feats['temporal_mod'] = tmod[cur]

        # ---------- 6. spectral shape descriptors
        fr = np.arange(P1.shape[1]) * SR / self.n1
        Pn = P1[:, 1:] + 1e-10
        Ps = Pn.sum(1)
        cen = (Pn * fr[1:]).sum(1) / Ps
        bw = np.sqrt((Pn * (fr[1:] - cen[:, None]) ** 2).sum(1) / Ps)
        flat = np.exp(np.log(Pn).mean(1)) / Pn.mean(1)
        cum = np.cumsum(Pn, 1) / Ps[:, None]
        roll = fr[1:][np.argmax(cum >= 0.85, 1)]
        lowr = np.log((P1[:, (fr > 80) & (fr < 1000)].sum(1) + eps) /
                      (P1[:, (fr >= 1000) & (fr < 8000)].sum(1) + eps))
        sd = np.stack([np.log(cen + 1), np.log(bw + 1), np.log(flat + 1e-6),
                       np.log(roll + 1), lowr], 1)
        # spectral flux (L2 of positive log-mel change)
        flux = np.sqrt((np.maximum(np.diff(lmel_s, axis=0, prepend=lmel_s[:1]), 0) ** 2).mean(1))
        sd = np.concatenate([sd, flux[:, None]], 1)
        feats['spectral_shape'] = _pool(sd, nF)[cur]                         # 6

        # ---------- 7. temporal context: causal EMAs (adaptation / recent history)
        cbF = np.log(_pool(melP.reshape(nS, self.coarse, -1).sum(2), nF) + eps)  # (nF, 16)
        ctx = []
        for tau in (0.3, 1.0, 3.0):
            ctx.append(_ema(cbF, tau, 30.0))
        on_tot = on.sum(1, keepdims=True)
        for tau in (0.3, 1.0):
            ctx.append(_ema(on_tot, tau, 30.0))
            ctx.append(_ema(voicF[:, None], tau, 30.0))
        feats['context'] = np.concatenate(ctx, 1)[cur]                      # 16*3 + 4 = 52

        out = np.concatenate([np.asarray(v, np.float64).reshape(T, -1) for v in feats.values()], 1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_groups = {k: np.asarray(v).reshape(T, -1).shape[1] for k, v in feats.items()}
        return out.astype(np.float32)
