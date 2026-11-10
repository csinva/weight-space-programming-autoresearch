"""Textbook acoustic feature space for auditory ECoG high-gamma prediction.

Closed-form signal processing only. All parameters are fixed; nothing is
learned or fit from data. Output is a (T, H) float32 array.
"""
import numpy as np
from scipy.signal import windows as sw_windows


EPS = 1e-10


class FeatureSpace:
    win = 2048  # 46.4 ms analysis window at 44.1 kHz

    def __init__(self, device='cpu'):
        self.device = device
        self.sr = 44100.0
        self.hop = 1470
        self.frame_sr = self.sr / self.hop  # 30 Hz
        self.n_fft = self.win
        self.n_freq = self.win // 2 + 1  # 1025
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr).astype(np.float32)
        self.window = sw_windows.hann(self.win, sym=False).astype(np.float32)

        # Pre-computed filter banks
        self.mel80_fb = self._mel_filterbank(80, 0.0, self.sr / 2.0)
        self.mel40_fb = self._mel_filterbank(40, 0.0, self.sr / 2.0)
        self.erb64_fb = self._erb_filterbank(64, 50.0, min(16000.0, self.sr / 2.0))
        self.bark_fb = self._bark_filterbank()
        self.oct_fb, self.oct_cfs = self._third_octave_filterbank()
        self.chroma_fb = self._chroma_filterbank(12, fmin=55.0)
        self.a_weight_lin = self._a_weighting_lin()  # (F,)

        # DCT-II matrix for MFCCs (ortho, HTK-style on 40-band log-mel)
        self.dct_mat = self._dct_matrix(40, 13)

        # Spectral-contrast band edges (octave, Jiang 2002)
        edges = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, self.sr / 2.0]
        self.contrast_masks = [
            (self.freqs >= edges[i]) & (self.freqs < edges[i + 1])
            for i in range(len(edges) - 1)
        ]
        self.n_contrast = len(self.contrast_masks)

        # Temporal-modulation rates (Hz). Frame rate is 30 Hz (Nyquist 15 Hz)
        self.mod_rates_erb = [1.0, 2.0, 4.0, 8.0]
        self.mod_rates_mel = [2.0, 4.0, 8.0]

        # Spectrotemporal modulation (Chi / Shamma 2005) parameters
        self.stm_rates_hz = [2.0, 4.0, 8.0]              # temporal rate
        self.stm_scales_cpb = [1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0]  # cyc/band (~1,2,4 cyc/oct on 40-mel)

    # ---- filter bank constructors ----

    @staticmethod
    def _hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    def _mel_filterbank(self, n_mels, fmin, fmax):
        # Slaney-style triangular filter bank with area-normalized weights
        mel_pts = np.linspace(self._hz_to_mel(fmin), self._hz_to_mel(fmax), n_mels + 2)
        hz_pts = self._mel_to_hz(mel_pts)
        fb = np.zeros((n_mels, self.n_freq), dtype=np.float32)
        for m in range(n_mels):
            lo, cen, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            left = (self.freqs - lo) / max(cen - lo, 1e-6)
            right = (hi - self.freqs) / max(hi - cen, 1e-6)
            tri = np.maximum(0.0, np.minimum(left, right))
            # Slaney normalization (constant energy per filter)
            tri *= 2.0 / max(hi - lo, 1e-6)
            fb[m] = tri
        return fb.astype(np.float32)

    @staticmethod
    def _hz_to_erb(f):
        # Glasberg & Moore 1990
        return 21.4 * np.log10(1.0 + 0.00437 * f)

    @staticmethod
    def _erb_to_hz(e):
        return (10 ** (e / 21.4) - 1.0) / 0.00437

    def _erb_filterbank(self, n_filters, fmin, fmax):
        # ERB-spaced triangular filter bank approximating the magnitude
        # response of a gammatone cochlear filter bank (Patterson 1992).
        e_pts = np.linspace(self._hz_to_erb(fmin), self._hz_to_erb(fmax), n_filters + 2)
        hz_pts = self._erb_to_hz(e_pts)
        fb = np.zeros((n_filters, self.n_freq), dtype=np.float32)
        for m in range(n_filters):
            lo, cen, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            left = (self.freqs - lo) / max(cen - lo, 1e-6)
            right = (hi - self.freqs) / max(hi - cen, 1e-6)
            fb[m] = np.maximum(0.0, np.minimum(left, right))
        return fb.astype(np.float32)

    def _bark_filterbank(self):
        # 24 critical bands (Zwicker 1961)
        bark_edges_hz = np.array([
            0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
            1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
            6400, 7700, 9500, 12000, 15500
        ], dtype=np.float32)
        n = len(bark_edges_hz) - 1
        fb = np.zeros((n, self.n_freq), dtype=np.float32)
        for b in range(n):
            lo, hi = bark_edges_hz[b], bark_edges_hz[b + 1]
            mask = (self.freqs >= lo) & (self.freqs < hi)
            fb[b] = mask.astype(np.float32)
        return fb

    def _third_octave_filterbank(self):
        # IEC 61260 preferred 1/3-octave center frequencies
        cfs = np.array([
            50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
            800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
            8000, 10000, 12500, 16000
        ], dtype=np.float32)
        lo_edge = cfs * (2.0 ** (-1.0 / 6.0))
        hi_edge = cfs * (2.0 ** (1.0 / 6.0))
        keep = hi_edge < (self.sr / 2.0 - 1.0)
        cfs = cfs[keep]
        lo_edge = lo_edge[keep]
        hi_edge = hi_edge[keep]
        n = len(cfs)
        fb = np.zeros((n, self.n_freq), dtype=np.float32)
        for i in range(n):
            mask = (self.freqs >= lo_edge[i]) & (self.freqs < hi_edge[i])
            fb[i] = mask.astype(np.float32)
        return fb, cfs

    def _chroma_filterbank(self, n_chroma, fmin):
        # Harte & Sandler-style chroma: each FFT bin receives a Gaussian
        # weight centered on its nearest pitch class.
        fb = np.zeros((n_chroma, self.n_freq), dtype=np.float32)
        for i, f in enumerate(self.freqs):
            if f < fmin or f > 5000.0:  # standard chroma uses up to ~5 kHz
                continue
            semis = 12.0 * np.log2(f / fmin)
            pc_center = semis % n_chroma
            # Gaussian spread of 0.5 semitone over pitch-class ring
            for pc in range(n_chroma):
                d = np.abs(pc - pc_center)
                d = min(d, n_chroma - d)
                fb[pc, i] = np.exp(-0.5 * (d / 0.5) ** 2)
        # Normalize columns so each FFT bin's contribution sums to 1
        col_sum = fb.sum(axis=0, keepdims=True) + EPS
        fb = fb / col_sum
        return fb.astype(np.float32)

    def _a_weighting_lin(self):
        # IEC 61672 A-weighting curve (linear magnitude)
        f = self.freqs
        f2 = f * f
        num = (12194.0 ** 2) * (f2 * f2)
        den = (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2)
        ra = num / (den + EPS)
        # Normalize so A-weighting is 1.0 at 1 kHz
        idx_1k = int(np.argmin(np.abs(f - 1000.0)))
        a_lin = ra / (ra[idx_1k] + EPS)
        return a_lin.astype(np.float32)

    @staticmethod
    def _dct_matrix(n_in, n_out):
        # Orthonormal DCT-II, keeping the first n_out coefficients (incl. DC).
        m = np.arange(n_in)
        k = np.arange(n_out)[:, None]
        mat = np.cos(np.pi * k * (2 * m + 1) / (2 * n_in))
        mat *= np.sqrt(2.0 / n_in)
        mat[0, :] *= np.sqrt(0.5)
        return mat.astype(np.float32)

    # ---- helpers ----

    @staticmethod
    def _delta(x, n=2):
        # Regression-based delta (Young 1996, HTK).
        T = x.shape[0]
        pad = np.pad(x, ((n, n), (0, 0)), mode='edge')
        denom = 2.0 * sum(i * i for i in range(1, n + 1))
        out = np.zeros_like(x)
        for i in range(1, n + 1):
            out += i * (pad[n + i:n + i + T] - pad[n - i:n - i + T])
        return (out / denom).astype(np.float32)

    def _rolloff(self, cs, tot, pct):
        thresh = (pct * tot)[:, None]  # (T, 1)
        idx = (cs < thresh).sum(axis=1).clip(0, self.n_freq - 1)
        return self.freqs[idx]

    def _pitch_autocorr(self, windows):
        # Autocorrelation-based F0 (Rabiner 1977). Linear autocorr via FFT.
        T, W = windows.shape
        N = 1
        while N < 2 * W:
            N *= 2
        x = np.zeros((T, N), dtype=np.float32)
        # Remove DC per frame for a cleaner autocorrelation
        x[:, :W] = windows - windows.mean(axis=1, keepdims=True)
        X = np.fft.rfft(x, axis=1)
        ac = np.fft.irfft(X.real * X.real + X.imag * X.imag, n=N, axis=1)[:, :W]
        lag_counts = (W - np.arange(W)).astype(np.float32)
        ac = ac / (lag_counts[None, :] + EPS)  # unbiased estimate
        ac0 = ac[:, :1] + EPS
        ac_n = ac / ac0
        f_min_hz, f_max_hz = 55.0, 500.0
        min_lag = max(1, int(self.sr / f_max_hz))
        max_lag = min(W - 2, int(self.sr / f_min_hz))
        search = ac_n[:, min_lag:max_lag + 1]
        local_idx = np.argmax(search, axis=1)
        lag_idx = local_idx + min_lag
        rows = np.arange(T)
        peak = ac_n[rows, lag_idx]
        left = ac_n[rows, (lag_idx - 1).clip(1, W - 2)]
        right = ac_n[rows, (lag_idx + 1).clip(1, W - 2)]
        denom = 2.0 * (2.0 * peak - left - right)
        offset = np.where(np.abs(denom) > EPS, (right - left) / (denom + EPS), 0.0)
        offset = np.clip(offset, -0.5, 0.5)
        lag_ref = lag_idx + offset
        f0 = self.sr / np.maximum(lag_ref, 1.0)
        voicing = np.clip(peak, 0.0, 1.0)
        salience = peak - np.median(search, axis=1)
        return (f0.astype(np.float32), voicing.astype(np.float32),
                salience.astype(np.float32))

    def _stm_energy(self, log_mel):
        # Separable 2D Gabor modulation filter bank in the modulation-frequency
        # plane (Chi, Ru & Shamma 2005). log_mel is (T, M). We pool across the
        # mel axis to get one feature per (rate, scale, direction, frame).
        T, M = log_mel.shape
        S = log_mel - log_mel.mean(axis=0, keepdims=True)
        S = S - S.mean(axis=1, keepdims=True)
        Sf = np.fft.fft2(S)
        w = np.fft.fftfreq(T, d=1.0 / self.frame_sr).astype(np.float32)
        Om = np.fft.fftfreq(M, d=1.0).astype(np.float32)  # cyc/band
        outs = []
        for rate_hz in self.stm_rates_hz:
            bw_r = max(0.5 * rate_hz, 0.5)
            for scale in self.stm_scales_cpb:
                bw_s = max(0.5 * scale, 0.05)
                # Two directions (upward / downward sweep): same rate sign
                # with opposite scale sign.
                gw_pos = np.exp(-0.5 * ((w - rate_hz) / bw_r) ** 2).astype(np.float32)
                gw_neg = np.exp(-0.5 * ((w + rate_hz) / bw_r) ** 2).astype(np.float32)
                gs_pos = np.exp(-0.5 * ((Om - scale) / bw_s) ** 2).astype(np.float32)
                gs_neg = np.exp(-0.5 * ((Om + scale) / bw_s) ** 2).astype(np.float32)
                # Downward sweep (rate > 0, scale > 0 and rate < 0, scale < 0)
                H_down = np.outer(gw_pos, gs_pos) + np.outer(gw_neg, gs_neg)
                # Upward sweep (rate > 0, scale < 0 and rate < 0, scale > 0)
                H_up = np.outer(gw_pos, gs_neg) + np.outer(gw_neg, gs_pos)
                for H in (H_down, H_up):
                    mod = np.fft.ifft2(Sf * H)
                    outs.append(np.abs(mod).mean(axis=1).astype(np.float32))
        return np.stack(outs, axis=1).astype(np.float32)

    @staticmethod
    def _modulation_energy(env, rates, frame_sr):
        # Temporal modulation energy via FFT bandpass with Gabor-shaped
        # filters centered on each rate. Input env: (T, B) real envelope.
        T, B = env.shape
        E = np.fft.fft(env, axis=0)
        f = np.fft.fftfreq(T, d=1.0 / frame_sr).astype(np.float32)
        outs = []
        for r in rates:
            bw = max(0.5 * r, 0.5)
            g = np.exp(-0.5 * ((f - r) / bw) ** 2) + np.exp(-0.5 * ((f + r) / bw) ** 2)
            g[0] = 0.0  # drop DC
            filt = E * g[:, None]
            mod = np.fft.ifft(filt, axis=0)
            outs.append(np.abs(mod).astype(np.float32))
        return np.concatenate(outs, axis=1).astype(np.float32)

    # ---- main entry point ----

    def __call__(self, windows):
        windows = np.asarray(windows, dtype=np.float32)
        T, W = windows.shape
        # Remove per-frame DC and apply Hann window
        x = (windows - windows.mean(axis=1, keepdims=True)) * self.window[None, :]

        # STFT, power and magnitude spectra
        X = np.fft.rfft(x, axis=1)
        power = (X.real * X.real + X.imag * X.imag).astype(np.float32)
        mag = np.sqrt(power + EPS)

        feats = []

        # 1. 80-band log-mel spectrogram
        mel80 = power @ self.mel80_fb.T
        log_mel80 = np.log(mel80 + EPS).astype(np.float32)
        feats.append(log_mel80)

        # 2. 80-band mel with cube-root compression (Stevens power law on power)
        mel80_cbrt = np.power(mel80 + EPS, 1.0 / 3.0).astype(np.float32)
        feats.append(mel80_cbrt)

        # 3. 64-band ERB/gammatone cochleogram envelope (amplitude^0.3)
        erb64_pow = power @ self.erb64_fb.T
        erb64_amp = np.sqrt(erb64_pow + EPS)
        erb64_comp = np.power(erb64_amp + EPS, 0.3).astype(np.float32)
        feats.append(erb64_comp)

        # 4. 64-band ERB log-power cochleogram
        log_erb64 = np.log(erb64_pow + EPS).astype(np.float32)
        feats.append(log_erb64)

        # 5. 1/3-octave band log power (IEC 61260)
        oct_pow = power @ self.oct_fb.T
        log_oct = np.log(oct_pow + EPS).astype(np.float32)
        feats.append(log_oct)

        # 6. 24-band Bark log power (Zwicker)
        bark_pow = power @ self.bark_fb.T
        log_bark = np.log(bark_pow + EPS).astype(np.float32)
        feats.append(log_bark)

        # 7. 24-band Bark specific loudness (Zwicker loudness law)
        bark_specific = np.power(bark_pow + EPS, 0.23).astype(np.float32)
        feats.append(bark_specific)

        # 8. MFCCs (13) from 40-band log-mel
        mel40 = power @ self.mel40_fb.T
        log_mel40 = np.log(mel40 + EPS).astype(np.float32)
        mfcc = (log_mel40 @ self.dct_mat.T).astype(np.float32)
        feats.append(mfcc)

        # 9. Delta and delta-delta MFCCs (Young 1996)
        dmfcc = self._delta(mfcc, n=2)
        ddmfcc = self._delta(dmfcc, n=2)
        feats.append(dmfcc)
        feats.append(ddmfcc)

        # 10. Spectral descriptors (Peeters 2004)
        f = self.freqs[None, :]
        tot_mag = mag.sum(axis=1) + EPS
        pn = mag / tot_mag[:, None]  # normalized magnitude spectrum
        centroid = (pn * f).sum(axis=1)
        var = (pn * (f - centroid[:, None]) ** 2).sum(axis=1)
        spread = np.sqrt(var + EPS)
        skewness = (pn * (f - centroid[:, None]) ** 3).sum(axis=1) / (spread ** 3 + EPS)
        kurtosis = (pn * (f - centroid[:, None]) ** 4).sum(axis=1) / (spread ** 4 + EPS) - 3.0

        cs = np.cumsum(mag, axis=1)
        tot_cs = cs[:, -1:] + EPS
        tot_vec = tot_cs[:, 0]
        rolloff30 = self._rolloff(cs, tot_vec, 0.30)
        rolloff50 = self._rolloff(cs, tot_vec, 0.50)
        rolloff85 = self._rolloff(cs, tot_vec, 0.85)
        rolloff95 = self._rolloff(cs, tot_vec, 0.95)

        log_mag = np.log(mag + EPS)
        flatness = np.exp(log_mag.mean(axis=1)) / (mag.mean(axis=1) + EPS)

        # Spectral slope (least-squares slope of magnitude vs frequency)
        lf = self.freqs
        lf_centered = lf - lf.mean()
        denom_slope = (lf_centered * lf_centered).sum() + EPS
        mag_mean = mag.mean(axis=1, keepdims=True)
        slope = ((mag - mag_mean) * lf_centered[None, :]).sum(axis=1) / denom_slope

        # Spectral decrease (Peeters)
        idx = np.arange(1, self.n_freq).astype(np.float32)
        decrease = ((mag[:, 1:] - mag[:, :1]) / idx[None, :]).sum(axis=1) / (mag[:, 1:].sum(axis=1) + EPS)

        # Spectral flux (positive L2 norm of diff; Dixon 2006). Log-scaled
        # so the resulting feature is robust to overall level differences.
        dmag = np.diff(mag, axis=0, prepend=mag[:1])
        flux = np.log1p(np.sqrt((np.maximum(dmag, 0.0) ** 2).sum(axis=1) + EPS))

        # Spectral entropy
        entropy = -(pn * np.log(pn + EPS)).sum(axis=1)

        # Spectral crest
        crest_spec = mag.max(axis=1) / (mag.mean(axis=1) + EPS)

        spec_desc = np.stack([
            centroid, spread, skewness, kurtosis,
            rolloff30, rolloff50, rolloff85, rolloff95,
            flatness, slope, decrease, flux, entropy, crest_spec
        ], axis=1).astype(np.float32)
        feats.append(spec_desc)

        # 11. Log-frequency versions of centroid and rolloffs (perceptual)
        log_centroid = np.log(centroid + 1.0)[:, None]
        log_rolloffs = np.log(np.stack(
            [rolloff30, rolloff50, rolloff85, rolloff95], axis=1) + 1.0
        )
        feats.append(log_centroid.astype(np.float32))
        feats.append(log_rolloffs.astype(np.float32))

        # 12. Temporal / level descriptors
        rms = np.sqrt((windows ** 2).mean(axis=1) + EPS)
        log_rms = np.log(rms + EPS)
        peak = np.abs(windows).max(axis=1) + EPS
        log_peak = np.log(peak + EPS)
        crest_time = peak / (rms + EPS)
        zcr = ((windows[:, :-1] * windows[:, 1:]) < 0).mean(axis=1)
        # A-weighted power (IEC 61672)
        a_pow = (power * (self.a_weight_lin[None, :] ** 2)).sum(axis=1) + EPS
        a_db = 10.0 * np.log10(a_pow)
        total_energy = power.sum(axis=1) + EPS
        log_total = np.log(total_energy)
        # Zwicker-style total loudness proxy (sum of Bark specific loudness)
        zwicker = bark_specific.sum(axis=1)
        # Spectral sharpness (Zwicker/Fastl 1990)
        bark_centers = (np.arange(bark_specific.shape[1], dtype=np.float32) + 0.5)
        sharp_w = np.where(bark_centers > 15.0,
                           0.066 * np.exp(0.171 * bark_centers), 1.0)
        sharpness = (bark_specific * (sharp_w * bark_centers)[None, :]).sum(axis=1) / (zwicker + EPS)
        temporal = np.stack([
            log_rms, log_peak, crest_time, zcr,
            a_db, log_total, np.log(zwicker + EPS), sharpness
        ], axis=1).astype(np.float32)
        feats.append(temporal)

        # 13. Chroma (12 pitch classes), L1-normalized
        chroma = (power @ self.chroma_fb.T).astype(np.float32)
        chroma = chroma / (chroma.sum(axis=1, keepdims=True) + EPS)
        feats.append(chroma)

        # 14. Spectral contrast peaks and valleys per octave band (Jiang 2002)
        contrast = np.zeros((T, 2 * self.n_contrast), dtype=np.float32)
        alpha = 0.02
        for i, mask in enumerate(self.contrast_masks):
            band = mag[:, mask]
            if band.shape[1] < 2:
                continue
            k = max(1, int(np.ceil(alpha * band.shape[1])))
            sb = np.sort(band, axis=1)
            peak_log = np.log(sb[:, -k:].mean(axis=1) + EPS)
            valley_log = np.log(sb[:, :k].mean(axis=1) + EPS)
            contrast[:, 2 * i] = (peak_log - valley_log).astype(np.float32)
            contrast[:, 2 * i + 1] = peak_log.astype(np.float32)
        feats.append(contrast)

        # 15. Pitch (autocorrelation F0, log-F0, voicing, salience)
        f0, voicing, salience = self._pitch_autocorr(windows)
        log_f0 = np.log(np.maximum(f0, 1.0)).astype(np.float32)
        pitch_feats = np.stack([f0, log_f0, voicing, salience], axis=1).astype(np.float32)
        feats.append(pitch_feats)

        # 16. Temporal modulation energy on ERB cochleogram
        mod_erb = self._modulation_energy(erb64_comp, self.mod_rates_erb, self.frame_sr)
        feats.append(mod_erb)

        # 17. Temporal modulation energy on log-mel (40 bands)
        mod_mel = self._modulation_energy(log_mel40, self.mod_rates_mel, self.frame_sr)
        feats.append(mod_mel)

        # 18. Spectrotemporal modulation energy (Chi / Shamma 2005) on log-mel
        stm = self._stm_energy(log_mel40)
        feats.append(stm)

        out = np.concatenate(feats, axis=1).astype(np.float32)
        # Final safety: replace any NaN/Inf with zero (e.g. silent segments)
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
