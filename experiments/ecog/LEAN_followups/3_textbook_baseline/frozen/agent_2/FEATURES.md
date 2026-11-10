# Feature space

A textbook, closed-form acoustic feature space for predicting auditory ECoG
high-gamma power from 44.1 kHz audio. Nothing is learned from data: every filter
bank, kernel, and parameter is fixed in `FeatureSpace.__init__`.

- Sample rate: 44,100 Hz (fixed by harness)
- Window: 2048 samples (46.4 ms) with a periodic Hann taper, per-frame DC removed
- Hop: 1470 samples (30 Hz frame rate; fixed by harness)
- FFT size: 2048 (1025 rfft bins; bin width 21.5 Hz)
- Total dimensions (H): **856**, finite, float32

Values are returned raw; the scoring harness adds 1/2/3-frame delayed copies and
z-scores every column within each segment, so quantities whose scale varies with
loudness (RMS, peak, flux) are placed in log space before output.

## 1. 80-band log-mel spectrogram — 80 dims
- Slaney-style triangular mel filter bank, area-normalized, 80 filters from
  0 Hz to Nyquist (Slaney 1998; Stevens, Volkmann & Newman 1937).
- Feature: `log(mel_power + eps)`.
- Reference: the standard log-mel representation used throughout speech science
  and auditory encoding models (e.g. Pasley et al. 2012; Hamilton et al. 2018).

## 2. 80-band cube-root mel — 80 dims
- Same 80 filter bank as (1), but compression is `mel_power ** (1/3)`.
- Reference: Stevens' power law for perceived loudness
  (Stevens 1957; Moore 2012). Cube-root compression of power is the compression
  used in the perceptually-motivated gammatone/cochleagram of Ellis (2009).

## 3. 64-band ERB cochleogram envelope (compressed) — 64 dims
- ERB-spaced triangular filters (Glasberg & Moore 1990) from 50 Hz to 16 kHz,
  approximating the magnitude response of a gammatone filter bank
  (Patterson et al. 1992; Slaney 1993).
- Feature: `(sqrt(erb_power)) ** 0.3`, i.e. amplitude envelope with the standard
  cochleogram compression (Mesgarani & Chang 2012; Chi, Ru & Shamma 2005).

## 4. 64-band ERB log-power cochleogram — 64 dims
- Same filter bank as (3), feature is `log(erb_power + eps)` — the cochleogram
  representation in log units commonly used in auditory encoding models
  (Theunissen et al. 2000; de Heer et al. 2017).

## 5. 1/3-octave band log power — 26 dims
- IEC 61260 preferred center frequencies from 50 Hz up to 16 kHz, with
  log-spaced edges `cf * 2 ** (±1/6)`. Feature: `log(band_power + eps)`.
- Reference: classical acoustic analysis (ISO 532; ANSI S1.11), used as the
  reference band structure for sound level measurement.

## 6. 24-band Bark log power — 24 dims
- Zwicker critical-band edges (Zwicker 1961). Feature: `log(bark_power + eps)`.

## 7. 24-band Bark specific loudness — 24 dims
- Same Bark bands as (6). Feature: `bark_power ** 0.23`, the specific-loudness
  compression from Zwicker & Fastl's loudness model (Zwicker & Fastl 1990).

## 8. MFCCs 0–12 — 13 dims
- DCT-II (orthonormal) of a 40-band Slaney log-mel; retain coefficients 0–12
  (Davis & Mermelstein 1980).

## 9. Delta MFCCs — 13 dims
- First-order regression delta with half-width n = 2 (HTK formula; Young 1996).

## 10. Delta-delta MFCCs — 13 dims
- Regression delta applied to the delta features (acceleration coefficients).

## 11. Spectral descriptors (Peeters 2004) — 14 dims
Computed per frame on the Hann-windowed magnitude spectrum.
- Spectral centroid (Hz)
- Spectral spread / bandwidth (Hz)
- Spectral skewness
- Spectral kurtosis (excess)
- Spectral rolloff at 30 %, 50 %, 85 %, 95 % of cumulative magnitude (Hz)
- Spectral flatness (geometric / arithmetic mean of magnitude)
- Spectral slope (least-squares slope of magnitude vs frequency)
- Spectral decrease (Peeters)
- Spectral flux: `log1p` of the half-wave-rectified L2 norm of the magnitude
  frame-to-frame difference (Dixon 2006; Bello et al. 2005). Log-scaled for
  stability.
- Spectral entropy of the normalized magnitude spectrum
- Spectral crest factor

## 12. Log centroid / log rolloffs — 1 + 4 = 5 dims
- `log(centroid + 1)` and `log(rolloff_{30, 50, 85, 95} + 1)` — perceptually
  relevant log-frequency versions of the stats above. Common in MIR
  (Lerch 2012).

## 13. Temporal / level descriptors — 8 dims
- Log RMS
- Log peak amplitude
- Temporal crest factor (peak / RMS)
- Zero-crossing rate (Kedem 1986)
- A-weighted sound level in dB (IEC 61672)
- Log total spectral energy
- Log total Zwicker loudness (sum of Bark specific loudness)
- Spectral sharpness (Zwicker & Fastl 1990, equation (5.5))

## 14. Chroma — 12 dims
- 12-bin pitch-class profile (Fujishima 1999; Harte & Sandler 2006). Each FFT
  bin contributes a Gaussian weight (σ = 0.5 semitone) to its nearest pitch
  class on the circle, using the range 55 Hz – 5 kHz. L1-normalized per frame.

## 15. Spectral contrast — 18 dims
- Jiang et al. (2002). Octave-band edges at 100/200/400/800/1600/3200/6400/
  12800 Hz plus Nyquist (9 bands). Per band, keep the α = 2 % extreme values
  and output both the log peak and `(log peak − log valley)` contrast.

## 16. Pitch (autocorrelation) — 4 dims
- Linear autocorrelation computed via FFT (next-power-of-two zero padding);
  unbiased normalization; search for the lag maximum in 55–500 Hz; parabolic
  interpolation around the peak (Rabiner 1977; Boersma 1993).
- Features: `F0 (Hz)`, `log F0`, voicing (normalized peak), pitch salience
  (peak − median of search region).

## 17. Temporal modulation energy on ERB cochleogram — 64 × 4 = 256 dims
- Across-frame FFT of each ERB envelope from (3), bandpass with Gabor-shaped
  modulation filters (Dau, Püschel & Kohlrausch 1996) at rates
  {1, 2, 4, 8} Hz, magnitude of inverse FFT. DC is removed.
- Reference: the modulation filter bank at the core of modern auditory
  perception models (Dau 1996; Elliott & Theunissen 2009).

## 18. Temporal modulation energy on log-mel — 40 × 3 = 120 dims
- Same scheme as (17), applied to the 40-band log-mel spectrogram; rates
  {2, 4, 8} Hz. Captures slower, speech-rate temporal modulations
  (Greenberg et al. 2003).

## 19. Spectrotemporal modulation energy — 3 rates × 3 scales × 2 directions = 18 dims
- Chi, Ru & Shamma (2005) separable 2D Gabor modulation filter bank in the
  modulation-frequency plane of the log-mel spectrogram.
- Temporal rates: {2, 4, 8} Hz. Spectral scales: {1/8, 1/4, 1/2} cyc/band
  (≈ 1, 2, 4 cyc/octave on 40-band mel). Upward and downward sweep directions
  are produced by opposite-sign quadrant combinations.
- Each filter's response magnitude is averaged across the mel axis to give one
  feature per frame.

---

Total: 80 + 80 + 64 + 64 + 26 + 24 + 24 + 13 + 13 + 13 + 14 + 5 + 8 + 12 + 18 + 4 + 256 + 120 + 18 = **856** dims.
