# Feature space — textbook acoustic baseline for auditory ECoG high-gamma encoding

All features are **closed-form signal processing** applied per 46.4 ms (2048-sample) Hann window
at the fixed 30 Hz frame rate. Pre-emphasis coefficient 0.97 (Rabiner & Juang 1993). No fitting,
estimation, or learning: every parameter below is hard-coded. Total dimensionality **H = 301**.

The harness supplies three temporal lags on top of this (0 / 33 / 66 / 100 ms context) and
z-scores each column within a segment, so the features below are described as instantaneous.

## Common front-end

- **Window:** Hann, 2048 samples (~46.4 ms) at 44.1 kHz — standard STFT windowing (Harris 1978).
- **Pre-emphasis:** first-order high-pass with α = 0.97 (Rabiner & Juang 1993, standard in speech
  analysis; compensates −6 dB/oct spectral tilt of voiced speech).
- **DFT size:** 2048 (equal to window; no zero-padding).

## 1. Log-mel cochleogram — 64 dims

- **Reference.** Davis & Mermelstein (1980); Slaney (1998) "Auditory Toolbox" normalization.
  Mel-spectrogram inputs are the de facto standard for auditory encoding models from
  Mesgarani & Chang (2014), Oganian & Chang (2019), de Heer et al. (2017) and many others.
- **Parameters.** 64 Slaney-style triangular mel filters from 20 Hz to 16 kHz, applied to the
  STFT power spectrum; unit-area normalized; then `log(power + 1e-6)`.

## 2. Delta log-mel (temporal derivative of cochleogram) — 64 dims

- **Reference.** Furui (1986) regression-coefficient delta formula, with M = 2 (five-tap).
- **Interpretation.** First temporal derivative of each mel band. Captures spectral change rate;
  commonly included alongside static spectrogram features in speech recognition pipelines.

## 3. Per-band peak rate (half-wave rectified envelope derivative) — 64 dims

- **Reference.** Oganian & Chang (2019), *Nature Human Behaviour* — "peak rate" was shown to
  drive STG high-gamma responses to speech; defined as the half-wave rectified first time
  derivative of the sound envelope.
- **Parameters.** `max(diff(log_mel, axis=time), 0)` per mel band.

## 4. MFCCs + Δ + ΔΔ — 13 + 13 + 13 = 39 dims

- **Reference.** Davis & Mermelstein (1980). Deltas and double-deltas: Furui (1986).
- **Parameters.** DCT-II (orthonormal) of the 64-band log-mel, keeping coefficients 0–12
  (including log-energy-like C0). Deltas and double-deltas via regression with M = 2.

## 5. Spectral shape descriptors — 10 dims

- **Reference.** Peeters (2004) "A large set of audio features for sound description"; also
  Scheirer & Slaney (1997). Standard low-level timbre descriptors used in MPEG-7.
- **Dimensions.** Spectral centroid, spread, skewness, kurtosis (central moments weighted by
  STFT power); spectral flatness (geometric/arithmetic mean ratio); spectral rolloff at 85 %
  and 95 % cumulative power; spectral slope (linear regression of log-power on Hz); spectral
  flux (frame-to-frame L2 of √power difference); spectral entropy of the normalized spectrum.

## 6. Octave-band spectral contrast — 7 dims

- **Reference.** Jiang, Lu, Zhang, Tao & Cai (2002), *IEEE ICME* — "Music type classification
  by spectral contrast feature".
- **Parameters.** Seven bands with upper edges 200, 400, 800, 1600, 3200, 6400, 20 000 Hz;
  contrast = log mean of top 2 % − log mean of bottom 2 % of power bins within each band.

## 7. Zero-crossing rate — 1 dim

- **Reference.** Rabiner & Schafer (1978). Fraction of adjacent audio samples with differing
  sign within the analysis window; a classical voicing / noisiness cue.

## 8. Short-time energy — 2 dims

- **RMS (linear)** and **RMS (dB)** of the raw windowed audio. Textbook broadband loudness
  surrogate (Rabiner & Schafer 1978).

## 9. A-weighted loudness — 1 dim

- **Reference.** IEC 61672-1 (A-weighting curve, originally Fletcher & Munson 1933).
  Perceptually weighted sound level: STFT power multiplied by the squared A-weighting
  transfer function (calibrated to unit gain at 1 kHz), summed across bins, in dB.

## 10. Pitch and voicing — 4 dims

- **Reference.** Classical autocorrelation-based pitch detection (Rabiner 1977; Boersma 1993,
  Praat). Normalized autocorrelation computed via FFT on the mean-centered window; max peak
  located in lag range corresponding to 50–500 Hz.
- **Dimensions.** `f0` (Hz), `log f0`, voicing strength (peak autocorrelation value in [0, 1]),
  harmonics-to-noise ratio HNR = 10 log10(v / (1 − v)).

## 11. Chroma — 12 dims

- **Reference.** Fujishima (1999) "Realtime chord recognition of musical sound"; Bartsch &
  Wakefield (2005).
- **Parameters.** Each STFT bin in 50–5000 Hz is routed to one of 12 pitch classes relative to
  A4 = 440 Hz; log-power of each class. A music / tonality descriptor relevant to hospital
  audio containing television and music.

## 12. Broadband envelope peak rate — 1 dim

- **Reference.** Oganian & Chang (2019). Half-wave rectified first time derivative of the
  log broadband power envelope; complements the per-band peak rate in group 3.

## 13. Temporal modulation filterbank — 4 rates × 8 bands = 32 dims

- **Reference.** Chi, Ru & Shamma (2005), *JASA*; Elliott & Theunissen (2009) "The modulation
  transfer function for speech intelligibility". Amplitude modulation at 1–16 Hz is a dominant
  driver of auditory-cortex responses.
- **Parameters.** The 64 log-mel bands are pooled into 8 contiguous frequency groups by
  averaging. Each group is zero-meaned in time and convolved (along the 30 Hz frame axis) with
  cosine/sine Hann-windowed Gabor filters at 1, 2, 4 and 8 Hz (four-cycle window). The
  modulation **envelope** `sqrt(cos² + sin²)` is kept — one value per group × rate.

## Dimension summary

| Group | Dims |
|---|---|
| 1. Log-mel cochleogram | 64 |
| 2. Delta log-mel | 64 |
| 3. Per-band peak rate | 64 |
| 4. MFCC + Δ + ΔΔ | 39 |
| 5. Spectral descriptors | 10 |
| 6. Spectral contrast | 7 |
| 7. Zero-crossing rate | 1 |
| 8. RMS (linear, log) | 2 |
| 9. A-weighted loudness | 1 |
| 10. Pitch / voicing / HNR | 4 |
| 11. Chroma | 12 |
| 12. Broadband peak rate | 1 |
| 13. Temporal modulation filterbank | 32 |
| **Total H** | **301** |
