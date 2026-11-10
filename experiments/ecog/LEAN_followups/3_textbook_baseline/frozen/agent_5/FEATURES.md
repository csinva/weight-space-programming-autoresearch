# FEATURES.md

Textbook, interpretable acoustic feature space for predicting ECoG high-gamma
power from naturalistic audio. All features are fixed, closed-form signal
processing applied to a single 2048-sample Hann-windowed frame (unless noted),
with pre-emphasis coefficient 0.97 applied inside each window. Sample rate
44 100 Hz, frame rate 30 Hz, analysis window 2048 samples (~46 ms). Audio is
scaled by 1/32768 (int16-range inputs assumed).

Total dimensionality: **H = 339**.

| # | Feature group | Dims | Parameters | Reference literature |
|---|---|---:|---|---|
| 1 | Log-mel spectrogram | 64 | 64 Slaney-normalized triangular filters, 50 Hz–22 050 Hz, mel scale (`m = 2595·log10(1 + f/700)`), `log(·)` with 1e-10 floor | Stevens, Volkmann & Newman (1937); Davis & Mermelstein (1980); Slaney (1998), *Auditory Toolbox*. The standard front-end for speech and the de-facto standard spectrogram for auditory cortical encoding studies. |
| 2 | Gammatone cochleagram | 32 | 32 ERB-rate-spaced centers 50 Hz–~20 950 Hz, 4th-order gammatone magnitude^2 approximation `\|H(f)\|² = (1 + ((f-f_c)/(1.019·ERB(f_c)))²)^{-4}`, peak-normalized, `log(·)` | Patterson et al. (1987, 1992), *Auditory Physiology and Perception*; Glasberg & Moore (1990); Slaney (1993), *An Efficient Implementation of the Patterson-Holdsworth Auditory Filter Bank*. Standard cochlear-model front-end used in Mesgarani/Chang-style spectrotemporal RF studies and Norman-Haignere cochleagrams. |
| 3 | MFCC | 13 | DCT-II (orthonormal) of log-mel, keep first 13 coefficients (includes C₀) | Davis & Mermelstein (1980); HTK book (Young et al.); Rabiner & Juang (1993). Workhorse of speech recognition. |
| 4 | Delta MFCC | 13 | HTK/Kaldi regression deltas with N=2: `Δc_t = Σ_{n=1..2} n(c_{t+n}-c_{t-n}) / (2·Σn²)` | Furui (1986), *Speaker-independent isolated word recognition using dynamic features*; HTK book. |
| 5 | Delta-Delta MFCC | 13 | Same operator applied to Δ-MFCC | Furui (1986); HTK book. |
| 6 | Spectral centroid | 1 | `Σ f·p(f)` with `p(f) = |X(f)| / Σ|X(f)|` | Peeters (2004), *A large set of audio features for sound description*; Tzanetakis & Cook (2002); MPEG-7 AudioSpectrumCentroid. |
| 7 | Spectral bandwidth | 1 | `sqrt(Σ (f-centroid)²·p(f))` | Peeters (2004); Klapuri & Davy (2006). |
| 8 | Spectral rolloff (5 percentiles) | 5 | Frequencies at which cumulative magnitude crosses 25%, 50%, 75%, 85%, 95% | Scheirer & Slaney (1997); Tzanetakis & Cook (2002); MPEG-7. 85% is the classic "rolloff frequency". |
| 9 | Log spectral flatness (Wiener entropy) | 1 | `log(geomean(|X|²)/mean(|X|²))` across all FFT bins | Johnston (1988); Dubnov (2004); MPEG-7 AudioSpectrumFlatness. |
| 10 | Spectral slope | 1 | Least-squares slope of log-power vs log-frequency | Peeters (2004); MPEG-7 AudioSpectrumSpread/Slope. |
| 11 | Spectral skewness | 1 | 3rd standardized central moment of the normalized spectrum | Peeters (2004). |
| 12 | Spectral excess kurtosis | 1 | 4th standardized central moment minus 3 | Peeters (2004). |
| 13 | Spectral entropy | 1 | `-Σ p(f)·log p(f)` with `p = |X|/Σ|X|` | Misra et al. (2004); Shen et al. (1998). |
| 14 | Log crest factor | 1 | `log(max|X| / mean|X|)` across FFT bins | Peeters (2004), MPEG-7. |
| 15 | Log spectral flux | 1 | `log sqrt(Σ(|X_t|-|X_{t-1}|)²)` (first frame uses itself) | Scheirer & Slaney (1997); Tzanetakis & Cook (2002); Dixon (2006). |
| 16 | Zero-crossing rate | 1 | Fraction of sign changes in the raw window | Rabiner & Schafer (1978); Panagiotakis & Tziritas (2005). |
| 17 | Log RMS energy | 1 | `log sqrt(mean(x²))` | Classic short-time energy — Rabiner & Schafer (1978). |
| 18 | Log peak amplitude | 1 | `log max|x|` | Standard loudness descriptor (MPEG-7 AudioPower). |
| 19 | Log chroma | 12 | 12 pitch classes; FFT-bin soft triangular assignment to 2 nearest semitones around A4=440 Hz, limited to 80–5000 Hz | Fujishima (1999); Bartsch & Wakefield (2005); Müller (2007), *Information Retrieval for Music and Motion*. |
| 20 | Autocorrelation log-F0 | 1 | FFT-based autocorrelation per window, search range 60–500 Hz, `log(sr / argmax_lag)` | Rabiner (1977); Boersma (1993), *Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound*. |
| 21 | Voicing (normalized autocorrelation peak) | 1 | Peak of normalized autocorrelation in the F0 search range; proxy for HNR/periodicity | Boersma (1993); Praat; de Cheveigné & Kawahara (2002). |
| 22 | Spectral contrast — peaks | 6 | 6 octave-spaced bands 200 Hz–22 050 Hz, log of mean of top 2% of power per band | Jiang, Lu, Zhang, Tao & Cai (2002), *Music type classification by spectral contrast feature*. |
| 23 | Spectral contrast — valleys | 6 | Same bands, log of mean of bottom 2% of power per band | Jiang et al. (2002). |
| 24 | Temporal modulation energies | 160 | 32-band mel log-envelope filtered across frames with zero-phase 4th-order Butterworth bandpass at center frequencies {0.5, 1, 2, 4, 8} Hz (bandwidth = max(f_m/2, 0.5) Hz), `|output|` per frame | Kingsbury et al. (1998), *Robust speech recognition using the modulation spectrogram*; Chi, Ru & Shamma (2005), *Multiresolution spectrotemporal analysis of complex sounds*; Elliott & Theunissen (2009); McDermott & Simoncelli (2011), *Sound texture perception via statistics of the auditory periphery*. Captures the 1–16 Hz envelope rates known to dominate STRF-style auditory cortex responses. |
| 25 | Onset envelope | 1 | Mean over mel bands of `max(log_mel_t − log_mel_{t-1}, 0)` | Klapuri (1999); Bello et al. (2005), *A tutorial on onset detection in music signals*. |

## Design notes

- **Pre-emphasis (0.97)** is applied inside each window before the Hann window
  and FFT to flatten the overall spectral tilt, as is standard in speech
  front-ends (HTK, Kaldi).
- **Spectrogram basis:** 64-band mel and 32-band ERB-spaced gammatone are kept
  as *separate* representations because the auditory literature uses both — the
  mel scale is standard in speech, the ERB-rate gammatone is the textbook
  cochlear model.
- **Temporal context:** the harness itself adds 1-, 2-, 3-frame delays
  (0–100 ms), so short-horizon temporal information is already available to
  the ridge model via lagged features. We provide longer-timescale information
  through (a) MFCC Δ/ΔΔ (regression over ±2 frames = 167 ms), and (b) the
  across-frame modulation energies, which integrate information over seconds
  at the 0.5–1 Hz rates.
- **Nonlinearities:** energies are always log-compressed (natural log with a
  1e-10 floor), matching standard auditory models of loudness.
- **Numerical safety:** every division uses a small epsilon; every log uses
  `safe_log`; the final output is passed through an `isfinite` guard that
  replaces any non-finite values with 0 before returning float32.
