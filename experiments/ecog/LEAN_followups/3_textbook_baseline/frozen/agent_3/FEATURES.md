# Feature space for ECoG high-gamma encoding model

A textbook interpretable acoustic feature space, built from closed-form
signal-processing routines with fixed parameters drawn from standard
practice in auditory neuroscience, speech science and audio signal
processing. Nothing is fitted or trained on data.

Audio is mono at 44,100 Hz. Frames are produced every 1,470 samples
(30 Hz). The analysis window is **win = 2048 samples (~46.4 ms)**.

Each window is DC-removed, pre-emphasized with alpha = 0.97, Hann-windowed
and transformed with the real FFT (1,025 bins). All subsequent features
are derived from this STFT (and from the raw / DC-removed windows where
noted). Total output dimensionality **H = 650**.

| # | Feature group | Dim | Reference |
|---|----------------|----:|-----------|
| 1 | Log-mel spectrogram | 80 | Davis & Mermelstein 1980; Slaney 1998 |
| 2 | Delta log-mel | 80 | Furui 1986 |
| 3 | Delta-delta log-mel | 80 | Furui 1986 |
| 4 | MFCC (DCT-II of log-mel) | 20 | Davis & Mermelstein 1980 |
| 5 | Delta MFCC | 20 | Furui 1986 |
| 6 | Delta-delta MFCC | 20 | Furui 1986 |
| 7 | F0 / voicing / HNR | 6 | Rabiner 1977; de Cheveigné & Kawahara 2002 |
| 8 | Spectral shape descriptors | 10 | Peeters 2004 |
| 9 | Spectral flux | 1 | Masri 1996; Dixon 2006 |
| 10 | Broadband energy / loudness | 6 | IEC 61672-1 (A-weighting) |
| 11 | Zero-crossing rate | 1 | Rabiner & Schafer 1978 |
| 12 | Temporal modulation filterbank | 320 | Chi, Ru & Shamma 2005 |
| 13 | Spectral modulation energy | 5 | Chi, Ru & Shamma 2005 |
| 14 | Broadband onset envelope | 1 | Bello et al. 2005; Böck & Widmer 2013 |
| | **Total** | **650** | |

The harness will internally append 1-, 2- and 3-frame delays (effective
4× feature count) and z-score every column per segment before fitting a
per-electrode ridge regression.

---

## 1. Log-mel spectrogram (80 dims)

Classical auditory-perceptual spectrogram used ubiquitously as the input
representation in ECoG speech/music encoding models (e.g. Pasley et al.
2012; Mesgarani & Chang 2012; de Heer et al. 2017; Khalighinejad et al.
2017).

* 80 triangular filters on the Mel scale from 20 Hz to 8,000 Hz.
* Slaney (area) normalization of each filter.
* Log compression: `log(mel_power + 1e-6)`.
* Pre-emphasis coefficient 0.97, Hann window, 2048-sample DFT.

## 2–3. Delta and delta-delta log-mel (160 dims)

HTK-style regression-coefficient deltas with N = 2 neighbours on either
side (Furui 1986), applied along the time axis of the log-mel
spectrogram. Delta-deltas are the deltas applied to the deltas.

## 4. MFCC (20 dims)

Orthonormal DCT-II of the 80 log-mel values, truncated to the first 20
coefficients (`c0`–`c19`). Standard Mel-frequency cepstral coefficients
(Davis & Mermelstein 1980).

## 5–6. Delta and delta-delta MFCC (40 dims)

Same HTK delta regression as (2–3), applied to the MFCC time series.

## 7. Pitch / voicing (6 dims)

Autocorrelation-based F0 estimator (Rabiner 1977; the YIN difference
function of de Cheveigné & Kawahara 2002 is a direct variant).

* Normalized autocorrelation computed via FFT on the Hann-windowed,
  DC-removed signal.
* Search range: τ corresponding to F0 ∈ [60 Hz, 500 Hz] (88 to 735
  samples), covering adult male/female speech and the bulk of musical
  pitch.
* The six outputs are: (F0 / 100 Hz), log F0, voicing probability (peak
  normalized autocorrelation), harmonic-to-noise ratio in dB,
  voiced-masked F0 / 100 Hz, voiced-masked log F0. Voiced mask threshold
  = 0.3.

## 8. Spectral shape descriptors (10 dims)

Standard MPEG-7 / MIR descriptors (Peeters 2004):

* Spectral centroid (Hz / 1000)
* Spectral spread / bandwidth (Hz / 1000)
* Spectral skewness
* Spectral kurtosis
* Spectral flatness (geometric / arithmetic mean of power)
* Spectral entropy (of the power spectrum, treated as a probability)
* 50 %, 85 %, 95 % spectral rolloff (Hz / 1000)
* Spectral slope (OLS of log |X| vs log f, excluding DC)

## 9. Spectral flux (1 dim)

Half-wave-rectified L2 difference of successive log-mel frames; a
classical onset function (Masri 1996; Dixon 2006).

## 10. Energy / loudness (6 dims)

Standard broadband amplitude descriptors:

* Linear RMS of the windowed time signal
* Log RMS
* A-weighted RMS using the IEC 61672-1 A-weighting filter shape applied
  to the magnitude spectrum
* Log A-weighted RMS
* Peak absolute amplitude
* Crest factor (peak / RMS)

## 11. Zero-crossing rate (1 dim)

Fraction of adjacent-sample sign changes per window (Rabiner & Schafer
1978). Correlates with spectral centroid for noisy signals and is a
classical fricative / sibilance indicator.

## 12. Temporal modulation filterbank (320 dims)

The temporal half of the Chi, Ru & Shamma (2005) spectro-temporal
modulation model. Each log-mel band is passed through a bank of
analytic Morlet-wavelet bandpass filters at modulation rates 1, 2, 4
and 8 Hz (Q = 2), and the envelope magnitude is taken. This captures
the slow amplitude-modulation energy that dominates speech and music
envelopes and strongly drives auditory cortex.

* 4 rates × 80 mel bands = 320 features.
* Implemented as a Gaussian-weighted positive-frequency mask on the
  per-band time-series FFT, followed by inverse FFT and magnitude.

## 13. Spectral modulation energy (5 dims)

The spectral half of the STRF model (Chi/Shamma 2005): for every frame,
a Gaussian band at each scale is applied to the magnitude FFT of the
log-mel spectrum along the mel-frequency axis. Scales are 0.25, 0.5,
1, 2 and 4 cycles/octave — the range psychophysical and physiological
work has identified as relevant for timbre and formant structure.

## 14. Broadband onset envelope (1 dim)

Half-wave-rectified first difference of log-RMS energy — the simplest
classical broadband onset detection function (Bello et al. 2005;
Böck & Widmer 2013).
