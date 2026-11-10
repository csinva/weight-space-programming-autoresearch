# Acoustic feature space — textbook baseline

A fixed, closed-form acoustic feature space designed as a conventional baseline
for predicting ECoG high-gamma power from naturalistic audio. All features are
standard in the auditory neuroscience, speech science and audio signal
processing literature, with standard parameter choices. No learning.

## Global analysis parameters
- Sample rate: 44 100 Hz (fixed).
- Frame rate: 30 frames / s (fixed by the harness; hop = 1 470 samples).
- Analysis window: `win = 2048` samples (~46 ms), Hann-tapered.
- Pre-emphasis (first-order high-pass) with α = 0.97 applied inside each window
  before FFT. Standard in speech processing (Rabiner & Schafer, *Theory and
  Applications of Digital Speech Processing*, 2011).
- FFT length = window length = 2048; rfft → 1 025 complex bins.
- Power spectrum `P[f]` and magnitude spectrum `M[f] = sqrt(P[f])`.
- Input audio is normalised by 32 768 (int16 → [-1, 1]) before any log-scale
  feature so that log magnitudes have a stable dynamic range.

Total output dimensionality: **H = 549** (≤ 4 000).

## Feature groups

### 1. Log-mel spectrogram — 80 dims
Classic auditory representation (Davis & Mermelstein, 1980; Mesgarani et al.,
*Science* 2014; Norman-Haignere et al., *Neuron* 2015). Triangular mel
filterbank, Slaney (htk-compatible) area-normalisation, 80 bands between 50 Hz
and 20 kHz, applied to the power spectrum followed by `log(x + 1e-6)`.

### 2. Δ log-mel — 80 dims
HTK-style regression delta over ±2 frames (Young et al., *HTK Book*, 2006).
First temporal derivative of the log-mel spectrogram.

### 3. ΔΔ log-mel — 80 dims
Delta of the delta (second temporal derivative). Same ±2 frame kernel.

### 4. MFCC C0…C19 — 20 dims
Mel-frequency cepstral coefficients: orthonormal type-II DCT of the 80-band
log-mel spectrum, first 20 coefficients (Davis & Mermelstein, 1980). C0 is
retained as broadband log-energy proxy, in line with HTK defaults.

### 5. Δ MFCC — 20 dims
HTK regression delta (±2 frames) of the MFCCs.

### 6. ΔΔ MFCC — 20 dims
Second-order HTK regression delta of the MFCCs.

### 7. Chroma (pitch-class profile) — 12 dims
Gaussian-tapered pitch-class filterbank, σ = 1 semitone, support between
55 Hz (A1) and 8 kHz, L1-normalised rows, applied to magnitude spectrum, then
L2-normalised per frame. Standard chroma for music information retrieval
(Bartsch & Wakefield, 2001; Fujishima, 1999; Ellis, 2007).

### 8. Spectral shape descriptors — 8 dims
Standard spectral descriptors (Peeters, *CUIDADO Technical Report*, 2004;
MPEG-7 Audio). All transformed via `log1p` or signed log for conditioning.
  - **Spectral centroid** — power-weighted mean frequency.
  - **Spectral bandwidth** — power-weighted standard deviation around the centroid.
  - **Spectral rolloff (85 %)** — frequency below which 85 % of the power lies.
  - **Spectral flatness (Wiener entropy)** — geometric / arithmetic mean of power.
  - **Spectral slope** — least-squares slope of log-power vs log-frequency.
  - **Spectral flux** — L2 of half-wave-rectified frame-to-frame magnitude difference
    (novelty / onset signal, Masri, 1996; Bello et al., 2005).
  - **Spectral crest** — peak / mean of the power spectrum.
  - **Spectral entropy** — normalised Shannon entropy of the power distribution.

### 9. Envelope, zero-crossing and voicing — 8 dims
Standard short-time features from speech processing.
  - **log RMS** — window RMS amplitude, logged.
  - **Δ log RMS** — HTK regression delta of log RMS.
  - **ΔΔ log RMS** — second regression delta of log RMS.
  - **log peak** — log of absolute peak amplitude in the window.
  - **Zero-crossing rate** — on the pre-emphasised, un-windowed signal.
  - **log F0** — autocorrelation pitch (Rabiner, 1977; Boersma, 1993).
    Range 70–500 Hz; parabolic sub-sample interpolation of the autocorrelation
    peak.
  - **Pitch confidence** — normalised autocorrelation peak value (voicing strength).
  - **Voicing flag** — `pitch_confidence > 0.3`.

### 10. Temporal modulation of log-mel subbands — 200 dims
Bandpass-filtered subband envelopes, the discrete modulation-bank cousin of
spectrotemporal receptive field models (Chi, Ru & Shamma, *JASA* 2005;
Elliott & Theunissen, *PLoS CB* 2009; Mesgarani et al., *Nature* 2014).

Log-mel bands are first down-sampled by averaging adjacent pairs → 40 bands,
then each band is filtered along time with 2nd-order Butterworth bandpass
filters at centre rates **1, 2, 4, 8, 12 Hz** (half-octave half-widths; `sosfiltfilt`
for zero-phase). Yields 40 × 5 = 200 features covering the modulation rates
most relevant for speech / music (syllabic rate ≈ 4 Hz).

### 11. Octave-band spectral contrast — 21 dims
Jiang, Lu, Zhang, Tao & Cai (*ICASSP* 2002). For each of 7 octave bands with
edges `[0, 200, 400, 800, 1600, 3200, 6400, 12800] Hz`, within-band peak and
valley of `log |M|` are estimated as the mean of the top / bottom `α = 2 %` of
bins. Features per band: `log_peak`, `log_valley`, and `contrast = peak – valley`
→ 7 × 3 = 21 features. Captures tonal vs noise-like balance across the
cochlea.

## Summary

| Group | Dims |
|---|---|
| Log-mel spectrogram | 80 |
| Δ log-mel | 80 |
| ΔΔ log-mel | 80 |
| MFCC (C0..C19) | 20 |
| Δ MFCC | 20 |
| ΔΔ MFCC | 20 |
| Chroma | 12 |
| Spectral shape | 8 |
| Envelope / ZCR / pitch / voicing | 8 |
| Mel subband temporal modulation (40 × 5) | 200 |
| Octave-band spectral contrast (7 × 3) | 21 |
| **Total** | **549** |

After the harness stacks 3 delayed copies, the ridge regression receives
`549 × 4 = 2196` features per electrode.
