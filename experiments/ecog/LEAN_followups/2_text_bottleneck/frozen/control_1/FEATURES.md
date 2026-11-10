# Acoustic feature space (`/work/feature_space.py`)

`win = 4096`. **H = 710** features per 30 Hz frame. Everything is closed-form signal processing with
fixed parameters. Nothing is learned, fitted or estimated from data.

## Pre-processing
- The overlapping windows are stitched back into the continuous segment waveform. Each window starts
  1470 samples after the previous one, so the waveform is recovered exactly.
- The waveform is divided by its own RMS over the segment (a fixed gain rule). This makes the fixed
  log floors independent of recording level.
- Spectra are computed on a 150 Hz sub-frame grid (hop 294 samples, 5 sub-frames per frame, centered on
  the frame center). Each 30 Hz frame is the mean over its 5 sub-frames, which acts as an anti-aliasing
  pool. The ends of the segment are reflect-padded.
- Three Hann STFTs are used: 1024 samples (23 ms) for the mel and spectral features, 2048 samples (46 ms)
  for the modulation spectrogram, and 4096 samples (93 ms) for pitch.
- **Lead copies:** the harness adds delays of 1–3 frames to a fixed 100 ms lag, so it only sees lags of
  100–200 ms. Some groups therefore also include the value 1 or 2 frames *ahead* (effective lags of 67 or
  33 ms). This lets the readout capture short-latency responses (e.g. in primary auditory cortex).
  Smoothed causal history (group 7) covers lags longer than 200 ms.

## Feature groups

| # | Group | Parameters | Dims |
|---|-------|------------|-----:|
| 1 | `mel` | Log-mel power spectrogram, 80 triangular bands from 50 Hz to 12 kHz, 23 ms window. Power is pooled over the frame, then `log(P + 1e-5)`. | 80 |
| 1b | `mel_lead1` | Same as `mel`, taken 1 frame ahead (+33 ms). | 80 |
| 2 | `onset16` | Onsets: half-wave-rectified sub-frame (150 Hz) time derivative of log-mel, averaged into 16 coarse bands (5 mel bands each) and summed over the frame. | 16 |
| 2b | `onset16_lead1` | Same as `onset16`, 1 frame ahead. | 16 |
| 2c | `offset16` | Offsets: the negative-going part of the same derivative. | 16 |
| 3 | `envelope` | Broadband log energy, peak rate (positive derivative of the Gaussian-smoothed (σ = 13 ms) log envelope) and trough rate (negative derivative), each at lead 0, 1 and 2 frames. | 9 |
| 3b | `subframe_level` | Broadband log energy of the 5 sub-frames minus the frame mean: the loudness trajectory within the frame (6.7 ms steps). | 5 |
| 4 | `pitch_salience` | Sub-harmonic summation on a whitened log-magnitude spectrum (93 ms window; spectral peaks = log-magnitude minus its 15-bin moving average, rectified). 37 semitone F0 bins from 65 to 520 Hz, 10 harmonics up to 5 kHz with weights 0.84^(h−1). Divided by the mean peak strength and gated by a level sigmoid, so silence goes to 0. | 37 |
| 4b | `voicing` | Periodicity strength (maximum salience) and voicing-weighted log2 F0, each at lead 0 and 1. | 4 |
| 4c | `pitch_change` | Change in log F0 from the previous frame (semitones), soft-binned with Gaussian bins (σ = 0.75 st) centered at {−6, −3, −1.5, −0.5, 0, 0.5, 1.5, 3, 6} and weighted by the voicing of both frames. | 9 |
| 5 | `stm` | Spectro-temporal modulation energy: separable complex Gabor filters applied to a 64-band constant-Q log spectrogram (100 Hz to 8 kHz, 46 ms window, 150 Hz frame rate). Rates 2, 4, 8, 16 and 32 Hz (temporal σ = 0.5/rate s). Scales 0.25, 0.5, 1, 2 and 4 cycles/octave (spectral σ = min(0.5/scale, 1.5) octaves). Upward and downward sweep directions. Filters are zero-mean. Magnitude is averaged over 6 frequency regions, pooled over the frame, then square-rooted. 5×5×2×6. | 300 |
| 5b | `temporal_mod` | Pure temporal-modulation magnitude (same temporal Gabor filters, no spectral filter) for 5 rates × 16 frequency bands, square-rooted. | 80 |
| 6 | `spectral_shape` | Log spectral centroid, log bandwidth, log flatness, log 85 % roll-off, log ratio of 80–1000 Hz to 1–8 kHz energy, and spectral flux (RMS of the rectified log-mel derivative). | 6 |
| 7 | `context` | Causal exponential moving averages (adaptation and recent history) of 16-band log energy with τ = 0.3, 1 and 3 s (48), plus total onset strength and voicing with τ = 0.3 and 1 s (4). | 52 |
| | **Total** | | **710** |

With the harness's 3 extra delays this gives 2840 regression columns.
