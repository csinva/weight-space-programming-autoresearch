# Acoustic feature space (`/work/feature_space.py`)

**H = 892 dimensions.** `win = 4096`. Everything uses closed-form signal processing with fixed parameters. Nothing is learned, fitted or decomposed. The only per-segment quantities are simple normalisations of the input itself: log floors, loudness and voicing reference percentiles, and running pitch statistics.

## Framing
Neighbouring windows (4096 samples, hop 1470) overlap, so the code reconstructs the continuous segment waveform exactly. It checks that the reconstruction is bit-identical to the source. All features are then computed on the whole waveform:
- Fine grid: hop 245 samples (180 Hz). Medium grid: hop 490 (90 Hz). Both are aligned so that grid points fall exactly on the frame centres.
- Pooling: each feature is projected onto the 30 Hz frame grid with a normalised Hann kernel spanning ±1 frame (±33 ms, zero at the ends), which acts as an anti-aliasing low-pass.
- Some features look ahead by 1–2 frames (33/67 ms). The harness's fixed 100 ms lag plus its 0–3 frame delays covers latencies of 100–200 ms; the look-ahead copies extend that down to about 33 ms, for early primary-auditory responses.
- STFTs use a Hann window with `center=True` and reflect padding. Log power uses a per-band floor of −30 dB relative to the band's segment mean (`log(P + 1e-3·mean_band(P))`), so it does not depend on gain.

## Feature groups

| # | Group | Parameters | Dims |
|---|-------|------------|-----:|
| 1 | Log-mel spectrogram | 2048-pt STFT (46 ms) at 180 Hz; 80 triangular mel bands from 40 Hz to 16 kHz; band power pooled per frame, then log | 80 |
| 2 | Look-ahead log-mel | 1024-pt STFT (23 ms), 32 mel bands from 80 Hz to 16 kHz, pooled then log, shifted +1 and +2 frames into the future | 64 |
| 3 | Spectral onsets / offsets | 32-band fine log-mel, smoothed over 3 fine frames; difference across 11 ms, half-wave rectified (positive = onset, negative = offset); pooled. Onsets also given at +1 frame | 96 |
| 4 | Broadband envelope dynamics | Fine log energy pooled; envelope smoothed with a ~100 ms Hann window, then the positive and negative parts of its derivative; peakRate (local maxima of the positive derivative), at 0 and +1 frame; max, min and std of fine log energy within ±1 frame | 8 |
| 5 | Loudness level (soft one-hot) | Frame energy in dB relative to the segment's 99th percentile; 10 Gaussian bins centred at −45…0 dB (5 dB spacing, σ = 3 dB) | 10 |
| 6 | Slow spectral context | Causal exponential moving averages of the 32-band pooled log-mel, τ = 0.25 s and 1 s | 64 |
| 7 | Syllabic-rate envelope | 2–8 Hz band-pass (raised-cosine edges at 1–2 and 8–12 Hz) of fine log energy, analytic signal: amplitude, cos(phase), sin(phase) | 3 |
| 8 | Spectral shape | log centroid, log bandwidth, spectral flatness (log geometric/arithmetic mean), log 85% roll-off, normalised spectral entropy, log ratio of energy above vs below 2 kHz (50 Hz–16 kHz); log zero-crossing rate over ±735 samples | 7 |
| 9 | Pitch / periodicity | Normalised autocorrelation (Boersma) on 46 ms frames at 90 Hz, computed from the power spectrum low-passed at 5 kHz (zero-padded 4096-pt FFT, window-ACF corrected). **(a)** Periodicity salience: max normalised ACF in 36 log-spaced F0 bins from 55 to 800 Hz (36). **(b)** Absolute pitch: 20 Gaussian soft bins of log2 F0 from 70 to 450 Hz, weighted by voicing (20). **(c)** Relative pitch: F0 z-scored against a voicing-weighted Gaussian window (σ = 2 s, sd floor 0.15 oct), 12 bins from −2.5 to 2.5, voicing-weighted (12). **(d)** Pitch change: semitones per 11 ms, 8 bins from −1.2 to 1.2, weighted by voicing in both frames (8). **(e)** Voicing weight (clip((r−0.45)/0.45), gated at −50 dB re the 99th-percentile energy), raw ACF peak, voicing×logF0, voicing×relative pitch (4). F0 = best ACF peak with a slight short-lag preference, refined by parabolic interpolation | 80 |
| 10 | Spectro-temporal modulation energy | Log-frequency log-spectrogram: 64 triangular bands from 100 Hz to 12.8 kHz (9.1 channels/octave), 2048-pt STFT at 90 Hz. Filtered by a bank of complex 2-D Gabor filters (zero-mean): rates {2, 4, 8, 16, 32} Hz × scales {0.25, 0.5, 1, 2, 4} cyc/oct × 2 directions (upward/downward sweeps), plus pure temporal (5 rates, σ = 0.4 oct) and pure spectral (5 scales, σ_t = 30 ms) filters, giving 60 filters. Gaussian envelope σ_t = 0.6/rate s, σ_f = 0.6/scale oct. Output is the log magnitude, pooled over ±1 frame, averaged over 8 frequency regions of 8 channels each (0.875 oct) | 480 |
| | **Total** | | **892** |

## Notes
- A deterministic dither (fixed seed, 1e-6 × the column's maximum absolute value) is added so that no column is exactly constant. This keeps per-segment z-scoring finite.
- Runtime is about 3 s per 2-minute segment on 4 CPU cores. The code runs in torch and works on GPU unchanged (`FeatureSpace(device='cuda')`).
