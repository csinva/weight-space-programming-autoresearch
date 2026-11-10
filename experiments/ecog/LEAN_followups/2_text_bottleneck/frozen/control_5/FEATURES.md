# Acoustic feature space (H = 382)

`FeatureSpace.win = 4096` samples (44.1 kHz). The output is 30 Hz frames. Every feature is closed-form signal
processing with fixed parameters. The only quantities taken from the data are per-segment normalisation
constants: the log floors (1e-5 × the segment's mean power) and the reference for relative pitch (the
voicing-weighted mean and SD of log-F0). Every temporal filter across frames is causal: it uses only the
current and past frames. The harness adds its own delays of 1–3 frames and z-scores the columns.

| # | Group | Parameters | Dims |
|---|-------|------------|-----:|
| 1 | **Log-mel spectrogram** | 2048-sample Hann window (46 ms) centred on the frame; 64 triangular mel bands, 60 Hz–10 kHz; log(power + 1e-5·segment mean) | 64 |
| 2 | **Level / loudness** | log total power; log A-weighted power | 2 |
| 3a | **Spectral flux (onsets) by band** | half-wave-rectified frame-to-frame change of log-mel, pooled into 16 contiguous bands | 16 |
| 3b | **Broadband onset/offset** | mean positive flux; mean negative flux (offsets); "peakRate", the rectified derivative of the A-weighted log-envelope after a 30 ms causal exponential moving average (EMA) | 3 |
| 4a | **Fine-time onsets (within one frame)** | 7 sub-frames of 512 samples (11.6 ms) at a 245-sample hop, covering the 33 ms frame hop; 16 mel bands (100 Hz–10 kHz); max rectified sub-frame-to-sub-frame log-power increase per band | 16 |
| 4b | **Fine-time envelope** | broadband log envelope of each of the 7 sub-frames | 7 |
| 4c | **Fine-time broadband onset** | max rectified increase of the 4b envelope | 1 |
| 5a | **Slow spectral context (adaptation/history)** | causal EMA of the 16-band log-mel at τ = 0.2, 0.6 and 2.0 s. The linear readout can form "current minus recent average" contrasts from these | 48 |
| 5b | **Onset density** | causal EMA (τ = 0.5 s) of mean positive flux | 1 |
| 6 | **Spectral shape** | log centroid, log spread, log spectral flatness, mel entropy, log 85 % roll-off | 5 |
| 7a | **Absolute pitch** | autocorrelation of the magnitude spectrum (50 Hz–5 kHz) on the 2048-sample window, corrected for window bias; F0 search 70–400 Hz with parabolic interpolation; 12 Gaussian bins over log-F0, scaled by soft voicing | 12 |
| 7b | **Relative pitch** | log-F0 z-scored within the segment (voicing-weighted); 8 Gaussian bins over z = −2…2, scaled by soft voicing | 8 |
| 7c | **Pitch change** | Δlog₂F0 between frames; 5 Gaussian bins at −0.1, −0.04, 0, +0.04, +0.1 oct (SD 0.03), scaled by voicing in both frames | 5 |
| 7d | **Voicing / harmonicity** | peak of the normalised ACF; soft voicing (sigmoid, centre 0.5, slope 0.07) | 2 |
| 8 | **Spectro-temporal modulation energy** | operates on the mean-removed 64-band log-mel. Causal complex temporal kernels (order-3 gammatone envelope) at rates of 2, 4 and 8 Hz, combined with complex Gabor spectral kernels at 0.5, 1 and 2 cycles/oct in both sweep directions (3×3×2 = 18). Adds pure temporal rates (3) and pure spectral scales (3, on a 0.1 s EMA). Takes the magnitude, pools it into 8 frequency regions, then takes log(· + 1e-3). | 24×8 = 192 |
| | **Total** | | **382** |

With the harness's 4 delays, the regression has 1528 columns. Runtime is about 1 s per 2-minute segment
on a CPU, using numpy/scipy only.
