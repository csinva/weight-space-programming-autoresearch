# Acoustic feature space (`/work/feature_space.py`)

Output: **H = 1117** float32 columns per 30 Hz frame. `win = 4096`.
Everything is fixed closed-form signal processing; nothing is fitted or estimated from data.

## Front end
* The continuous waveform is rebuilt exactly from the overlapping windows: the central 1470 samples of
  each window are concatenated. This lets the features use sub-frame time resolution and causal history.
* **Sub-frame grid:** 6 sub-frames per 30 Hz frame (hop 245 samples = 5.56 ms, 180 Hz), centred
  symmetrically inside each frame.
* **Auditory spectrogram:** 30 ms Hann window (1323 samples), 2048-point FFT, and 64 triangular filters
  equally spaced on the ERB-rate scale from 80 Hz to 12 kHz (each filter is at least 3 FFT bins wide and
  has unit area). Power is compressed with `log10(P + 1e-6)`. The 1e-6 floor is fixed and assumes
  int16-scale input, as in the examples.

## Feature groups (before frame shifts)
| Group | Dims | Definition |
|---|---|---|
| Spectrogram | 64 | log10 of the mean band power over the frame's 6 sub-frames |
| Spectral onsets | 64 | per band: sum over sub-frames of the half-wave-rectified 11 ms log-power increase, `max(L[j]-L[j-2],0)` |
| Spectral offsets | 32 | the same for decreases, averaged over adjacent band pairs |
| Broadband | 9 | log loudness (total power); broadband onset and offset sums; peakRate (max positive slope of the 28 ms-smoothed log loudness); within-frame loudness range (max - mean); spectral centroid and spread (in ERB-channel units); spectral flatness (log geometric/arithmetic mean); high/low ratio (>~2.3 kHz vs below, log10) |
| Pitch / periodicity | 27 | normalized autocorrelation (NACF): 46 ms Hann window, bias-corrected by the window's autocorrelation, lags for 60-500 Hz. Includes 24 log-spaced F0 bins (8 per octave; each value is the max NACF in the bin, clipped to [0, 1]); harmonicity (max NACF); voicing-weighted log2(F0/150 Hz); and the frame-to-frame change in log F0, weighted by voicing. All are frame means over sub-frames |
| **Core total** | **196** | all of the above |
| Coarse long-lag copy | 34 | 16-band spectrogram (4-channel means), 16-band onsets (4-channel sums), loudness, broadband onset |
| Spectro-temporal modulation | 640 | Chi/Ru/Shamma-style cortical filters applied to the 180 Hz log spectrogram (2-D FFT). Temporal kernel: causal `(rt)^2 exp(-3.5rt) sin(2πrt)`, made analytic, with rates 2, 4, 8, 16 and 32 Hz. Spectral kernel: Mexican hat with scales 0.25, 0.5, 1 and 2 cyc/oct, assuming 10 ERB channels per octave. Two sweep directions (upward and downward), via quadrant selection. Output is the magnitude, averaged over each frame and over 16 frequency regions of 4 channels each: 5 x 4 x 2 x 16 = 640 |
| Slow context | 51 | causal leaky integrators (time constants 0.2, 0.5 and 1.5 s) of the 16-band log spectrogram and the loudness: 17 x 3. These carry adaptation and context information |

## Frame shifts (temporal coverage)
The harness adds delays of 0-3 frames and a fixed 100 ms lag, so a column `X[t+s]` covers audio
`(1-s) ... (4-s)` frames (x 33 ms) before the neural sample.
| Block | Shift `s` | Effective audio lags | Dims |
|---|---|---|---|
| Core | +2 | 33-133 ms | 196 |
| Core | -2 | 167-267 ms | 196 |
| Coarse copy | -6 | 300-400 ms | 34 |
| Spectro-temporal modulation | +2 | 33-133 ms (the kernels themselves integrate up to several hundred ms of history) | 640 |
| Slow context | +2 | 33-133 ms (+ the integration time) | 51 |
| **Total** | | | **1117** |

At the segment edges, shifted copies repeat the first or last frame.
Run time is about 14 s per 2-minute segment on 4 CPU cores (numpy/scipy only).
