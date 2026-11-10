# Acoustic feature space (`feature_space.py`)

**H = 1478 dimensions** at 30 Hz. `win = 4096`. Every step is closed-form signal processing with fixed
parameters; nothing is learned or estimated from data.

## Front end (shared)
- Overlapping windows are stitched back into the continuous waveform of the segment. The mean is removed and
  the signal is scaled to unit RMS over the segment (a fixed gain normalization, so absolute recording level does
  not matter).
- **Cochlear-like spectrogram**: 64 channels equally spaced on the ERB-number scale from 50 Hz to 15 kHz
  (triangular weights, about 0.58 ERB spacing). Power comes from a Hann STFT with 2048-sample windows for channels
  below 1 kHz and 1024-sample windows for channels at or above 1 kHz. The hop is 245 samples, giving
  **6 sub-frames per 30 Hz frame** (180 Hz).
  Log compression: `10*log10(P + 1e-8)` (-80 dB floor relative to the segment RMS).
- Features computed at 180 Hz are mean-pooled over the 6 sub-frames of each frame, which spans ±16.7 ms
  around the frame centre.

## Groups
| # | Group | Dims | Description / parameters |
|---|-------|------|--------------------------|
| 1 | `logspec` | 64 | Log-power (dB) of the 64 ERB channels, frame mean. |
| 2 | `powspec` | 32 | Power-law (cube-root) compressed power, with adjacent channel pairs merged into 32 bands. |
| 3 | `onset` | 32 | Half-wave rectified positive temporal derivative of the log spectrogram (dB change over 11 ms). The spectrogram is first smoothed across channels (Gaussian, σ = 1 channel), then merged into 32 bands and summed within the frame. |
| 4 | `offset` | 32 | Same as `onset`, using the negative derivative. |
| 5 | `envelope` | 9 | Broadband level in dB: mean, max, min and std over the 6 sub-frames. Rectified rise and fall of the broadband level. Spectral flux in its positive and negative parts. Maximum rise within the frame. |
| 6 | `subframe_env` | 54 | Sub-frame (5.6 ms) envelope detail: dB level in 8 broad bands (8 ERB channels each) × 6 sub-frames, plus broadband level × 6 sub-frames. |
| 7 | `modulation` | 928 | Spectro-temporal modulation energy, computed with a Gabor STRF-like filterbank on the 180 Hz log spectrogram (details below). |
| 8 | `spectral_stats` | 9 | Spectral centroid and spread (ERB), log spectral flatness, normalized spectral entropy, 85 % roll-off channel, spectral slope (dB/ERB), mid-low and high-mid band level differences (<500 Hz, 0.5–4 kHz, >4 kHz), and log zero-crossing rate. |
| 9 | `pitch` | 58 | Periodicity features (details below). |
| 10 | `context` | 260 | Temporal context for a compact core set of 65 features (details below). |

### Modulation filterbank (group 7)
- The log spectrogram has its channel means removed. Temporal filtering uses analytic complex Gabor filters in
  the frequency domain. **Rates are 1, 2, 4, 8, 16 and 32 Hz**, each with a Gaussian passband of σ = rate/3
  (about 1.2 octaves at half maximum) and zero DC response.
- Spectral filtering applies complex Gabor kernels along the channel axis. **Scales are 1/24, 1/12, 1/6 and
  1/3 cycles per channel**, roughly 0.33–2.7 cycles per octave above 500 Hz. Each kernel has σ = 3/(2π·scale)
  channels, reflect boundaries and zero DC response.
- Each rate × scale pair gives an upward-sweep and a downward-sweep response, using the kernel and its complex
  conjugate. Each rate also gets a pure temporal-modulation response (no spectral filter). Each scale also gets a
  pure spectral-modulation response, computed on a temporally low-passed (4 Hz Gaussian) spectrogram.
- That gives 4 + 6 × (1 + 4 × 2) = 58 filters. The magnitude of each response is averaged within
  **16 frequency regions** (4 adjacent channels each) and over the frame, then compressed with `log1p`.
  58 × 16 = 928.

### Pitch / periodicity (group 9)
The autocorrelation is computed per frame from a 2048-sample Hann window at the frame centre, zero-padded to 4096.
It is the inverse FFT of the magnitude spectrum restricted to 50–4000 Hz, normalized by lag 0 and divided by the
window's own autocorrelation.
- 4 summary features:
  - voicing strength, the peak normalized ACF over F0 60–480 Hz
  - soft voicing weight `clip((v-0.2)/0.25, 0, 1)`
  - voicing-weighted log2 F0 relative to 150 Hz
  - voicing-weighted relative pitch
- 36 pitch-salience bins: the maximum normalized ACF within log-spaced F0 bins from 60 to 480 Hz
  (12 bins per octave).
- 12 relative-pitch bins: log2 F0 minus the segment median log2 F0 of voiced frames (voicing > 0.35), encoded
  with Gaussian bins (centres from −1 to +1 octave, σ = 1/11 oct) and weighted by voicing. This is
  speaker-normalized pitch in the style of Tang et al. (2017).
- 6 pitch-change bins: the frame-to-frame change in log2 F0, with Gaussian bins centred at ±0.02, ±0.06 and
  ±0.12 oct/frame (σ = 0.03), weighted by voicing in both frames.

### Temporal context (group 10)
The core set has 65 features:
- 32 log bands (pairs of `logspec` channels)
- 16 onset bands
- broadband level, rise, fall and spectral flux
- voicing
- 12 relative-pitch bins

The harness supplies lags 0–100 ms on top of a fixed 100 ms alignment lag. This group extends that window in
both directions:
- **Look-ahead copy** (65 dims): the core features from 2 frames (67 ms) in the future. Combined with the harness
  delays, this covers response latencies of 33–133 ms, which the 100 ms alignment otherwise misses.
- **Causal exponential moving averages** with time constants of **0.25 s, 0.75 s and 2.5 s** (3 × 65 dims). These
  give longer acoustic history, which lets the linear readout capture adaptation and context effects.

## Robustness
`nan_to_num` is applied to the output. Columns that are constant within a segment (for example, pitch bins
during silence) get a fixed-seed dither of 1e-3, so that per-segment z-scoring stays finite.
Runtime is about 1–2 s per 2-minute segment on CPU.
