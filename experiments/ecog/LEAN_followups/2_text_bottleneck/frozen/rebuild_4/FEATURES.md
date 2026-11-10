# Feature space: 3,769 dims, win = 4096

Implemented in `feature_space.py`. All parameters are fixed constants; nothing is learned or fitted. The only
quantities that depend on the segment are ones the harness already allows: the segment RMS used to normalise
the audio (so the features do not depend on input scale), causal/acausal filtering across frames, and
z-scoring within the segment, which is used only inside the interaction group.

Shared notation (frames at 30 Hz):
- `LP_τ`: causal one-pole lowpass over frames, `y_t = y_{t-1} + α (x_t − y_{t-1})` with `α = 1/(1+τ)`, initialised at `x_0`.
  "Acausal LP" means a forward pass followed by a backward pass (zero phase).
- `dyn(x) = [x − LP_4.5, x − LP_15, x − LP_60]`. "+dyn" means `[x, dyn(x)]`, which is 4× the dimensions.
- The audio is divided by the segment RMS, and the power floor is `EPS = 1e-10`.

## A. Cochleagrams: 1,740
Hann power spectra of the centred 1024-, 2048- and 4096-sample sub-windows. Each is weighted by 20 fourth-order
gammatone power responses, `(1+((f−fc)/b)²)^−4` with `b = 1.019·ERB(fc)`, ERB-spaced from 50 to 8000 Hz, and then
logged. `L` is the 3 × 20 = 60 log energies.

| features | dims |
|---|---|
| `L` (3 window sizes × 20 channels) | 60 |
| `L − LP_τ`, τ ∈ {1.5, 4.5, 15, 60, 150} | 300 |
| tonic `LP_60`, `LP_150` of log energy and of cube-root energy | 240 |
| acausal `LP_60` of `L` | 60 |
| past `L_t − L_{t−k}` and future `L_{t+k} − L_t` differences, k ∈ {3, 6, 12}, each with rectified halves `[d]₊`, `[−d]₊` | 1080 |

## B. Spectrotemporal modulation: 1,045
- **2-D Gabor quadrature energy** (`|complex Gabor * L|`) on the 20-channel 4096 cochleagram. There are 17 rate/scale
  pairs: rate 0 Hz × scale {0.125, 0.25, 0.5}; rates {1, 3, 6, 12} Hz × scale {0, +0.25, −0.25}; and 4 Hz × ±0.5
  cycles/channel. The Gaussian envelopes are σ_t = 1/(π·rate) (0.2 s at rate 0, minimum 1 frame) and
  σ_c = 1/(π·|scale|) (3 channels at scale 0, minimum 0.75). Kernels are zero-mean, and the edges are reflect-padded.
  This gives 17 × 20 = **340** dims.
- **40-channel cochleagram** (4096 spectrum, 50 to 8000 Hz), using rates {0.5, 1, 2, 4, 8, 12} Hz × scales
  {0, ±0.25, ±0.5}, which is 30 filters. The cosine (real), sine (imag) and magnitude outputs are each averaged over
  the frequency thirds (channels 0–12, 13–26 and 27–39), giving 3 × 90 = **270**. On top of that:
  - dyn of the magnitudes: **270**
  - acausal residuals, magnitude − acausal LP_60: **90**
  - cross-rate energy products, `sqrt(mag_r · mag_{r+1})` for adjacent rates, per scale and third: 5 × 5 × 3 = **75**

## C. Descriptors: 456
These are computed from the 4096 Hann spectrum, except where noted.

There are 47 scalars:
- 8-harmonic log-comb pitch salience (harmonics weighted by 1/√h, inter-harmonic log-power subtracted) on a
  ±1-bin max-smoothed log spectrum. It is evaluated on a 1/24-octave grid from 75 to 450 Hz, then max-pooled to
  11 log-spaced F0s: **11**
- best F0 (log₂, parabolic refinement) and its salience strength: **2**
- ACF HNR (Boersma window correction, lags 2.2–13.3 ms): **1**
- cepstral peak prominence (peak relative to the regression line over quefrency, with a fixed design matrix): **1**
- voicing gate `clip((r_max − 0.4)/0.4, 0, 1)`: **1**
- log centroid, log spread, asinh skewness, log kurtosis (50 Hz to 11 kHz), flatness, log 85% rolloff, log
  loudness: **7**
- octave spectral contrast (8 octaves from 62.5 Hz to 16 kHz, top 20% minus bottom 20% of log power) and
  channel-group contrast (8 groups of 5 channels of the 40-channel cochleagram, max − min): **16**
- softmax formants (β = 1.5 on the cepstrally smoothed log spectrum, 30 coefficients) in F1 250–1000, F2 800–2800
  and F3 1800–3600 Hz: log frequency and relative amplitude for each: **6**
- spectral flux (mean positive and mean absolute frame-to-frame change of the 20-channel log cochleagram): **2**

Then:
- voicing-gated cochleagram, `gate · log(1 + E/EPS)` on the 20 channels: **20**
- (47 scalars + 20 gated channels) + dyn: 67 × 4 = **268**
- temporal-Gabor envelopes (magnitude of a complex Gabor along frames, σ = 1/(π·rate), zero-mean) at
  2, 4, 6 and 10 Hz of the 47 z-scored scalars: **188**

## D. Interactions: 240
- There are 14 key scalars:
  - 4 phonetic band contrasts on log band power: fricative (3–8 kHz) − F1 band (0.3–1 kHz),
    F2 band (1–3 kHz) − F1 band, low (50–300 Hz) − F1 band, and fricative − (F1 + F2 bands)
  - 6 more: pitch strength, voicing-gated log F0, loudness, positive flux, F1 and F2
  - 4 more: HNR, CPP, centroid and flatness
- Their dyn adds 42, and the pairwise products of the 4 band contrasts × the 6 others add 24, giving 80.
- Everything is z-scored within the segment and expanded as `[z]₊, [−z]₊, z³`: 80 × 3 = **240**

## E. Within-window scattering: 288
- First order: analytic Gaussian (Morlet-like) filters with Q = 4 (FWHM = fc/4), 36 centres from 40 Hz × 2^(k/4)
  (9 octaves, up to about 17 kHz). The modulus is computed at a decimated baseband rate. The log of the
  Hann-weighted mean modulus gives `S1` (36), and S1 + dyn is **144**.
- Second order: on 9 channels (one per octave, the 3rd filter of each), the Hann-windowed envelope is filtered
  with Q = 1 Morlets at 8 modulation frequencies from 32 to 2000 Hz. The output is log(mean modulus / S1):
  9 × 8 = **72**.
- Instantaneous frequency and coherence on the same 9 channels. The first is the amplitude-weighted mean
  phase-derivative deviation from fc divided by the filter bandwidth. The second is the lag-1 phase coherence,
  reported as −log(1 − coherence). These 18 dims + dyn give **72**.

## Total
A 1740 + B 1045 + C 456 + D 240 + E 288 = **3,769** (≤ 4000). Runtime is about 5 s per 2-minute segment on 4 CPU
cores (numpy/scipy only).
