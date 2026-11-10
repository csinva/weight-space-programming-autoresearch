# Feature space: 3,852 dims, win = 4096

Everything is closed-form and uses fixed parameters. Nothing is fitted or learned. The only
within-segment operations are lowpasses over frames, z-scoring and log floors set relative to the
segment's mean energy (1e-6 × mean), which make the features independent of overall audio scale.

Notation: `LP_τ` is a causal one-pole lowpass over frames, `y[t] = y[t-1] + α(x[t] - y[t-1])` with
α = 1/(1+τ), initialised at x[0]. The acausal LP is the same filter run forward and then backward.
`dyn(x) = [x, x-LP_4.5, x-LP_15, x-LP_60]` gives 4× the dims of x. Temporal convolutions over frames are
centred, with reflect padding.

| Group | Dims |
|---|---|
| 1. Cochleagrams | 1,800 |
| 2. Spectrotemporal modulation | 1,045 |
| 3. Descriptors | 416 |
| 4. Interactions | 303 |
| 5. Within-window scattering | 288 |
| **Total** | **3,852** |

## 1. Cochleagrams (1,800)
- Hann power spectra of the centred 1024-, 2048- and 4096-sample sub-windows, normalised by Σw².
- Each spectrum is weighted by 20 fourth-order gammatone power responses
  `(1+((f-fc)/b)²)^-4` with b = 1.019·ERB(fc) and fc ERB-spaced over 50–8000 Hz. Rows sum to 1.
  This gives E with 3 × 20 = 60 channels. L = log E.
- Dims:
  - L: 60
  - L − LP_τ for τ ∈ {1.5, 4.5, 15, 60, 150}: 300
  - Tonic LP_60 and LP_150 of L and of cube-root energy (E/mean E)^(1/3): 240
  - Acausal LP_60 of L and of the cube-root energy: 120
  - Past difference L[t]−L[t−k] and future difference L[t+k]−L[t] for k ∈ {3, 6, 12}, each as
    [d, [d]₊, [−d]₊]: 1,080

## 2. Spectrotemporal modulation (1,045)
2-D Gabors are separable: a complex Gabor over channels (σ = 0.5/|scale| channels, or a Gaussian with
σ = 2 when scale = 0) and a complex Gabor over time (σ = 0.5 cycle, i.e. 15/rate frames; a Gaussian
with σ = 3 frames when rate = 0). Both are zero-mean where the modulation frequency is nonzero.
- **Quadrature energy on the 20-channel 4096 log cochleagram:** 340 dims = 17 pairs × 20 channels,
  as log(|Y|² + floor). The 17 (rate Hz, scale cyc/ch) pairs are (0, 0.25), (0, 0.5) and
  {3, 6, 12} × {0, ±0.25, ±0.5}.
- **40-channel log cochleagram** (4096 spectrum, 40 gammatones over 50–8000 Hz): 30 filters, rates
  {0.5, 1, 2, 4, 8, 12} Hz × scales {0, ±0.25, ±0.5}. Each output is averaged over 3 frequency thirds
  (13/13/14 channels).
  - Cosine (real) and sine (imaginary) parts: 180
  - Magnitude, normalised by its segment mean: 90
  - dyn residuals (x − LP_4.5/15/60) of the magnitudes: 270
  - Acausal residual, magnitude − acausal LP_60: 90
  - Cross-rate energy products, mag(rate_i) × mag(rate_i+1) for 5 adjacent rate pairs × 5 scales
    × 3 thirds: 75

## 3. Descriptors (416), from the 4096 Hann spectrum
There are 42 scalars:
- **Pitch salience (11):** an 8-harmonic log-comb with peaks at h·F0, troughs at (h±½)·F0 and
  weights 1/√h, run on 121 log-spaced F0s over 75–450 Hz and max-pooled into 11 log-spaced F0 bins.
- **Best F0 and strength (2):** log2 of the argmax F0 and the max salience.
- **ACF HNR (1):** the window-corrected normalised autocorrelation, peak r over lags 98–588 samples,
  as log(r/(1−r)).
- **Cepstral peak prominence (1):** peak minus a fixed least-squares line, over quefrencies
  1/450–1/75 s.
- **Spectral shape (6):** centroid, spread, skewness and log kurtosis over 50–8000 Hz, plus
  flatness (log geometric/arithmetic mean) and the 85% rolloff.
- **Octave spectral contrast (7):** bands with edges 50/200/400/800/1600/3200/6400/11025 Hz, as
  log(top 20%) − log(bottom 20%).
- **Channel-group contrast (4):** max − min of the 20-channel log cochleagram within 4 groups of 5
  channels.
- **Softmax formants (3):** the softmax-weighted frequency (β = 2 per log unit) of a liftered log
  spectrum (40 cepstral coefficients), in 250–900, 800–2500 and 2000–3500 Hz.
- **Flux, loudness, voicing (3):** spectral flux is the mean of [Δ log cochleagram]₊. Loudness is
  log ΣE. Voicing is sigmoid(10·(r − 0.5)).
- **Phonetic band contrasts (4):** band log energies for <300, 300–1k, 1k–3k and >3k Hz, taken as
  B2−B1, B3−B2, B4−B3 and B4−loudness.

| Part | Dims |
|---|---|
| dyn(42 scalars) | 168 |
| dyn(voicing × 20-channel log cochleagram) | 80 |
| Temporal-Gabor envelopes of the z-scored scalars at 2/4/6/10 Hz (σ = 0.5 cycle, quadrature magnitude) | 168 |

## 4. Interactions (303)
- **Key scalars (16):** loudness, flux, log F0, pitch strength, HNR, CPP, F1, F2, F3, centroid,
  flatness, voicing and the 4 phonetic band contrasts. With dyn this is 64 dims.
- **Pairwise products of z-scored factors (37):**
  - The 4 phonetic contrasts × {log F0, strength, loudness, flux, F1, F2, F3}: 28
  - strength×loudness, flux×loudness, voicing×F1/F2/F3, strength×flux, F0×loudness,
    HNR×loudness and CPP×loudness: 9
- All 101 columns are z-scored within the segment and expanded as [z]₊, [−z]₊ and clip(z, ±4)³/8,
  giving 303 dims.

## 5. Within-window scattering (288)
The Hann-tapered 4096 window is filtered with 9 analytic Gaussian (Morlet-like) Q=4 filters, with fc =
125·2^(0.75k) Hz for k = 0..8 (125–8000 Hz).
- **First order (9):** log mean |x_c|.
- **Second order (45):** each envelope |x_c| is filtered with Q=1 wavelets at 25/50/100/200/400 Hz,
  as log mean |·| minus the first order.
- **Instantaneous frequency (9):** the phase advance of x_c at lag fs/(fc/4), relative to the fc
  phase, in cycles.
- **Phase coherence (9):** |Σ x_c[t+lag]·x_c*[t]| / Σ|x_c[t+lag]||x_c[t]|.

These 72 scalars get dyn, giving 288 dims.

## Runtime
About 3 s per 2-minute segment on 4 CPU threads (`python /work/check.py` passes, H = 3852).
