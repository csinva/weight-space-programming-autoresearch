# Feature space — 3,852 dims, win = 4096

All closed-form; nothing is learned or fitted. The whole segment is first divided by its RMS, a single scalar, so that the fixed log floors (1e-8) mean the same thing at any input level.

Notation: `LP_τ` is a causal one-pole lowpass over frames, `y[t] = y[t-1] + α(x[t] - y[t-1])`, with α = 1/(1+τ), initialised at x[0]. The acausal LP_τ runs the same filter forward and then backward. **dyn(x)** = [x − LP_4.5, x − LP_15, x − LP_60]. "with dyn" means [x, dyn(x)], which is 4× the base dimensions. When a filter is centred in time, the edges are padded by repeating the edge frame.

| Group | Dims |
|---|---|
| A. Cochleagrams | 1,740 |
| B. Spectrotemporal modulation | 1,336 |
| C. Descriptors | 392 |
| D. Interactions | 249 |
| E. Within-window scattering | 135 |
| **Total** | **3,852** |

## A. Cochleagrams (1,740)
- Hann power spectra of the centred 1024-, 2048- and 4096-sample sub-windows. Each is weighted by 20 fourth-order gammatone power responses, `(1+((f−fc)/b)^2)^-4` with `b = 1.019·ERB(fc)`. The centre frequencies are ERB-spaced from 50 to 8000 Hz and each row sums to 1. The result is log energy **E**: 3 × 20 = **60**.
- E − LP_τ for τ ∈ {1.5, 4.5, 15, 60, 150}: **300**
- Tonic LP_60 and LP_150 of log E and of cube-root energy: **240**
- Acausal LP_60 of log E: **60**
- Past differences E[t]−E[t−k] and future differences E[t+k]−E[t] for k ∈ {3, 6, 12}. Each is given as itself, its positive half [d]₊ and its negative half [−d]₊: 6 × 3 × 60 = **1,080**

## B. Spectrotemporal modulation (1,336)
The kernels are separable complex 2-D Gabors. In time, the Gaussian σ = 0.6/rate s (0.25 s at rate 0). Across channels, σ = 0.6/|scale| channels (2 channels at scale 0). Both kernels are zero-mean except at rate or scale 0, and both are centred. The cochleagram mean is removed before filtering.
- **20-channel 4096 cochleagram:** quadrature energy |·| per channel for 17 rate/scale pairs, 17 × 20 = **340**. The pairs are:
  - rates {1, 3, 6, 12} Hz × scales {0, +0.25, −0.25} cycles/channel
  - rates {3, 6, 12} × scale 0.5
  - rate 0 × scales {0.25, 0.5}
- **40-channel (ERB, 50–8000 Hz) 4096 cochleagram:**
  - rates {0.5, 1, 2, 4, 8, 12} Hz × scales {0, +0.25, −0.25, 0.5}. Note that +0.5 and −0.5 are the same filter, so they count once.
  - For each, the cosine, sine and magnitude, each averaged over the three frequency thirds: 216, with dyn **864**
  - Acausal residuals of the 72 third-averaged magnitudes (magnitude − acausal LP_15): **72**
  - Cross-rate energy products √(m_r · m_{r+1}) for adjacent rates, per scale and third: 5 × 4 × 3 = **60**

## C. Descriptors (392)
These are computed from the 4096-sample Hann spectrum.
- **Scalars (39):**
  - 8-harmonic log-comb pitch salience at 11 F0s, log-spaced from 75 to 450 Hz. For each F0, the salience is Σ_h h^-½ [L(hF0) − L((h−½)F0)], where L is the log spectrum with its cepstral envelope (80 coefficients) removed and then max-filtered over 3 bins.
  - Best F0 (log₂) and its strength, taken from a finer grid of 61 F0s.
  - Voicing: the peak of the window-corrected normalised ACF between 75 and 450 Hz.
  - ACF HNR: 10·log10(r/(1−r)).
  - Cepstral peak prominence: the peak minus the mean over quefrencies of 2.2–13.3 ms.
  - Loudness: log of the total cochlear energy.
  - Spectral moments over 50 Hz–10 kHz: centroid, spread, skewness and log kurtosis.
  - Flatness (log geometric mean / arithmetic mean) and 85 % rolloff.
  - Octave spectral contrast: the top 20 % minus the bottom 20 % of log power in each of the 7 octaves from 62.5 to 8000 Hz.
  - Channel-group contrast: max − min of the 20-channel log cochleagram within 5 groups of 4 channels.
  - Softmax formants F1–F3: softmax(2·envelope)-weighted frequency in 250–900, 850–2500 and 2300–3500 Hz, given as log Hz.
  - Spectral flux: the mean positive change of the log cochleagram.
- **Voicing-gated cochleagram:** voicing × log E (20 channels): **20**
- [39 scalars + 20 gated] with dyn: **236**
- Temporal-Gabor envelopes of the z-scored 39 scalars at 2, 4, 6 and 10 Hz (σ = 0.5/rate s, magnitude of a complex centred Gabor): **156**

## D. Interactions (249)
- **Key scalars (14):** F0, pitch strength, voicing, loudness, flux, F1, F2, F3, HNR, centroid and flatness, plus 3 phonetic band contrasts on the cochleagram (mid−low, high−mid, high−low). The bands are <500 Hz, 500–2000 Hz and ≥2000 Hz.
- The key scalars and their dyn: 14 + 42.
- Pairwise products of the z-scored band contrasts with F0, strength, loudness, flux, F1, F2, F3, HNR and centroid: 27.
- All 83 are z-scored within the segment, clipped to ±8, and expanded as [z]₊, [−z]₊ and z³: **249**

## E. Within-window scattering (135)
- **First order:** analytic Gaussian (Morlet) filters with Q = 4 (FWHM = fc/4) on 9 channels, fc = 125·2^(0.75k) Hz for k = 0…8 (125–8000 Hz). They are applied to the Hann-windowed 4096-sample frame. S1 is the log of the mean envelope, with dyn: **36**
- **Second order:** Q = 1 filters at modulation frequencies of 25–800 Hz (6 octaves) applied to each envelope. The output is log(S2) − log(S1): 9 × 6 = **54**
- **Instantaneous frequency:** the power-weighted mean phase advance, given as relative deviation from fc, with dyn: **36**
- **Phase coherence:** |Σ z̄ₙzₙ₊₁| / Σ|zₙ||zₙ₊₁|: **9**
