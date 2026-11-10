# Feature space: 3,852 dims, win = 4096

Everything is fixed, closed-form signal processing. Nothing is learned or fitted. Notation:
- `LP_τ` is a causal one-pole lowpass over frames, with α = 1/(1+τ), initialised at frame 0.
- `acausal LP_τ` is the same filter run forward and then backward.
- `dyn(x)` = [x − LP_4.5, x − LP_15, x − LP_60], i.e. 3× the dims of x.
- Input windows are divided by the segment RMS (taken over the central 1470 samples of all windows) so that
  log floors (ε = 1e-8) do not depend on the recording gain.

| Group | Dims |
|---|---|
| Cochleagrams | 1740 |
| Spectrotemporal modulation | 1375 |
| Descriptors | 340 |
| Interactions | 321 |
| Within-window scattering | 76 |
| **Total** | **3852** |

## 1. Cochleagrams (1740)
- Hann power spectra are computed on centred 1024/2048/4096-sample sub-windows. Each is weighted by 20
  fourth-order gammatone power responses `(1+((f−fc)/1.019·ERB(fc))²)^−4`. The centre frequencies are
  ERB-rate spaced over 50–8000 Hz, and each response is normalised to unit sum. The result is the energy E
  for 3 resolutions × 20 channels = 60 channels.
- log E: 60
- log E − LP_τ for τ ∈ {1.5, 4.5, 15, 60, 150}: 300
- Tonic LP_60 and LP_150 of log E and of E^(1/3): 240
- Acausal LP_60 of log E: 60
- Past differences x[t]−x[t−k] and future differences x[t+k]−x[t] for k ∈ {3, 6, 12} frames. Each
  difference is given raw, as [d]₊ and as [−d]₊: 2·3·3·60 = 1080

## 2. Spectrotemporal modulation (1375)
- **2-D Gabor quadrature energy** on the 20-channel 4096 log-cochleagram. Each Gabor is separable: a
  complex temporal Gabor (σ_t = 0.5/rate s, or a 0.25 s Gaussian for rate 0) times a complex spectral Gabor
  (σ = 1/(2|scale|) channels, or a 3-channel Gaussian for scale 0). Both are zero-DC. The output is the
  modulus per channel. There are 17 (rate Hz, scale cyc/ch) pairs:
  - (0, 0.125), (0, 0.25), (0, 0.5)
  - (2|4|8|12, 0)
  - (2|4|8|12, ±0.25)
  - (4, ±0.5)

  17 × 20 = 340 dims.
- **40-channel cochleagram** (4096 spectrum, 40 ERB-spaced gammatones over 50–8000 Hz). Rates are
  {0.5, 1, 2, 4, 8, 12} Hz and scales are {0, ±0.25, ±0.5}, giving 30 filters. Cosine (real), sine (imag)
  and magnitude are each averaged over the frequency thirds (channels 0–12, 13–25, 26–39): 270.
- dyn of the 90 magnitudes: 270
- Acausal residuals (x − acausal LP_15) of all 270: 270
- Cross-rate energy products: geometric mean of the magnitudes for every pair of rates (15 pairs),
  within each scale and third. 15 × 5 × 3 = 225 dims.

## 3. Descriptors (340), all from the 4096 Hann spectrum unless stated
The base set has 58 dims:
- 8-harmonic log-comb pitch salience at 11 F0s, log-spaced over 75–450 Hz (11 dims). Salience is
  Σ_h w_h [L(hF0) − ½(L((h−½)F0) + L((h+½)F0))], with w_h ∝ h^−½ and L = log power.
- Best F0 (log2), taken from a 63-point grid, and its strength, i.e. the maximum salience (2)
- ACF HNR: the autocorrelation is normalised by the window autocorrelation, its maximum is taken over lags
  of 1/450–1/75 s, and HNR = 10·log10(r/(1−r)) (1)
- Cepstral peak prominence: the peak over 1/450–1/75 s, measured above a fixed least-squares line over
  quefrencies of 1–22 ms (1)
- Spectral centroid, spread, skewness and log-kurtosis (50–11025 Hz), log flatness (50–8000 Hz) and the
  85 % rolloff (log2 Hz) (6)
- Octave spectral contrast: the mean of the top 20 % minus the bottom 20 % of the log spectrum in each of
  7 octaves from 100 Hz to 12.8 kHz (7)
- Channel-group ("phonetic band") contrast on the 20-channel cochleagram. The groups are < 300 Hz,
  300–1000 Hz, 1–2.5 kHz, 2.5–5 kHz and > 5 kHz; each is the group mean minus the all-channel mean (5)
- Softmax formants: on the cepstrally smoothed envelope (40 coefficients), a softmax(2·env)-weighted
  frequency within each of 200–1000, 800–2800 and 1800–3800 Hz, in log2 Hz (3)
- Spectral flux: the mean positive frame difference of the log cochleagram (1)
- Loudness: the log of the summed cochleagram energy (1)
- Voicing-gated cochleagram: sigmoid((strength − 3)/0.75) × max(log E − log ε, 0) for the 20 channels (20)

The group then adds:
- The base set plus its dyn: 58 × 4 = 232
- Temporal-Gabor envelopes, i.e. the modulus of a complex Gabor at 2/4/6/10 Hz, of the 27 scalars (all
  base dims except the salience vector and the gated cochleagram): 108

## 4. Interactions (321)
- Key scalars (16): loudness, flux, F0, pitch strength, HNR, CPP, centroid, flatness, F1–F3 and the 5
  band contrasts.
- Each key scalar is z-scored within the segment, and the dyn of the z-scores is added (48).
- Pairwise products (35): the 5 band contrasts × {F0, strength, loudness, flux, F1, F2, F3}.
- Further products (8): strength·loud, flux·loud, F1·F2, strength·flux, HNR·loud, F0·strength,
  centroid·loud and flatness·flux.
- The resulting 107 columns are z-scored within the segment, clipped to ±6 and expanded as [z]₊, [−z]₊
  and z³. 107 × 3 = 321 dims.

## 5. Within-window scattering (76)
- The windowed 4096-sample frame goes through an analytic first-order Morlet bank with Q = 4: 4 filters per
  octave over 9 octave channels from 62.5 Hz. Filter centres are capped at 20 kHz, so the top octave's 4
  filters coincide.
- The 4 moduli in each octave are summed to give U_m.
- S1: the log mean of U_m over the central half-window (9)
- S2: U_m is passed through a Q = 1 Morlet bank at 25/50/100/200/400 Hz, and the log of
  mean|U_m ∗ ψ| / S1 is taken (45)
- Instantaneous frequency (log2 of the amplitude-weighted phase-advance frequency relative to the channel
  centre) and phase coherence (|Σ z_{t+1} z̄_t| / Σ|·|) per channel (18)
- Broadband log energy (S0), broadband IF, broadband coherence and the mean channel coherence (4)
