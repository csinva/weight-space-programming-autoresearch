# Feature space — 3,837 dims, win = 4096

All closed-form, fixed parameters. Only within-segment operations that depend on the data: the
segment's audio is divided by its global RMS (so log-power floors are relative, EPS = 1e-6 ≈ −60 dB),
the STM cochleagrams have their per-channel segment mean removed before filtering, and the interaction
block is z-scored within the segment.

Notation: frames at 30 Hz. `LP_τ` = causal one-pole lowpass over frames, α = 1/(1+τ), initialised at
the first frame. Acausal LP = the same filter run forward then backward. **dyn** = x − LP_τ for
τ ∈ {4.5, 15, 60} (3× the input dims).

| # | Group | Dims |
|---|-------|-----:|
| 1 | Cochleagrams | 1380 |
| 2 | Spectrotemporal modulation, 20-ch (17 Gabors) | 680 |
| 3 | Spectrotemporal modulation, 40-ch | 705 |
| 4 | Descriptors (+dyn, +temporal-Gabor envelopes) | 456 |
| 5 | Interactions | 168 |
| 6 | Within-window scattering (+dyn) | 288 |
| 7 | 40-ch log cochleagram (+dyn) | 160 |
| | **Total** | **3837** |

## 1. Cochleagrams (1380)
Centred Hann sub-windows of 1024, 2048 and 4096 samples → power spectra, each weighted by 20
fourth-order gammatone power responses |H(f)|² = (1+((f−fc)/1.019·ERB(fc))²)⁻⁴, with fc ERB-spaced over
50–8000 Hz and weights normalised to sum 1 → E (60 = 3×20 channels).
- log E: 60
- log E − LP_τ(log E), τ ∈ {1.5, 4.5, 15, 60, 150}: 300
- tonic LP_60 and LP_150 of log E and of E^(1/3): 240
- acausal LP_60 of log E: 60
- past differences x[t]−x[t−k] and future differences x[t+k]−x[t], k ∈ {3, 6, 12}: 360
- rectified (positive) halves of those differences: 360. The negative half is left out because
  it equals [d]₊ − d, so it would be an exact linear duplicate.

## 2. 2-D Gabor STM on the 20-ch 4096 cochleagram (680)
Separable complex Gabors applied to the log cochleagram (channel means removed). Temporal σ = 0.75/rate s
(0.2 s for rate 0), spectral σ = 0.75/|scale| channels (3 for scale 0). Both are zero-mean whenever the
modulation is non-zero, and the edges are reflect-padded. 17 (rate Hz, scale cyc/ch) pairs:
(0,.25), (1,0), (1,±.25), (3,0), (3,±.25), (3,±.5), (6,0), (6,±.25), (6,±.5), (12,0), (12,±.25).
- quadrature energy |z| per channel: 17×20 = 340
- acausal residual |z| − acausal LP_15(|z|): 340

## 3. STM on the 40-ch cochleagram (705)
40 gammatone channels (ERB 50–8000 Hz) from the 4096 spectrum, log. Rates {0.5, 1, 2, 4, 8, 12} Hz ×
scales {0, ±0.25, ±0.5} cyc/ch = 30 filters, each averaged over the low, mid and high frequency thirds.
- cosine (Re) and sine (Im): 30×3×2 = 180
- magnitude: 90
- dyn of the magnitudes: 270
- acausal residual of the magnitudes (τ = 15): 90
- cross-rate energy products √(M_r·M_r+1) for adjacent rates: 5×5×3 = 75

## 4. Descriptors (456)
Taken from the 4096 Hann power spectrum unless stated otherwise. There are 47 scalars:
- log loudness (50 Hz–11 kHz)
- pitch: an 8-harmonic comb on the log spectrum. Each harmonic's peak-picked level (±2 bins) minus the
  inter-harmonic level, harmonic weights 0.85^(h−1). This gives salience at 11 F0s log-spaced over
  75–450 Hz (11), plus the best F0 (log2) and its strength from a 61-point grid (2)
- ACF peak in the 98–588 sample lag range (normalised by the window ACF) → HNR in dB and the raw peak (2)
- cepstral peak prominence (peak minus mean over quefrency 1/450–1/60 s) (1)
- centroid (log), spread (log), skewness, kurtosis (log), flatness (log GM/AM) and 85 % rolloff (log) (6)
- octave spectral contrast, top 20 % minus bottom 20 % of log bins, 7 octaves from 100 Hz to 12.8 kHz (7)
- channel-group contrast on the 40-ch cochleagram: max − mean in 5 groups of 8 channels (5)
- softmax formants on the cepstrally liftered log spectrum (40 coefficients) in the ranges
  200–1000, 700–2600 and 1800–3600 Hz: log frequency and level (6)
- spectral flux: positive log-cochleagram flux and RMS positive flux of the log spectrum (2)
- phonetic band contrasts (fricative−low, F2−F1, high−mid, low−all) (4)

Voicing-gated cochleagram: clip((ACF peak − 0.3)/0.5, 0, 1) × 20-ch 4096 log cochleagram (floor 0) (20).

Base = 67. Output = base 67 + dyn 201 + temporal-Gabor envelope magnitudes of the 47 scalars
(segment-z-scored) at 2/4/6/10 Hz (σ = 1/(πf) s, zero-mean) 188 = 456.

## 5. Interactions (168)
- 8 key scalars: loudness, flux, pitch strength, log F0, HNR, centroid, F1, F2
- their dyn: 24
- products of the 4 phonetic band contrasts × {pitch strength, log F0, loudness, flux, F1, F2}: 24

That is 56 columns. They are z-scored within the segment, clipped to ±6, and expanded to
[z]₊, [−z]₊ and z³/10, giving 168.

## 6. Within-window scattering (288)
On the full 4096-sample window, 9 analytic Gaussian Morlet filters (Q = 4, fc log-spaced over
150–7000 Hz). First order: S1 = Hann-weighted mean of |u1|. Second order: Q = 1 Morlets at
20/40/80/160/320 Hz applied to |u1|, with S2 = Hann-weighted mean of |u2|.
- log S1: 9
- log(S2/S1): 45
- instantaneous-frequency deviation log2(IF/fc), with IF taken from the weighted lag-1 phase product: 9
- phase coherence |Σ w·u[n+1]u*[n]| / Σ w·|u[n+1]||u[n]|: 9

That is 72, and with its dyn (216) the group totals 288.

## 7. 40-ch log cochleagram (160)
log of the 40-ch cochleagram (40) and its dyn (120).
