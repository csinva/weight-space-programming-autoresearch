# LEAN-v20 feature catalog

Source: `LEAN-v20.py` (this folder), class `WaveEmbedder`, scored config `interact_set="v3e"`, `cond_mode="none"`.
Written Nov 5 2026 from a full read of the code. The block sizes below were counted from the code and
add up to exactly the size reported by the scored runs (3,852 raw → 15,408 after 4 FIR delays,
`runs-wave/lean-whisper/diag_v20_23989056.out` and `runs-wave/exceed-v20-*/results/overall_results.csv`).

**Score (canonical fixed 23-chunk test, 500 train, per-electrode metric):** all-166 R = 0.0808, grid-125 = 0.0904
(Whisper 4-lag ridge: 0.0817 / 0.0933). No weights are trained; ridge regression is the only step fit to data.

---

## 1. How the feature matrix is laid out

* **Input:** one 4096-sample audio window per 30 Hz frame (SR 44.1 kHz → 92.9 ms window, 33.3 ms hop), centered on
  the mel-frame center (`evolve-wave/src/data.py:frame_windows`). The short (1024 = 23 ms) and medium (2048 = 46 ms)
  windows are center crops of the same 4096 window.
* **Raw vector:** 3,852 columns per frame, in the order of the table in §3 (51 named groups).
* **Delays:** the harness stacks 4 copies at delays 0–3 frames, **delay-major**: column for raw index `j` at delay `d`
  is `d*3852 + j`. Features at frame t (and t−1..t−3) predict high-gamma at t+3 (100 ms lag).
* **Normalization:** every delayed column is z-scored **within each chunk**, so only within-chunk variation reaches
  the ridge. Slow "tonic" features therefore encode level relative to that chunk's mean, not absolute level.
* **Ridge:** bootstrapped ridge, alphas `logspace(1, 4, 12)`.

### Notation used below

| Symbol | Meaning |
|---|---|
| `EMA_τ(x)` | causal leaky integrator, α = 1/(1+τ), τ in frames |
| `aEMA_τ(x)` | the same, run backwards in time (looks ahead) |
| `dev_τ(x)` | `x − EMA_τ(x)` ("adapt-delta": deviation from recent past) |
| τ = 1.5 / 4.5 / 15 / 30 / 60 / 150 frames | 50 ms / 150 ms (phonetic) / 0.5 s (syllabic) / 1 s / 2 s (word–phrase) / 5 s (prosodic state) |
| `Δx` | 1-frame difference `x[t] − x[t−1]` |
| `asym(x)` | per-chunk z-score `z` → `[ReLU(z), ReLU(−z), z³]` (3 columns) |
| Time support | **W** = within the 93 ms window only · **C** = causal (past only) · **A±n** = centered/acausal, reaches ±n frames · **F+n** = uses audio n frames ahead |

---

## 2. Shared front-end signals

| Signal | Definition |
|---|---|
| `log_e_{short,med,long}` | log power through a 20-channel ERB-spaced 4th-order gammatone bank (power-domain), 50–8000 Hz, on the 1024 / 2048 / 4096 window. Channel centers (Hz): 50, 104, 169, 247, 340, 451, 583, 742, 931, 1157, 1427, 1750, 2136, 2597, 3148, 3806, 4593, 5534, 6657, 8000 |
| `cube_*` | the same filterbank power, cube-root compressed instead of log |
| `log_e_fine_long` | 40-channel ERB gammatone, 4096 window. Split into 3 bins for the modulation banks: low 49–561 Hz (ch 0–12), mid 632–2212 Hz (ch 13–25), high 2433–8000 Hz (ch 26–39) |
| `power_long` | linear power spectrum of the Hann-windowed 4096 window (2049 bins, 0–22.05 kHz, 10.8 Hz spacing) |
| `best_strength`, `log_f0` | max and argmax over 11 F0 candidates of the harmonic-comb salience (below) |
| `long_loud` | mean over 20 channels of `log_e_long` |
| `long_centroid` | soft-argmax channel index: Σ softmax(`log_e_long`)·channel |
| `flat` | spectral flatness of `power_long`, log(geometric/arithmetic mean); always ≤ 0 |
| `hnr_peak`, `hnr_dB` | autocorrelation peak (lags for 75–400 Hz), clamped to [0, 0.999], and 10·log10(p/(1−p)) |
| `f1_loc`, `f2_loc`, `f1_amp`, `f2_amp` | "formants": softmax(2·`log_e_long`)-weighted channel position and level over ch 0–9 (50–1157 Hz, F1 proxy) and ch 10–19 (1427–8000 Hz, F2 proxy) |
| `phon` (5) | phonetic templates on `log_e_long` (see `phonetic`) |
| `flux_abs_long` | Σ_ch \|Δ`log_e_long`\| |
| `comod` (6) | products of mean log-energy in 6 overlapping channel bands ([0–3],[3–6],[6–9],[9–12],[12–15],[15–19]) for pairs (0,1),(1,2),(2,3),(3,4),(0,5),(2,5). Not output as its own group; used only inside `modulation` |

---

## 3. Catalog (in column order)

Cols = raw column range (0-based, before delays). Total 3,852.

### A. Cochleagram core — 360 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `cochleagram` | 0–59 | 60 | log gammatone energy, 20 ch × 3 windows | [1024 ch0–19, 2048 ch0–19, 4096 ch0–19] | W |
| `adapt_delta` | 60–359 | 300 | `dev_τ(log_e)` per channel: onset/adaptation at 5 time scales | window-major → τ (1.5, 4.5, 15, 60, 150) → 20 ch | C |

### B. Pitch and voicing, part 1 — 19 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `pitch_salience` | 360–370 | 11 | harmonic comb: mean log-power at the first 8 harmonic bins of each F0 minus mean log-power of the whole spectrum | F0 = 75, 95, 115, 140, 170, 200, 230, 270, 320, 380, 450 Hz | W |
| `pitch_summary` | 371–372 | 2 | `best_strength`, `log_f0` | | W |
| `pitch_dyn` | 373–378 | 6 | `dev_τ` of log F0 and strength | τ (4.5, 15, 60) × [logF0, strength] | C |

### C. Spectral shape — 14 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `spectral_moments` | 379–382 | 4 | on `power_long` (0–22 kHz, linear Hz) | [log centroid, log spread, skew (±10), kurtosis (0–50)] | W |
| `spectral_flatness` | 383 | 1 | `flat` | | W |
| `spectral_rolloff` | 384–385 | 2 | log frequency below which 85% / 95% of energy lies | | W |
| `hnr` | 386–387 | 2 | `hnr_peak`, `hnr_dB` | | W |
| `phonetic` | 388–392 | 5 | contrasts of channel-band means of `log_e_long` | sibilant = mean(ch14–19) − mean(ch0–5); vowel = mean(ch5–14) − std across channels; nasal = mean(ch1–5) − mean(ch8–13); plosive = std − mean; fricative = mean(ch17–19) − mean(ch5–14) | W |

### D. Spectral flux — 36 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `flux` | 393–428 | 36 | broadband Σ_ch of positive / negative / absolute Δ`log_e`, per window, plus deviations | [raw 9, dev_15 9, dev_60 9, dev_150 9]; each 9 = window (1024, 2048, 4096) × [pos, neg, abs] | C |

### E. Temporal modulation of 1-D summary signals — 132 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `modulation` | 429–560 | 132 | magnitude of a centered Gabor band-pass (σ = 1 period, ±4σ, unit-norm) applied along time to 33 summary signals | rate (2, 4, 6, 10 Hz) × 33 signals: loud 0, centroid 1, hnr 2–3, strength 4, logF0 5, flux 6–14, phon 15–19, moments 20–23, flat 24, rolloff 25–26, comod 27–32 | A±60 at 2 Hz … A±12 at 10 Hz |

### F. Spectro-temporal modulation (Chi–Ru–Shamma style) — 357 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `stmf` | 561–900 | 340 | per-channel magnitude of a 2-D Gabor (rate Hz × scale cyc/ch) on 20-ch `log_e_long`; sign of scale = sweep direction | 17 (rate, scale) pairs × 20 ch, pair order: (2,±.25), (4,±.25), (4,±.5), (8,±.25), (8,±.5), (12,±.25), (4,0), (8,0), (12,0), (0,.25), (0,.5) | A±2 (12 Hz) … A±8 (2 Hz); A±29 for rate 0 |
| `stmf_mean_gabor` | 901–917 | 17 | channel-mean of each STMF map, then its 2 Hz Gabor envelope | 17 pairs, same order | A, ~±2 s |

### G. Formants — 33 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `formants` | 918–941 | 24 | [f1_loc, f2_loc, f2−f1 loc, f1_amp, f2_amp, f2_amp−f1_amp] and their deviations | [raw 6, dev_4.5 6, dev_15 6, dev_60 6] | C |
| `formant_dyn` | 942–950 | 9 | Δ of (f1_loc, f2_loc, f2−f1), plus half-wave rectified | [Δ 3, ReLU(Δ) 3, ReLU(−Δ) 3] | C |

### H. Slow envelope, context and lookahead — 600 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `mod_depth` | 951–990 | 40 | per-channel modulation depth: `EMA_15(|log_e_long − EMA_hp(log_e_long)|)` | hp τ = 4.5 (20 ch), then hp τ = 1.5 (20 ch) | C |
| `tonic` | 991–1290 | 300 | slow smoothed cochleagrams (state at word/prosodic scale) | per window (1024, 2048, 4096): [EMA_60 log, EMA_60 cube, EMA_150 log, EMA_150 cube, aEMA_60 log] × 20 ch | C, plus one A (aEMA) |
| `future_past` | 1291–1550 | 260 | per-channel `log_e_long` change to the future and from the past | for k = 3, 6, 12: [log_e[t+k]−log_e[t], ReLU(+), ReLU(−), log_e[t]−log_e[t−k]] × 20 ch; then cube-root future Δ at k = 6 (20) | F+3 / F+6 / F+12 |

### I. Pitch and voicing dynamics — 7 dims (v16)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `f0_dyn` | 1551–1554 | 4 | log F0 slope and curvature | [Δ, Δ², ReLU(Δ), ReLU(−Δ)] | C |
| `hnr_dyn` | 1555–1557 | 3 | HNR (dB) change | [Δ, ReLU(Δ), ReLU(−Δ)] | C |

*(v16 ends at col 1557: 1,558 raw dims, R = 0.0760 all-166.)*

### J. Interpretable nonlinear interactions `interact_v3e` — 405 dims (v17)

Named products and half-wave/cubic nonlinearities of existing signals, all z-scored within the chunk. Key signals
(10): long_loud, best_strength, log_f0, flux_abs_long, hnr_peak, long_centroid, flat, hnr_dB, f2_loc−f1_loc, f1_amp+f2_amp.

| Sub-block | Cols | Dim | What it is |
|---|---|---|---|
| raw asymmetry | 1558–1587 | 30 | `asym(sig)` for the 10 key signals |
| key-signal deviation asymmetry | 1588–1677 | 90 | `asym(dev_τ(sig))`, 10 signals × τ (4.5, 15, 60) × 3 |
| phonetic deviation asymmetry | 1678–1722 | 45 | `asym(dev_τ(phon_j))`, 5 × 3 τ × 3 |
| formant deviation asymmetry | 1723–1758 | 36 | `asym(dev_τ(·))` for f1_loc, f2_loc, f1_amp, f2_amp × 3 τ × 3 |
| cross-lag products | 1759–1764 | 6 | flux[t−2]·ΔHNR[t]; ΔHNR[t−2]·flux[t]; flux[t−2]·loud[t]; strength[t]·ΔlogF0[t+2]; loud[t]·flux[t+2]; voicing[t]·(1−voicing)[t+2] (voiced→unvoiced transition). Uses F+2 |
| phonetic × key signal | 1765–1824 | 60 | `asym(phon_j · s)`, s ∈ {strength, loud, flux_abs, ΔHNR}; 5 × 4 × 3 |
| phonetic × formant | 1825–1884 | 60 | `asym(phon_j · f)`, f ∈ {f1_loc, f2_loc, f1_amp, f2_amp}; 5 × 4 × 3 |
| key signal × formant | 1885–1932 | 48 | `asym(s · f)`, 4 × 4 × 3 |
| onset/offset-gated phonetics | 1933–1962 | 30 | gate = ReLU(±z(dev_15(flux_abs))); `asym(phon_j · gate)`, 5 × 2 × 3 |

### K. Voice quality — 12 dims (v18, from exceed-frontend)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `cpp` | 1963–1966 | 4 | cepstral peak prominence: real-cepstrum peak in 60–400 Hz quefrency range minus a linear-trend baseline | [CPP, dev_4.5, dev_15, dev_60] | W / C |
| `region_flat` | 1967–1974 | 8 | flatness of `power_long` restricted to 4–8 kHz and to 60–500 Hz | [hi, lo] then dev_4.5, dev_15, dev_60 | W / C |

### L. Spectral and cochleagram contrast — 192 dims (v18, from exceed-frontend)

All "contrast" = mean of the top-k minus mean of the bottom-k log values within a band (peak-vs-valley sharpness).
Cochleagram groups are 5 blocks of 4 adjacent channels: 49–246, 339–741, 930–1749, 2135–3806, 4593–8000 Hz.

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `spectral_contrast` | 1975–1998 | 24 | FFT-bin contrast (top/bottom 25%) on `power_long` in 6 octaves 62–4000 Hz | [6 octaves, dev_4.5 ×6, dev_15 ×6, dev_60 ×6] | W / C |
| `contrast_short` | 1999–2022 | 24 | same on the 1024 window (top/bottom third) | same | W / C |
| `contrast_med` | 2023–2046 | 24 | same on the 2048 window (top/bottom 25%) | same | W / C |
| `cochlea_contrast` | 2047–2066 | 20 | top-2 − bottom-2 channels in each 4-channel group, `log_e_long` | [5 groups, dev_4.5, dev_15, dev_60] | W / C |
| `med_cochlea_contrast` | 2067–2086 | 20 | same on `log_e_med` | same | W / C |
| `short_cochlea_contrast` | 2087–2106 | 20 | same on `log_e_short` | same | W / C |
| `multitau_contrast` | 2107–2116 | 10 | group contrast of `dev_τ(log_e_long)` | τ = 1.5 (5 groups), τ = 15 (5) | C |
| `med_adapt_contrast` | 2117–2121 | 5 | group contrast of `dev_4.5(log_e_med)` | 5 groups | C |
| `topbot` | 2122–2131 | 10 | top-2 and bottom-2 means separately, `log_e_long` | per group [top, bottom] | W |
| `topbot_short` | 2132–2141 | 10 | same on `log_e_short` | per group [top, bottom] | W |
| `fut_cochlea_contrast` | 2142–2146 | 5 | group contrast of `log_e_long` 3 frames ahead | 5 groups | F+3 |
| `fut_cochlea_contrast_med` | 2147–2151 | 5 | same, `log_e_med`, 3 ahead | 5 groups | F+3 |
| `fut_med_contrast_6` | 2152–2156 | 5 | same, `log_e_med`, 6 ahead | 5 groups | F+6 |
| `fut_minus_cur_contrast` | 2157–2161 | 5 | `fut_med_contrast_6` − current long contrast | 5 groups | F+6 |
| `cube_fut_contrast` | 2162–2166 | 5 | group contrast of cube-root long cochleagram, 3 ahead | 5 groups | F+3 |

### M. Quadrature spectro-temporal modulation bank — 954 dims (v18.q1–q17, exceed-v18)

2-D Gabor (same kernel as `stmf`) on the 40-channel `log_e_fine_long`, keeping **phase**: the even (cos) and odd (sin)
responses are kept separately alongside the magnitude, each averaged over the 3 frequency bins. Index within the
180-column bank: `((rate_i*5 + scale_i)*3 + bin)*3 + phase`, scales (0, +.25, −.25, +.5, −.5) cyc/ch, bins
(low, mid, high), phase (cos, sin, mag; mag = mean of per-channel magnitudes).

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `quad_modbank` | 2167–2346 | 180 | rates 2, 4, 8, 12 Hz × 5 scales × 3 bins × 3 phases | as above | A±2 … A±8 |
| `quad_modbank_dyn` | 2347–2886 | 540 | `dev_τ(quad_modbank)` | τ (4.5, 15, 60) × 180 | C (on A input) |
| `quad_modbank_acausal` | 2887–3066 | 180 | `quad_modbank − aEMA_60(quad_modbank)`: deviation from the next ~2 s | 180 | A, ~2 s ahead |
| `quad_xrate_energy` | 3067–3084 | 18 | per bin, energy = Σ over scales of magnitude; log1p of products across rate pairs (slow × fast coincidence) | pairs (2×8, 2×12, 4×12, 4×8, 8×12, 2×4 Hz) × 3 bins | A |
| `quad_xrate_energy_dyn` | 3085–3120 | 36 | `dev_τ` of the above | τ (4.5, 15) × 18 | C |

### N. Voicing gates — 84 dims (v19: q19, q29)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `cochlea_voiced_gated` | 3121–3160 | 40 | `log_e_long` × g and × (1−g), g = sigmoid(8·(best_strength − 0.3)) | [voiced 20 ch, unvoiced 20 ch] | W |
| `cochlea_voiced_gated_dyn` | 3161–3180 | 20 | `dev_15` of the voiced-gated cochleagram | 20 ch | C |
| `gate_signals` | 3181–3186 | 6 | soft thresholds: sigmoid(8·(strength − θ)), θ = 0.15, 0.3, 0.6; sigmoid(10·(flat − θ)), θ = 0.2, 0.4, 0.6 | as listed | W |
| `gate_signals_dyn` | 3187–3204 | 18 | `dev_τ(gate_signals)` | τ (4.5, 15, 60) × 6 | C |

### O. Slow quadrature modulation bank — 360 dims (v20: q42–q45)

| Group | Cols | Dim | What it is | Layout | Time |
|---|---|---|---|---|---|
| `slow_quad_modbank` | 3205–3384 | 180 | same as `quad_modbank` at prosodic rates 0.5, 1, 1.5, 2 Hz | same index formula | A±29 (0.5 Hz) … A±8 (2 Hz) |
| `slow_quad_modbank_dyn` | 3385–3564 | 180 | `dev_30(slow_quad_modbank)` | 180 | C (on A input) |

### P. Wavelet scattering — 287 dims (v18, ported from SCAT-v103)

Computed on the raw 4096-sample waveform. First-order Morlet wavelets (Q = 4/oct) at 9 paths
j1 = 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800 Hz (σ = fc/4). Analytic envelope `u_j1` via Hilbert transform.
Second-order wavelets (Q = 1/oct) at 0.5, 1, 2, … 4096 Hz, kept only where fc2 < fc1/2 (6 → 14 paths per j1, 90 total).

| Sub-block | Cols | Dim | What it is |
|---|---|---|---|
| `S2m_log` | 3565–3654 | 90 | log1p(mean \|u_j1 ∗ ψ_j2\| / mean u_j1), j1-major |
| `S2s_log` | 3655–3744 | 90 | log1p(std \|u_j1 ∗ ψ_j2\| / mean u_j1) |
| `S2peak_log` | 3745–3753 | 9 | cube-root of max over (j2, samples) of \|u_j1 ∗ ψ_j2\| (not normalized) |
| `joint_log` | 3754–3761 | 8 | log1p(\|S1[j1+1] − S1[j1]\| / pair mean): across-frequency envelope slope |
| `contour` | 3762–3770 | 9 | log(mean envelope in last quarter / first quarter of window): rising vs falling within 93 ms |
| `s1_sub_log` | 3771–3779 | 9 | log mean envelope per j1 |
| `s1_sub_dyn` | 3780–3806 | 27 | `dev_τ(s1_sub_log)`, τ (15, 60, 150) × 9 |
| `s1_peak_log` | 3807–3815 | 9 | log max envelope within the window |
| `if_feat` | 3816–3824 | 9 | mean instantaneous frequency / fc (clamped ±3) |
| `coh_feat` | 3825–3833 | 9 | atanh of phase coherence \|⟨y[n+1]y*[n]⟩\| / ⟨\|y\|²⟩ (tonal ≈ 1, noise ≈ 0) |
| `if_jitter` | 3834–3842 | 9 | std of per-sample instantaneous frequency / fc |
| `s2_entropy` | 3843–3851 | 9 | entropy of the normalized S2 distribution per j1 (flat vs peaked modulation spectrum) |

---

## 4. Summary by family

| Family | Dim | Share | Added in |
|---|---|---|---|
| Cochleagram core + slow context + lookahead (A, H) | 960 | 25% | v16 |
| Modulation (E, F, M, O) | 1,803 | 47% | v16 (E, F), v18.q (M), v20 (O) |
| Pitch / voicing / voice quality (B, I, K, N) | 122 | 3% | v16, v18, v19 |
| Spectral shape + formants + flux (C, D, G) | 83 | 2% | v16 |
| Contrast (L) | 192 | 5% | v18 |
| Interactions (J) | 405 | 10.5% | v17 |
| Scattering (P) | 287 | 7.5% | v18 |
| **Total** | **3,852** | | |

Modulation-type features are nearly half the vector, and most of them are the quadrature banks M and O added in
the final two days.

### Lineage (grid-125 R on the fixed test set unless noted)

| Version | What was added | Score |
|---|---|---|
| v16 | groups A–I (1,558 dims) | 0.0760 all-166 |
| v17 | `interact_v3e` | 0.0778 all-166 / 0.0866 grid |
| v18 | CPP, region flatness, contrast family (v32→v107: +0.0011 grid over 17 small steps), scattering (COMBINE-v5) | 0.0800 all-166 / 0.0891 grid |
| v18.q1–q17 | quadrature bank, its deviations, acausal deviation, cross-rate energy | 0.0902 grid |
| v19 | voiced/unvoiced-gated cochleagram (q19), gate signals (q29) | 0.0805 all-166 / 0.0903 grid |
| v20 | slow quadrature bank 0.5–2 Hz (q42–q45) | 0.0808 all-166 / 0.0904 grid |

v20 is q45 (5 scales), which scored 0.09043 vs q44's 0.09047; the difference is far inside the ±0.0008 noise floor.

---

## 5. Quirks found while cataloging (all checked against the code)

1. **`build_embedder()` defaults to `interact_set="none"`** (3,447 raw dims). The scored v20 used `"v3e"` (3,852),
   set by the `__main__` argparse default and by `diag_v20.py`. Anything that imports `build_embedder()` without
   passing `interact_set="v3e"` gets a different model from the one that was scored.
2. **Scattering second-order paths at 0.5, 1 and 2 Hz are effectively empty.** A 4096-sample window at 44.1 kHz has
   10.8 Hz FFT bins. The largest weight these wavelets put on any non-DC bin is 1e-23, 2e-21 and 7e-5. That covers
   27 `S2m` + 27 `S2s` columns. Their output is numerical noise, which the per-chunk z-score likely rescales to unit
   variance. The "4 Hz" path actually responds mostly at ~11 Hz. Within one 93 ms window, nothing below ~11 Hz can be
   resolved.
3. **`contrast_short` octave 62–125 Hz is always zero.** The 1024 window has only 1 bin in that octave and the code
   writes zeros when there are fewer than 2 (cols 1999, 2005, 2011, 2017).
4. **The flatness "gates" never open.** Flatness is log(geometric/arithmetic mean) ≤ 0, so sigmoid(10·(flat − θ)) with
   θ = 0.2–0.6 is always < 0.12. They work as exponentially compressed flatness, not as thresholds. The code comment
   also assumes `best_strength` ∈ [0, 1], which the definition (a log-power difference) does not guarantee. Its actual
   range on data was not checked here.
5. **Computed but never output:** the dense 33-band S1 (`S1_all_log`), `peak_pos` and `s1_centroid` in scattering.
   Defined but never used: `fb_sub`, `fb_fine_med`, `dct_basis`, `_dog_channels`, `STMF_RATES_ADAPT/FAST/MID`, and
   the `cond_mode` replication machinery when `cond_mode="none"`.
6. **Comments that disagree with the code:** the `phonetic_templates` docstring channel ranges and "harmonicity bonus"
   (the code uses the ranges listed in §3 and no harmonicity term), and the `(T, 108)` comment on `slow_quad_modbank`
   (it is 180).
7. **Lookahead:** several groups use audio after frame t (`future_past` up to +12 frames = 400 ms; the `fut_*`
   contrasts; `quad_modbank_acausal` ~2 s; the centered Gabor/STMF kernels up to ±2 s; `tonic`'s aEMA_60; 3 of
   the v3e cross-lag terms). Because the target is ECoG at t+3, any lookahead beyond 3 frames uses sound that comes
   after the neural sample being predicted. This is legitimate for an acausal encoding model, but it matters when
   interpreting latency.
