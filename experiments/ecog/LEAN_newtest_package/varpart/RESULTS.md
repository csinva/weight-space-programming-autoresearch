# LEAN-v20 variance partitioning — new held-out test set

New test set: 317 chunks of the held-out session 450a36e4 (runs-wave/newtest/PROTOCOL.md; never used during the search). Same 500 training chunks and fits as varpart-v20, harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the canonical per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model (validation)

- grid: mean r = 0.0735 (old test 0.0904), R² = 1.476% of variance
- all166: mean r = 0.0646 (old test 0.0808), R² = 1.180% of variance

## Family level — grid (full R² = 1.476%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 1.384 | 94% | +0.1000 [+0.0893, +0.1109] | +6.8% |
| Interpretable interactions (v3e) | 405 | 1.212 | 82% | +0.0289 [+0.0254, +0.0325] | +2.0% |
| Wavelet scattering | 287 | 1.109 | 75% | +0.0168 [+0.0130, +0.0210] | +1.1% |
| Spectral/cochleagram contrast | 192 | 1.037 | 70% | +0.0105 [+0.0074, +0.0136] | +0.7% |
| Cochleagram core + slow context + lookahead | 960 | 1.136 | 77% | +0.0074 [-0.0019, +0.0168] | +0.5% |
| Pitch, voicing, voice quality | 122 | 0.792 | 54% | +0.0061 [+0.0049, +0.0073] | +0.4% |
| Spectral shape, flux, formants | 83 | 1.012 | 69% | +0.0061 [+0.0047, +0.0075] | +0.4% |

Sum of unique shares: 11.9% → shared (not attributable to any one unit): 88.1%

## Family level — all166 (full R² = 1.180%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 1.103 | 94% | +0.0796 [+0.0707, +0.0890] | +6.8% |
| Interpretable interactions (v3e) | 405 | 0.966 | 82% | +0.0266 [+0.0235, +0.0297] | +2.3% |
| Wavelet scattering | 287 | 0.886 | 75% | +0.0127 [+0.0095, +0.0160] | +1.1% |
| Cochleagram core + slow context + lookahead | 960 | 0.914 | 78% | +0.0100 [+0.0024, +0.0178] | +0.8% |
| Spectral/cochleagram contrast | 192 | 0.830 | 70% | +0.0097 [+0.0070, +0.0123] | +0.8% |
| Pitch, voicing, voice quality | 122 | 0.639 | 54% | +0.0081 [+0.0069, +0.0093] | +0.7% |
| Spectral shape, flux, formants | 83 | 0.810 | 69% | +0.0046 [+0.0035, +0.0057] | +0.4% |

Sum of unique shares: 12.8% → shared (not attributable to any one unit): 87.2%

## Section level — grid (full R² = 1.476%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| J. Interactions (v3e) | 405 | 1.212 | 82% | +0.0289 [+0.0254, +0.0325] | +2.0% |
| M. Quadrature modulation bank | 954 | 1.290 | 87% | +0.0251 [+0.0211, +0.0293] | +1.7% |
| E. 1-D temporal modulation | 132 | 0.845 | 57% | +0.0180 [+0.0129, +0.0231] | +1.2% |
| P. Wavelet scattering | 287 | 1.109 | 75% | +0.0168 [+0.0130, +0.0210] | +1.1% |
| F. Spectro-temporal modulation (STMF) | 357 | 1.108 | 75% | +0.0117 [+0.0086, +0.0148] | +0.8% |
| L. Spectral/cochleagram contrast | 192 | 1.037 | 70% | +0.0105 [+0.0074, +0.0136] | +0.7% |
| H. Slow context + lookahead | 600 | 1.109 | 75% | +0.0087 [+0.0011, +0.0167] | +0.6% |
| D. Spectral flux | 36 | 0.721 | 49% | +0.0061 [+0.0049, +0.0072] | +0.4% |
| O. Slow quadrature bank | 360 | 1.059 | 72% | +0.0046 [+0.0031, +0.0061] | +0.3% |
| B. Pitch | 19 | 0.417 | 28% | +0.0034 [+0.0026, +0.0042] | +0.2% |
| K. Voice quality (CPP, region flatness) | 12 | 0.194 | 13% | +0.0015 [+0.0009, +0.0021] | +0.1% |
| N. Voicing gates | 84 | 0.717 | 49% | +0.0015 [+0.0006, +0.0024] | +0.1% |
| A. Cochleagram core | 360 | 0.812 | 55% | +0.0009 [-0.0006, +0.0024] | +0.1% |
| G. Formants | 33 | 0.733 | 50% | +0.0003 [-0.0001, +0.0007] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.057 | 4% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| C. Spectral shape | 14 | 0.720 | 49% | -0.0001 [-0.0007, +0.0005] | -0.0% |

Sum of unique shares: 9.4% → shared (not attributable to any one unit): 90.6%

## Section level — all166 (full R² = 1.180%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| J. Interactions (v3e) | 405 | 0.966 | 82% | +0.0266 [+0.0235, +0.0297] | +2.3% |
| M. Quadrature modulation bank | 954 | 1.031 | 87% | +0.0197 [+0.0164, +0.0229] | +1.7% |
| E. 1-D temporal modulation | 132 | 0.669 | 57% | +0.0185 [+0.0139, +0.0231] | +1.6% |
| P. Wavelet scattering | 287 | 0.886 | 75% | +0.0127 [+0.0095, +0.0160] | +1.1% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.885 | 75% | +0.0114 [+0.0088, +0.0141] | +1.0% |
| L. Spectral/cochleagram contrast | 192 | 0.830 | 70% | +0.0097 [+0.0070, +0.0123] | +0.8% |
| H. Slow context + lookahead | 600 | 0.895 | 76% | +0.0085 [+0.0020, +0.0153] | +0.7% |
| O. Slow quadrature bank | 360 | 0.855 | 72% | +0.0061 [+0.0048, +0.0075] | +0.5% |
| B. Pitch | 19 | 0.328 | 28% | +0.0054 [+0.0046, +0.0062] | +0.5% |
| D. Spectral flux | 36 | 0.580 | 49% | +0.0042 [+0.0033, +0.0051] | +0.4% |
| A. Cochleagram core | 360 | 0.654 | 55% | +0.0030 [+0.0015, +0.0045] | +0.3% |
| N. Voicing gates | 84 | 0.581 | 49% | +0.0022 [+0.0015, +0.0030] | +0.2% |
| K. Voice quality (CPP, region flatness) | 12 | 0.158 | 13% | +0.0012 [+0.0007, +0.0017] | +0.1% |
| G. Formants | 33 | 0.589 | 50% | +0.0003 [-0.0001, +0.0007] | +0.0% |
| C. Spectral shape | 14 | 0.577 | 49% | +0.0000 [-0.0005, +0.0005] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.044 | 4% | -0.0001 [-0.0003, +0.0000] | -0.0% |

Sum of unique shares: 11.0% → shared (not attributable to any one unit): 89.0%

## Group level — grid (full R² = 1.476%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| interact_v3e | 405 | 1.212 | 82% | +0.0289 [+0.0254, +0.0325] | +2.0% |
| modulation | 132 | 0.845 | 57% | +0.0180 [+0.0129, +0.0231] | +1.2% |
| scattering | 287 | 1.109 | 75% | +0.0168 [+0.0130, +0.0210] | +1.1% |
| stmf | 340 | 1.099 | 74% | +0.0110 [+0.0080, +0.0141] | +0.7% |
| future_past | 260 | 0.855 | 58% | +0.0086 [+0.0068, +0.0106] | +0.6% |
| mod_depth | 40 | 0.782 | 53% | +0.0085 [+0.0065, +0.0105] | +0.6% |
| quad_modbank_dyn | 540 | 0.990 | 67% | +0.0072 [+0.0057, +0.0087] | +0.5% |
| flux | 36 | 0.721 | 49% | +0.0061 [+0.0049, +0.0072] | +0.4% |
| quad_xrate_energy | 18 | 0.843 | 57% | +0.0048 [+0.0036, +0.0060] | +0.3% |
| slow_quad_modbank | 180 | 1.012 | 69% | +0.0045 [+0.0035, +0.0055] | +0.3% |
| quad_modbank | 180 | 1.051 | 71% | +0.0035 [+0.0025, +0.0044] | +0.2% |
| quad_modbank_acausal | 180 | 0.815 | 55% | +0.0034 [+0.0017, +0.0052] | +0.2% |
| tonic | 300 | 0.788 | 53% | +0.0031 [-0.0042, +0.0115] | +0.2% |
| spectral_contrast | 24 | 0.562 | 38% | +0.0031 [+0.0022, +0.0041] | +0.2% |
| quad_xrate_energy_dyn | 36 | 0.574 | 39% | +0.0028 [+0.0021, +0.0036] | +0.2% |
| pitch_salience | 11 | 0.359 | 24% | +0.0028 [+0.0020, +0.0035] | +0.2% |
| stmf_mean_gabor | 17 | 0.080 | 5% | +0.0025 [+0.0016, +0.0034] | +0.2% |
| contrast_med | 24 | 0.494 | 33% | +0.0022 [+0.0015, +0.0029] | +0.1% |
| region_flat | 8 | 0.099 | 7% | +0.0019 [+0.0014, +0.0024] | +0.1% |
| adapt_delta | 300 | 0.770 | 52% | +0.0018 [+0.0003, +0.0032] | +0.1% |
| gate_signals | 6 | 0.001 | 0% | +0.0016 [+0.0008, +0.0024] | +0.1% |
| formant_dyn | 9 | 0.217 | 15% | +0.0016 [+0.0012, +0.0019] | +0.1% |
| phonetic | 5 | 0.599 | 41% | +0.0014 [+0.0009, +0.0018] | +0.1% |
| cochlea_voiced_gated | 40 | 0.677 | 46% | +0.0013 [+0.0010, +0.0016] | +0.1% |
| gate_signals_dyn | 18 | 0.000 | 0% | +0.0011 [+0.0005, +0.0019] | +0.1% |
| short_cochlea_contrast | 20 | 0.390 | 26% | +0.0010 [+0.0006, +0.0014] | +0.1% |
| fut_cochlea_contrast | 5 | 0.427 | 29% | +0.0009 [+0.0006, +0.0012] | +0.1% |
| fut_med_contrast_6 | 5 | 0.185 | 13% | +0.0007 [+0.0004, +0.0011] | +0.0% |
| pitch_dyn | 6 | 0.256 | 17% | +0.0005 [+0.0004, +0.0007] | +0.0% |
| multitau_contrast | 10 | 0.641 | 43% | +0.0005 [-0.0002, +0.0012] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.394 | 27% | +0.0004 [+0.0002, +0.0006] | +0.0% |
| topbot | 10 | 0.720 | 49% | +0.0004 [+0.0002, +0.0007] | +0.0% |
| cochleagram | 60 | 0.706 | 48% | +0.0004 [-0.0002, +0.0010] | +0.0% |
| med_adapt_contrast | 5 | 0.496 | 34% | +0.0004 [+0.0001, +0.0006] | +0.0% |
| cube_fut_contrast | 5 | 0.477 | 32% | +0.0003 [-0.0000, +0.0007] | +0.0% |
| slow_quad_modbank_dyn | 180 | 0.829 | 56% | +0.0003 [-0.0003, +0.0010] | +0.0% |
| formants | 24 | 0.672 | 46% | +0.0003 [-0.0001, +0.0007] | +0.0% |
| spectral_rolloff | 2 | 0.306 | 21% | +0.0002 [-0.0000, +0.0005] | +0.0% |
| cochlea_contrast | 20 | 0.499 | 34% | +0.0002 [-0.0003, +0.0007] | +0.0% |
| fut_minus_cur_contrast | 5 | 0.096 | 7% | +0.0002 [+0.0001, +0.0003] | +0.0% |
| med_cochlea_contrast | 20 | 0.469 | 32% | +0.0002 [-0.0001, +0.0004] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.488 | 33% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| f0_dyn | 4 | 0.040 | 3% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| hnr | 2 | 0.323 | 22% | +0.0001 [-0.0000, +0.0002] | +0.0% |
| pitch_summary | 2 | 0.278 | 19% | +0.0000 [-0.0000, +0.0001] | +0.0% |
| hnr_dyn | 3 | 0.021 | 1% | -0.0000 [-0.0000, +0.0000] | -0.0% |
| spectral_flatness | 1 | 0.302 | 20% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| cpp | 4 | 0.172 | 12% | -0.0001 [-0.0004, +0.0002] | -0.0% |
| topbot_short | 10 | 0.626 | 42% | -0.0001 [-0.0003, +0.0001] | -0.0% |
| spectral_moments | 4 | 0.514 | 35% | -0.0001 [-0.0004, +0.0002] | -0.0% |
| contrast_short | 24 | 0.402 | 27% | -0.0002 [-0.0006, +0.0003] | -0.0% |

Sum of unique shares: 10.5% → shared (not attributable to any one unit): 89.5%

## Group level — all166 (full R² = 1.180%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| interact_v3e | 405 | 0.966 | 82% | +0.0266 [+0.0235, +0.0297] | +2.3% |
| modulation | 132 | 0.669 | 57% | +0.0185 [+0.0139, +0.0231] | +1.6% |
| scattering | 287 | 0.886 | 75% | +0.0127 [+0.0095, +0.0160] | +1.1% |
| mod_depth | 40 | 0.629 | 53% | +0.0118 [+0.0097, +0.0139] | +1.0% |
| stmf | 340 | 0.879 | 74% | +0.0109 [+0.0084, +0.0134] | +0.9% |
| future_past | 260 | 0.685 | 58% | +0.0086 [+0.0070, +0.0105] | +0.7% |
| quad_modbank_dyn | 540 | 0.782 | 66% | +0.0054 [+0.0042, +0.0066] | +0.5% |
| contrast_med | 24 | 0.388 | 33% | +0.0044 [+0.0036, +0.0051] | +0.4% |
| flux | 36 | 0.580 | 49% | +0.0042 [+0.0033, +0.0051] | +0.4% |
| quad_xrate_energy | 18 | 0.672 | 57% | +0.0039 [+0.0030, +0.0049] | +0.3% |
| slow_quad_modbank_dyn | 180 | 0.660 | 56% | +0.0038 [+0.0031, +0.0045] | +0.3% |
| adapt_delta | 300 | 0.614 | 52% | +0.0037 [+0.0023, +0.0051] | +0.3% |
| cochleagram | 60 | 0.566 | 48% | +0.0035 [+0.0028, +0.0043] | +0.3% |
| slow_quad_modbank | 180 | 0.814 | 69% | +0.0034 [+0.0025, +0.0042] | +0.3% |
| quad_modbank | 180 | 0.840 | 71% | +0.0029 [+0.0022, +0.0037] | +0.2% |
| tonic | 300 | 0.631 | 54% | +0.0028 [-0.0033, +0.0097] | +0.2% |
| quad_modbank_acausal | 180 | 0.648 | 55% | +0.0024 [+0.0009, +0.0039] | +0.2% |
| quad_xrate_energy_dyn | 36 | 0.454 | 39% | +0.0020 [+0.0015, +0.0026] | +0.2% |
| region_flat | 8 | 0.082 | 7% | +0.0018 [+0.0014, +0.0022] | +0.2% |
| spectral_contrast | 24 | 0.441 | 37% | +0.0018 [+0.0009, +0.0026] | +0.2% |
| pitch_salience | 11 | 0.281 | 24% | +0.0018 [+0.0011, +0.0024] | +0.1% |
| stmf_mean_gabor | 17 | 0.065 | 6% | +0.0016 [+0.0008, +0.0024] | +0.1% |
| gate_signals | 6 | 0.001 | 0% | +0.0012 [+0.0006, +0.0019] | +0.1% |
| formant_dyn | 9 | 0.171 | 14% | +0.0012 [+0.0009, +0.0015] | +0.1% |
| phonetic | 5 | 0.483 | 41% | +0.0011 [+0.0007, +0.0014] | +0.1% |
| cochlea_voiced_gated | 40 | 0.544 | 46% | +0.0008 [+0.0005, +0.0010] | +0.1% |
| gate_signals_dyn | 18 | 0.001 | 0% | +0.0007 [+0.0002, +0.0013] | +0.1% |
| fut_cochlea_contrast | 5 | 0.341 | 29% | +0.0007 [+0.0004, +0.0009] | +0.1% |
| fut_med_contrast_6 | 5 | 0.152 | 13% | +0.0005 [+0.0003, +0.0008] | +0.0% |
| pitch_dyn | 6 | 0.199 | 17% | +0.0004 [+0.0003, +0.0006] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.315 | 27% | +0.0004 [+0.0002, +0.0005] | +0.0% |
| cube_fut_contrast | 5 | 0.386 | 33% | +0.0004 [+0.0001, +0.0007] | +0.0% |
| formants | 24 | 0.540 | 46% | +0.0003 [-0.0001, +0.0007] | +0.0% |
| short_cochlea_contrast | 20 | 0.310 | 26% | +0.0002 [-0.0002, +0.0007] | +0.0% |
| med_cochlea_contrast | 20 | 0.374 | 32% | +0.0002 [-0.0001, +0.0004] | +0.0% |
| topbot | 10 | 0.576 | 49% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| med_adapt_contrast | 5 | 0.393 | 33% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| multitau_contrast | 10 | 0.508 | 43% | +0.0001 [-0.0005, +0.0007] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.388 | 33% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| cochlea_contrast | 20 | 0.396 | 34% | +0.0000 [-0.0005, +0.0005] | +0.0% |
| pitch_summary | 2 | 0.218 | 18% | +0.0000 [-0.0000, +0.0001] | +0.0% |
| spectral_rolloff | 2 | 0.243 | 21% | +0.0000 [-0.0002, +0.0002] | +0.0% |
| hnr_dyn | 3 | 0.016 | 1% | -0.0000 [-0.0000, +0.0000] | -0.0% |
| spectral_flatness | 1 | 0.237 | 20% | -0.0000 [-0.0001, +0.0001] | -0.0% |
| fut_minus_cur_contrast | 5 | 0.076 | 6% | -0.0000 [-0.0002, +0.0001] | -0.0% |
| topbot_short | 10 | 0.501 | 42% | -0.0001 [-0.0003, +0.0000] | -0.0% |
| f0_dyn | 4 | 0.031 | 3% | -0.0001 [-0.0003, +0.0000] | -0.0% |
| hnr | 2 | 0.254 | 22% | -0.0002 [-0.0003, -0.0000] | -0.0% |
| spectral_moments | 4 | 0.408 | 35% | -0.0002 [-0.0005, +0.0001] | -0.0% |
| contrast_short | 24 | 0.318 | 27% | -0.0004 [-0.0007, -0.0000] | -0.0% |
| cpp | 4 | 0.137 | 12% | -0.0005 [-0.0009, -0.0002] | -0.0% |

Sum of unique shares: 12.3% → shared (not attributable to any one unit): 87.7%

