# LEAN-v20 variance partitioning — patient1ALL, held-out session 450a36e4

493 test chunks (the whole held-out session, never used in training); 500 training chunks from the other sessions; 166 electrodes, grid = first 125. Harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model

- grid: mean r = 0.0690, R² = 1.254% of variance
- all: mean r = 0.0614, R² = 1.014% of variance

## Family level — grid (full R² = 1.254%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 1.173 | 94% | +0.0908 [+0.0819, +0.0996] | +7.2% |
| Interpretable interactions (v3e) | 405 | 1.014 | 81% | +0.0246 [+0.0215, +0.0278] | +2.0% |
| Wavelet scattering | 287 | 0.934 | 74% | +0.0151 [+0.0120, +0.0188] | +1.2% |
| Spectral/cochleagram contrast | 192 | 0.871 | 69% | +0.0085 [+0.0061, +0.0110] | +0.7% |
| Pitch, voicing, voice quality | 122 | 0.669 | 53% | +0.0056 [+0.0047, +0.0065] | +0.4% |
| Spectral shape, flux, formants | 83 | 0.838 | 67% | +0.0049 [+0.0038, +0.0062] | +0.4% |
| Cochleagram core + slow context + lookahead | 960 | 0.950 | 76% | +0.0047 [-0.0021, +0.0118] | +0.4% |

Sum of unique shares: 12.3% → shared (not attributable to any one unit): 87.7%

## Family level — all (full R² = 1.014%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.947 | 93% | +0.0738 [+0.0665, +0.0813] | +7.3% |
| Interpretable interactions (v3e) | 405 | 0.819 | 81% | +0.0228 [+0.0202, +0.0256] | +2.2% |
| Wavelet scattering | 287 | 0.756 | 75% | +0.0112 [+0.0087, +0.0142] | +1.1% |
| Spectral/cochleagram contrast | 192 | 0.705 | 70% | +0.0073 [+0.0053, +0.0095] | +0.7% |
| Pitch, voicing, voice quality | 122 | 0.548 | 54% | +0.0072 [+0.0063, +0.0082] | +0.7% |
| Cochleagram core + slow context + lookahead | 960 | 0.775 | 76% | +0.0069 [+0.0014, +0.0131] | +0.7% |
| Spectral shape, flux, formants | 83 | 0.680 | 67% | +0.0038 [+0.0029, +0.0048] | +0.4% |

Sum of unique shares: 13.1% → shared (not attributable to any one unit): 86.9%

## Section level — grid (full R² = 1.254%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| J. Interactions (v3e) | 405 | 1.014 | 81% | +0.0246 [+0.0215, +0.0278] | +2.0% |
| M. Quadrature modulation bank | 954 | 1.080 | 86% | +0.0227 [+0.0194, +0.0259] | +1.8% |
| E. 1-D temporal modulation | 132 | 0.703 | 56% | +0.0185 [+0.0145, +0.0225] | +1.5% |
| P. Wavelet scattering | 287 | 0.934 | 74% | +0.0151 [+0.0120, +0.0188] | +1.2% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.934 | 74% | +0.0109 [+0.0082, +0.0134] | +0.9% |
| L. Spectral/cochleagram contrast | 192 | 0.871 | 69% | +0.0085 [+0.0061, +0.0110] | +0.7% |
| H. Slow context + lookahead | 600 | 0.924 | 74% | +0.0059 [+0.0001, +0.0123] | +0.5% |
| D. Spectral flux | 36 | 0.571 | 45% | +0.0049 [+0.0040, +0.0058] | +0.4% |
| O. Slow quadrature bank | 360 | 0.877 | 70% | +0.0037 [+0.0026, +0.0048] | +0.3% |
| B. Pitch | 19 | 0.346 | 28% | +0.0029 [+0.0023, +0.0035] | +0.2% |
| N. Voicing gates | 84 | 0.607 | 48% | +0.0017 [+0.0010, +0.0023] | +0.1% |
| K. Voice quality (CPP, region flatness) | 12 | 0.160 | 13% | +0.0010 [+0.0005, +0.0015] | +0.1% |
| A. Cochleagram core | 360 | 0.681 | 54% | +0.0006 [-0.0007, +0.0019] | +0.0% |
| G. Formants | 33 | 0.607 | 48% | +0.0005 [+0.0001, +0.0008] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.044 | 4% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| C. Spectral shape | 14 | 0.605 | 48% | -0.0002 [-0.0007, +0.0002] | -0.0% |

Sum of unique shares: 9.7% → shared (not attributable to any one unit): 90.3%

## Section level — all (full R² = 1.014%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| J. Interactions (v3e) | 405 | 0.819 | 81% | +0.0228 [+0.0202, +0.0256] | +2.2% |
| E. 1-D temporal modulation | 132 | 0.564 | 56% | +0.0189 [+0.0152, +0.0227] | +1.9% |
| M. Quadrature modulation bank | 954 | 0.874 | 86% | +0.0180 [+0.0154, +0.0207] | +1.8% |
| P. Wavelet scattering | 287 | 0.756 | 75% | +0.0112 [+0.0087, +0.0142] | +1.1% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.755 | 75% | +0.0108 [+0.0085, +0.0130] | +1.1% |
| L. Spectral/cochleagram contrast | 192 | 0.705 | 70% | +0.0073 [+0.0053, +0.0095] | +0.7% |
| H. Slow context + lookahead | 600 | 0.756 | 75% | +0.0054 [+0.0004, +0.0109] | +0.5% |
| O. Slow quadrature bank | 360 | 0.719 | 71% | +0.0048 [+0.0038, +0.0058] | +0.5% |
| B. Pitch | 19 | 0.273 | 27% | +0.0045 [+0.0039, +0.0052] | +0.4% |
| D. Spectral flux | 36 | 0.468 | 46% | +0.0033 [+0.0026, +0.0040] | +0.3% |
| A. Cochleagram core | 360 | 0.557 | 55% | +0.0026 [+0.0014, +0.0039] | +0.3% |
| N. Voicing gates | 84 | 0.499 | 49% | +0.0023 [+0.0017, +0.0029] | +0.2% |
| K. Voice quality (CPP, region flatness) | 12 | 0.131 | 13% | +0.0008 [+0.0004, +0.0012] | +0.1% |
| G. Formants | 33 | 0.495 | 49% | +0.0005 [+0.0002, +0.0008] | +0.1% |
| I. Pitch/voicing dynamics | 7 | 0.035 | 3% | -0.0001 [-0.0002, +0.0001] | -0.0% |
| C. Spectral shape | 14 | 0.492 | 49% | -0.0001 [-0.0005, +0.0003] | -0.0% |

Sum of unique shares: 11.2% → shared (not attributable to any one unit): 88.8%

## Group level — grid (full R² = 1.254%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| interact_v3e | 405 | 1.014 | 81% | +0.0246 [+0.0215, +0.0278] | +2.0% |
| modulation | 132 | 0.703 | 56% | +0.0185 [+0.0145, +0.0225] | +1.5% |
| scattering | 287 | 0.934 | 74% | +0.0151 [+0.0120, +0.0188] | +1.2% |
| stmf | 340 | 0.925 | 74% | +0.0098 [+0.0074, +0.0121] | +0.8% |
| future_past | 260 | 0.705 | 56% | +0.0089 [+0.0072, +0.0106] | +0.7% |
| mod_depth | 40 | 0.642 | 51% | +0.0073 [+0.0057, +0.0089] | +0.6% |
| quad_modbank_dyn | 540 | 0.811 | 65% | +0.0057 [+0.0043, +0.0069] | +0.5% |
| flux | 36 | 0.571 | 45% | +0.0049 [+0.0040, +0.0058] | +0.4% |
| quad_xrate_energy | 18 | 0.675 | 54% | +0.0043 [+0.0033, +0.0053] | +0.3% |
| slow_quad_modbank | 180 | 0.835 | 67% | +0.0041 [+0.0033, +0.0049] | +0.3% |
| quad_modbank_acausal | 180 | 0.664 | 53% | +0.0036 [+0.0020, +0.0052] | +0.3% |
| quad_modbank | 180 | 0.864 | 69% | +0.0032 [+0.0025, +0.0039] | +0.3% |
| quad_xrate_energy_dyn | 36 | 0.457 | 36% | +0.0028 [+0.0022, +0.0034] | +0.2% |
| spectral_contrast | 24 | 0.463 | 37% | +0.0024 [+0.0017, +0.0032] | +0.2% |
| stmf_mean_gabor | 17 | 0.071 | 6% | +0.0024 [+0.0016, +0.0031] | +0.2% |
| pitch_salience | 11 | 0.297 | 24% | +0.0023 [+0.0017, +0.0028] | +0.2% |
| contrast_med | 24 | 0.406 | 32% | +0.0018 [+0.0013, +0.0024] | +0.1% |
| region_flat | 8 | 0.083 | 7% | +0.0016 [+0.0011, +0.0020] | +0.1% |
| gate_signals | 6 | 0.001 | 0% | +0.0015 [+0.0009, +0.0021] | +0.1% |
| formant_dyn | 9 | 0.174 | 14% | +0.0014 [+0.0011, +0.0016] | +0.1% |
| adapt_delta | 300 | 0.647 | 52% | +0.0014 [+0.0002, +0.0026] | +0.1% |
| gate_signals_dyn | 18 | 0.001 | 0% | +0.0012 [+0.0007, +0.0018] | +0.1% |
| phonetic | 5 | 0.504 | 40% | +0.0011 [+0.0008, +0.0015] | +0.1% |
| cochlea_voiced_gated | 40 | 0.571 | 46% | +0.0011 [+0.0009, +0.0013] | +0.1% |
| fut_cochlea_contrast | 5 | 0.336 | 27% | +0.0010 [+0.0007, +0.0013] | +0.1% |
| multitau_contrast | 10 | 0.539 | 43% | +0.0008 [+0.0003, +0.0013] | +0.1% |
| fut_med_contrast_6 | 5 | 0.150 | 12% | +0.0007 [+0.0004, +0.0010] | +0.1% |
| cube_fut_contrast | 5 | 0.381 | 30% | +0.0006 [+0.0003, +0.0009] | +0.0% |
| short_cochlea_contrast | 20 | 0.308 | 25% | +0.0006 [+0.0003, +0.0009] | +0.0% |
| topbot | 10 | 0.595 | 47% | +0.0006 [+0.0003, +0.0008] | +0.0% |
| pitch_dyn | 6 | 0.211 | 17% | +0.0006 [+0.0004, +0.0007] | +0.0% |
| cochleagram | 60 | 0.592 | 47% | +0.0005 [-0.0000, +0.0010] | +0.0% |
| formants | 24 | 0.558 | 44% | +0.0005 [+0.0001, +0.0008] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.309 | 25% | +0.0004 [+0.0003, +0.0006] | +0.0% |
| med_adapt_contrast | 5 | 0.411 | 33% | +0.0004 [+0.0002, +0.0006] | +0.0% |
| spectral_rolloff | 2 | 0.265 | 21% | +0.0003 [+0.0001, +0.0005] | +0.0% |
| med_cochlea_contrast | 20 | 0.372 | 30% | +0.0002 [+0.0001, +0.0005] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.405 | 32% | +0.0002 [-0.0000, +0.0003] | +0.0% |
| fut_minus_cur_contrast | 5 | 0.073 | 6% | +0.0001 [+0.0000, +0.0002] | +0.0% |
| f0_dyn | 4 | 0.031 | 2% | +0.0001 [-0.0000, +0.0003] | +0.0% |
| pitch_summary | 2 | 0.236 | 19% | +0.0000 [-0.0000, +0.0001] | +0.0% |
| hnr | 2 | 0.275 | 22% | +0.0000 [-0.0001, +0.0001] | +0.0% |
| hnr_dyn | 3 | 0.016 | 1% | -0.0000 [-0.0000, +0.0000] | -0.0% |
| spectral_flatness | 1 | 0.249 | 20% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| spectral_moments | 4 | 0.428 | 34% | -0.0000 [-0.0003, +0.0002] | -0.0% |
| topbot_short | 10 | 0.519 | 41% | -0.0001 [-0.0002, +0.0001] | -0.0% |
| cochlea_contrast | 20 | 0.396 | 32% | -0.0001 [-0.0005, +0.0004] | -0.0% |
| cpp | 4 | 0.140 | 11% | -0.0001 [-0.0004, +0.0001] | -0.0% |
| contrast_short | 24 | 0.332 | 27% | -0.0002 [-0.0005, +0.0001] | -0.0% |
| slow_quad_modbank_dyn | 180 | 0.677 | 54% | -0.0002 [-0.0007, +0.0003] | -0.0% |
| tonic | 300 | 0.662 | 53% | -0.0016 [-0.0069, +0.0043] | -0.1% |

Sum of unique shares: 10.8% → shared (not attributable to any one unit): 89.2%

## Group level — all (full R² = 1.014%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| interact_v3e | 405 | 0.819 | 81% | +0.0228 [+0.0202, +0.0256] | +2.2% |
| modulation | 132 | 0.564 | 56% | +0.0189 [+0.0152, +0.0227] | +1.9% |
| scattering | 287 | 0.756 | 75% | +0.0112 [+0.0087, +0.0142] | +1.1% |
| mod_depth | 40 | 0.524 | 52% | +0.0103 [+0.0088, +0.0121] | +1.0% |
| stmf | 340 | 0.749 | 74% | +0.0099 [+0.0079, +0.0119] | +1.0% |
| future_past | 260 | 0.572 | 56% | +0.0088 [+0.0073, +0.0104] | +0.9% |
| quad_modbank_dyn | 540 | 0.645 | 64% | +0.0043 [+0.0032, +0.0053] | +0.4% |
| contrast_med | 24 | 0.320 | 32% | +0.0037 [+0.0031, +0.0044] | +0.4% |
| quad_xrate_energy | 18 | 0.546 | 54% | +0.0035 [+0.0027, +0.0043] | +0.3% |
| flux | 36 | 0.468 | 46% | +0.0033 [+0.0026, +0.0040] | +0.3% |
| adapt_delta | 300 | 0.520 | 51% | +0.0033 [+0.0021, +0.0045] | +0.3% |
| cochleagram | 60 | 0.481 | 47% | +0.0032 [+0.0026, +0.0039] | +0.3% |
| slow_quad_modbank | 180 | 0.681 | 67% | +0.0031 [+0.0025, +0.0037] | +0.3% |
| slow_quad_modbank_dyn | 180 | 0.542 | 53% | +0.0030 [+0.0025, +0.0036] | +0.3% |
| quad_modbank | 180 | 0.699 | 69% | +0.0027 [+0.0021, +0.0033] | +0.3% |
| quad_modbank_acausal | 180 | 0.532 | 52% | +0.0026 [+0.0012, +0.0040] | +0.3% |
| quad_xrate_energy_dyn | 36 | 0.364 | 36% | +0.0020 [+0.0016, +0.0025] | +0.2% |
| region_flat | 8 | 0.070 | 7% | +0.0016 [+0.0012, +0.0019] | +0.2% |
| stmf_mean_gabor | 17 | 0.059 | 6% | +0.0015 [+0.0009, +0.0021] | +0.1% |
| pitch_salience | 11 | 0.233 | 23% | +0.0013 [+0.0008, +0.0018] | +0.1% |
| spectral_contrast | 24 | 0.364 | 36% | +0.0011 [+0.0004, +0.0018] | +0.1% |
| gate_signals | 6 | 0.001 | 0% | +0.0011 [+0.0006, +0.0016] | +0.1% |
| formant_dyn | 9 | 0.138 | 14% | +0.0010 [+0.0008, +0.0013] | +0.1% |
| phonetic | 5 | 0.414 | 41% | +0.0009 [+0.0006, +0.0012] | +0.1% |
| gate_signals_dyn | 18 | 0.001 | 0% | +0.0008 [+0.0004, +0.0012] | +0.1% |
| fut_cochlea_contrast | 5 | 0.270 | 27% | +0.0007 [+0.0005, +0.0010] | +0.1% |
| cochlea_voiced_gated | 40 | 0.465 | 46% | +0.0007 [+0.0005, +0.0008] | +0.1% |
| cube_fut_contrast | 5 | 0.312 | 31% | +0.0006 [+0.0003, +0.0009] | +0.1% |
| formants | 24 | 0.455 | 45% | +0.0005 [+0.0002, +0.0008] | +0.1% |
| fut_med_contrast_6 | 5 | 0.124 | 12% | +0.0005 [+0.0003, +0.0007] | +0.0% |
| pitch_dyn | 6 | 0.164 | 16% | +0.0004 [+0.0003, +0.0006] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.249 | 25% | +0.0004 [+0.0002, +0.0005] | +0.0% |
| multitau_contrast | 10 | 0.431 | 42% | +0.0003 [-0.0001, +0.0007] | +0.0% |
| topbot | 10 | 0.483 | 48% | +0.0003 [+0.0001, +0.0005] | +0.0% |
| med_adapt_contrast | 5 | 0.329 | 32% | +0.0002 [+0.0000, +0.0004] | +0.0% |
| med_cochlea_contrast | 20 | 0.299 | 30% | +0.0002 [+0.0000, +0.0004] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.324 | 32% | +0.0001 [-0.0000, +0.0003] | +0.0% |
| spectral_rolloff | 2 | 0.213 | 21% | +0.0001 [-0.0001, +0.0002] | +0.0% |
| pitch_summary | 2 | 0.186 | 18% | +0.0000 [-0.0000, +0.0001] | +0.0% |
| hnr_dyn | 3 | 0.012 | 1% | -0.0000 [-0.0000, +0.0000] | -0.0% |
| spectral_flatness | 1 | 0.196 | 19% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| fut_minus_cur_contrast | 5 | 0.058 | 6% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| topbot_short | 10 | 0.422 | 42% | -0.0001 [-0.0002, +0.0001] | -0.0% |
| f0_dyn | 4 | 0.024 | 2% | -0.0001 [-0.0002, +0.0001] | -0.0% |
| spectral_moments | 4 | 0.343 | 34% | -0.0001 [-0.0004, +0.0001] | -0.0% |
| hnr | 2 | 0.218 | 21% | -0.0002 [-0.0003, -0.0000] | -0.0% |
| short_cochlea_contrast | 20 | 0.247 | 24% | -0.0002 [-0.0006, +0.0002] | -0.0% |
| cochlea_contrast | 20 | 0.317 | 31% | -0.0003 [-0.0007, +0.0001] | -0.0% |
| contrast_short | 24 | 0.264 | 26% | -0.0004 [-0.0006, -0.0001] | -0.0% |
| cpp | 4 | 0.113 | 11% | -0.0005 [-0.0008, -0.0002] | -0.1% |
| tonic | 300 | 0.538 | 53% | -0.0020 [-0.0067, +0.0031] | -0.2% |

Sum of unique shares: 12.6% → shared (not attributable to any one unit): 87.4%

