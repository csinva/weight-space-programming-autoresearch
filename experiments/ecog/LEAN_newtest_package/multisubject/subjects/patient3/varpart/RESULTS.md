# LEAN-v20 variance partitioning — patient3, held-out session e3d31b95

416 test chunks (the whole held-out session, never used in training); 500 training chunks from the other sessions; 116 electrodes, grid = first 63. Harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model

- grid: mean r = 0.0604, R² = 1.128% of variance
- all: mean r = 0.0523, R² = 0.838% of variance

## Family level — grid (full R² = 1.128%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 1.111 | 99% | +0.0694 [+0.0610, +0.0777] | +6.2% |
| Pitch, voicing, voice quality | 122 | 0.669 | 59% | +0.0082 [+0.0058, +0.0107] | +0.7% |
| Wavelet scattering | 287 | 0.872 | 77% | +0.0077 [+0.0051, +0.0105] | +0.7% |
| Interpretable interactions (v3e) | 405 | 0.943 | 84% | +0.0032 [-0.0002, +0.0065] | +0.3% |
| Spectral shape, flux, formants | 83 | 0.796 | 71% | -0.0021 [-0.0044, +0.0001] | -0.2% |
| Spectral/cochleagram contrast | 192 | 0.882 | 78% | -0.0023 [-0.0057, +0.0013] | -0.2% |
| Cochleagram core + slow context + lookahead | 960 | 0.872 | 77% | -0.0247 [-0.0341, -0.0150] | -2.2% |

Sum of unique shares: 5.3% → shared (not attributable to any one unit): 94.7%

## Family level — all (full R² = 0.838%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.823 | 98% | +0.0427 [+0.0359, +0.0492] | +5.1% |
| Wavelet scattering | 287 | 0.644 | 77% | +0.0065 [+0.0044, +0.0088] | +0.8% |
| Pitch, voicing, voice quality | 122 | 0.482 | 58% | +0.0061 [+0.0042, +0.0080] | +0.7% |
| Spectral/cochleagram contrast | 192 | 0.654 | 78% | +0.0015 [-0.0010, +0.0042] | +0.2% |
| Interpretable interactions (v3e) | 405 | 0.675 | 81% | +0.0009 [-0.0016, +0.0032] | +0.1% |
| Spectral shape, flux, formants | 83 | 0.572 | 68% | -0.0011 [-0.0026, +0.0004] | -0.1% |
| Cochleagram core + slow context + lookahead | 960 | 0.673 | 80% | -0.0169 [-0.0252, -0.0077] | -2.0% |

Sum of unique shares: 4.7% → shared (not attributable to any one unit): 95.3%

## Section level — grid (full R² = 1.128%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| E. 1-D temporal modulation | 132 | 0.742 | 66% | +0.0325 [+0.0269, +0.0378] | +2.9% |
| P. Wavelet scattering | 287 | 0.872 | 77% | +0.0077 [+0.0051, +0.0105] | +0.7% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.912 | 81% | +0.0067 [+0.0041, +0.0092] | +0.6% |
| K. Voice quality (CPP, region flatness) | 12 | 0.260 | 23% | +0.0061 [+0.0040, +0.0083] | +0.5% |
| C. Spectral shape | 14 | 0.595 | 53% | +0.0055 [+0.0033, +0.0077] | +0.5% |
| D. Spectral flux | 36 | 0.566 | 50% | +0.0033 [+0.0014, +0.0053] | +0.3% |
| J. Interactions (v3e) | 405 | 0.943 | 84% | +0.0032 [-0.0002, +0.0065] | +0.3% |
| M. Quadrature modulation bank | 954 | 1.022 | 91% | +0.0022 [-0.0025, +0.0067] | +0.2% |
| B. Pitch | 19 | 0.280 | 25% | +0.0021 [+0.0014, +0.0029] | +0.2% |
| N. Voicing gates | 84 | 0.574 | 51% | +0.0003 [-0.0008, +0.0012] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.024 | 2% | -0.0001 [-0.0002, -0.0000] | -0.0% |
| O. Slow quadrature bank | 360 | 0.838 | 74% | -0.0018 [-0.0044, +0.0006] | -0.2% |
| L. Spectral/cochleagram contrast | 192 | 0.882 | 78% | -0.0023 [-0.0057, +0.0013] | -0.2% |
| A. Cochleagram core | 360 | 0.621 | 55% | -0.0046 [-0.0081, -0.0011] | -0.4% |
| G. Formants | 33 | 0.567 | 50% | -0.0075 [-0.0096, -0.0057] | -0.7% |
| H. Slow context + lookahead | 600 | 0.861 | 76% | -0.0249 [-0.0337, -0.0159] | -2.2% |

Sum of unique shares: 2.5% → shared (not attributable to any one unit): 97.5%

## Section level — all (full R² = 0.838%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| E. 1-D temporal modulation | 132 | 0.540 | 64% | +0.0189 [+0.0156, +0.0222] | +2.3% |
| P. Wavelet scattering | 287 | 0.644 | 77% | +0.0065 [+0.0044, +0.0088] | +0.8% |
| K. Voice quality (CPP, region flatness) | 12 | 0.190 | 23% | +0.0057 [+0.0041, +0.0076] | +0.7% |
| C. Spectral shape | 14 | 0.429 | 51% | +0.0030 [+0.0018, +0.0042] | +0.4% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.687 | 82% | +0.0030 [+0.0010, +0.0049] | +0.4% |
| D. Spectral flux | 36 | 0.425 | 51% | +0.0029 [+0.0015, +0.0043] | +0.3% |
| B. Pitch | 19 | 0.199 | 24% | +0.0019 [+0.0014, +0.0025] | +0.2% |
| L. Spectral/cochleagram contrast | 192 | 0.654 | 78% | +0.0015 [-0.0010, +0.0042] | +0.2% |
| J. Interactions (v3e) | 405 | 0.675 | 81% | +0.0009 [-0.0016, +0.0032] | +0.1% |
| N. Voicing gates | 84 | 0.424 | 51% | -0.0002 [-0.0012, +0.0007] | -0.0% |
| I. Pitch/voicing dynamics | 7 | 0.016 | 2% | -0.0003 [-0.0006, -0.0001] | -0.0% |
| O. Slow quadrature bank | 360 | 0.646 | 77% | -0.0007 [-0.0026, +0.0009] | -0.1% |
| M. Quadrature modulation bank | 954 | 0.771 | 92% | -0.0016 [-0.0060, +0.0026] | -0.2% |
| G. Formants | 33 | 0.406 | 48% | -0.0038 [-0.0050, -0.0026] | -0.5% |
| A. Cochleagram core | 360 | 0.463 | 55% | -0.0056 [-0.0089, -0.0026] | -0.7% |
| H. Slow context + lookahead | 600 | 0.672 | 80% | -0.0188 [-0.0269, -0.0096] | -2.3% |

Sum of unique shares: 1.6% → shared (not attributable to any one unit): 98.4%

## Group level — grid (full R² = 1.128%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| modulation | 132 | 0.742 | 66% | +0.0325 [+0.0269, +0.0378] | +2.9% |
| quad_xrate_energy | 18 | 0.651 | 58% | +0.0078 [+0.0054, +0.0104] | +0.7% |
| scattering | 287 | 0.872 | 77% | +0.0077 [+0.0051, +0.0105] | +0.7% |
| region_flat | 8 | 0.126 | 11% | +0.0057 [+0.0036, +0.0078] | +0.5% |
| future_past | 260 | 0.703 | 62% | +0.0055 [+0.0033, +0.0074] | +0.5% |
| quad_xrate_energy_dyn | 36 | 0.310 | 28% | +0.0053 [+0.0033, +0.0072] | +0.5% |
| stmf | 340 | 0.905 | 80% | +0.0038 [+0.0012, +0.0062] | +0.3% |
| contrast_med | 24 | 0.378 | 34% | +0.0037 [+0.0022, +0.0054] | +0.3% |
| gate_signals_dyn | 18 | 0.000 | 0% | +0.0035 [+0.0021, +0.0048] | +0.3% |
| flux | 36 | 0.566 | 50% | +0.0033 [+0.0014, +0.0053] | +0.3% |
| slow_quad_modbank | 180 | 0.785 | 70% | +0.0033 [+0.0016, +0.0050] | +0.3% |
| interact_v3e | 405 | 0.943 | 84% | +0.0032 [-0.0002, +0.0065] | +0.3% |
| multitau_contrast | 10 | 0.664 | 59% | +0.0029 [+0.0023, +0.0035] | +0.3% |
| med_cochlea_contrast | 20 | 0.389 | 34% | +0.0021 [+0.0012, +0.0031] | +0.2% |
| pitch_salience | 11 | 0.214 | 19% | +0.0017 [+0.0010, +0.0023] | +0.1% |
| spectral_moments | 4 | 0.382 | 34% | +0.0016 [+0.0006, +0.0028] | +0.1% |
| cochlea_contrast | 20 | 0.409 | 36% | +0.0012 [+0.0006, +0.0017] | +0.1% |
| mod_depth | 40 | 0.662 | 59% | +0.0012 [-0.0017, +0.0040] | +0.1% |
| phonetic | 5 | 0.489 | 43% | +0.0008 [+0.0001, +0.0015] | +0.1% |
| fut_minus_cur_contrast | 5 | 0.050 | 4% | +0.0004 [-0.0000, +0.0009] | +0.0% |
| formant_dyn | 9 | 0.251 | 22% | +0.0004 [-0.0000, +0.0008] | +0.0% |
| fut_cochlea_contrast | 5 | 0.371 | 33% | +0.0003 [+0.0000, +0.0005] | +0.0% |
| cpp | 4 | 0.197 | 18% | +0.0003 [+0.0001, +0.0005] | +0.0% |
| pitch_summary | 2 | 0.199 | 18% | +0.0002 [+0.0001, +0.0003] | +0.0% |
| gate_signals | 6 | 0.001 | 0% | +0.0002 [-0.0006, +0.0008] | +0.0% |
| short_cochlea_contrast | 20 | 0.339 | 30% | +0.0001 [-0.0007, +0.0010] | +0.0% |
| fut_med_contrast_6 | 5 | 0.195 | 17% | +0.0001 [-0.0000, +0.0002] | +0.0% |
| spectral_flatness | 1 | 0.108 | 10% | +0.0000 [-0.0001, +0.0002] | +0.0% |
| cochlea_voiced_gated | 40 | 0.545 | 48% | -0.0000 [-0.0003, +0.0002] | -0.0% |
| f0_dyn | 4 | 0.022 | 2% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| hnr_dyn | 3 | 0.003 | 0% | -0.0000 [-0.0001, +0.0001] | -0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.311 | 28% | -0.0001 [-0.0001, +0.0000] | -0.0% |
| hnr | 2 | 0.073 | 6% | -0.0001 [-0.0002, +0.0000] | -0.0% |
| fut_cochlea_contrast_med | 5 | 0.353 | 31% | -0.0001 [-0.0003, +0.0000] | -0.0% |
| stmf_mean_gabor | 17 | 0.058 | 5% | -0.0002 [-0.0006, +0.0003] | -0.0% |
| topbot | 10 | 0.560 | 50% | -0.0002 [-0.0005, +0.0000] | -0.0% |
| med_adapt_contrast | 5 | 0.528 | 47% | -0.0002 [-0.0005, -0.0000] | -0.0% |
| topbot_short | 10 | 0.494 | 44% | -0.0003 [-0.0005, -0.0001] | -0.0% |
| cube_fut_contrast | 5 | 0.436 | 39% | -0.0003 [-0.0006, -0.0000] | -0.0% |
| spectral_rolloff | 2 | 0.205 | 18% | -0.0004 [-0.0006, -0.0002] | -0.0% |
| pitch_dyn | 6 | 0.177 | 16% | -0.0004 [-0.0007, -0.0002] | -0.0% |
| contrast_short | 24 | 0.335 | 30% | -0.0011 [-0.0031, +0.0008] | -0.1% |
| quad_modbank_dyn | 540 | 0.722 | 64% | -0.0012 [-0.0043, +0.0018] | -0.1% |
| spectral_contrast | 24 | 0.406 | 36% | -0.0028 [-0.0046, -0.0010] | -0.2% |
| quad_modbank | 180 | 0.885 | 78% | -0.0028 [-0.0050, -0.0008] | -0.2% |
| slow_quad_modbank_dyn | 180 | 0.571 | 51% | -0.0036 [-0.0058, -0.0016] | -0.3% |
| adapt_delta | 300 | 0.537 | 48% | -0.0041 [-0.0072, -0.0010] | -0.4% |
| cochleagram | 60 | 0.566 | 50% | -0.0070 [-0.0094, -0.0047] | -0.6% |
| quad_modbank_acausal | 180 | 0.579 | 51% | -0.0071 [-0.0104, -0.0037] | -0.6% |
| formants | 24 | 0.517 | 46% | -0.0076 [-0.0097, -0.0057] | -0.7% |
| tonic | 300 | 0.568 | 50% | -0.0291 [-0.0383, -0.0200] | -2.6% |

Sum of unique shares: 2.7% → shared (not attributable to any one unit): 97.3%

## Group level — all (full R² = 0.838%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| modulation | 132 | 0.540 | 64% | +0.0189 [+0.0156, +0.0222] | +2.3% |
| scattering | 287 | 0.644 | 77% | +0.0065 [+0.0044, +0.0088] | +0.8% |
| quad_xrate_energy | 18 | 0.485 | 58% | +0.0047 [+0.0032, +0.0062] | +0.6% |
| region_flat | 8 | 0.092 | 11% | +0.0045 [+0.0030, +0.0060] | +0.5% |
| future_past | 260 | 0.556 | 66% | +0.0041 [+0.0027, +0.0054] | +0.5% |
| quad_xrate_energy_dyn | 36 | 0.234 | 28% | +0.0030 [+0.0019, +0.0042] | +0.4% |
| flux | 36 | 0.425 | 51% | +0.0029 [+0.0015, +0.0043] | +0.3% |
| slow_quad_modbank | 180 | 0.612 | 73% | +0.0027 [+0.0015, +0.0039] | +0.3% |
| gate_signals_dyn | 18 | 0.000 | 0% | +0.0025 [+0.0015, +0.0035] | +0.3% |
| contrast_med | 24 | 0.257 | 31% | +0.0021 [+0.0013, +0.0030] | +0.3% |
| pitch_salience | 11 | 0.146 | 17% | +0.0020 [+0.0015, +0.0026] | +0.2% |
| med_cochlea_contrast | 20 | 0.263 | 31% | +0.0020 [+0.0014, +0.0027] | +0.2% |
| multitau_contrast | 10 | 0.450 | 54% | +0.0014 [+0.0009, +0.0019] | +0.2% |
| spectral_moments | 4 | 0.271 | 32% | +0.0010 [+0.0003, +0.0018] | +0.1% |
| interact_v3e | 405 | 0.675 | 81% | +0.0009 [-0.0016, +0.0032] | +0.1% |
| cochlea_contrast | 20 | 0.277 | 33% | +0.0008 [+0.0003, +0.0012] | +0.1% |
| gate_signals | 6 | 0.001 | 0% | +0.0007 [+0.0000, +0.0012] | +0.1% |
| stmf | 340 | 0.683 | 82% | +0.0006 [-0.0017, +0.0028] | +0.1% |
| fut_minus_cur_contrast | 5 | 0.037 | 4% | +0.0006 [+0.0002, +0.0009] | +0.1% |
| mod_depth | 40 | 0.459 | 55% | +0.0006 [-0.0014, +0.0026] | +0.1% |
| fut_med_contrast_6 | 5 | 0.157 | 19% | +0.0004 [+0.0002, +0.0006] | +0.1% |
| phonetic | 5 | 0.364 | 43% | +0.0004 [-0.0000, +0.0008] | +0.0% |
| stmf_mean_gabor | 17 | 0.043 | 5% | +0.0002 [-0.0003, +0.0008] | +0.0% |
| pitch_dyn | 6 | 0.116 | 14% | +0.0002 [+0.0000, +0.0004] | +0.0% |
| contrast_short | 24 | 0.226 | 27% | +0.0002 [-0.0010, +0.0014] | +0.0% |
| formant_dyn | 9 | 0.171 | 20% | +0.0002 [-0.0001, +0.0004] | +0.0% |
| short_cochlea_contrast | 20 | 0.229 | 27% | +0.0001 [-0.0005, +0.0007] | +0.0% |
| pitch_summary | 2 | 0.136 | 16% | +0.0001 [+0.0000, +0.0002] | +0.0% |
| fut_cochlea_contrast | 5 | 0.275 | 33% | +0.0000 [-0.0002, +0.0003] | +0.0% |
| spectral_flatness | 1 | 0.070 | 8% | +0.0000 [-0.0000, +0.0001] | +0.0% |
| hnr_dyn | 3 | 0.002 | 0% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.225 | 27% | -0.0001 [-0.0002, +0.0000] | -0.0% |
| cpp | 4 | 0.143 | 17% | -0.0001 [-0.0004, +0.0002] | -0.0% |
| topbot_short | 10 | 0.356 | 43% | -0.0001 [-0.0003, -0.0000] | -0.0% |
| spectral_rolloff | 2 | 0.149 | 18% | -0.0002 [-0.0004, -0.0001] | -0.0% |
| f0_dyn | 4 | 0.016 | 2% | -0.0003 [-0.0006, -0.0000] | -0.0% |
| cube_fut_contrast | 5 | 0.353 | 42% | -0.0003 [-0.0007, +0.0001] | -0.0% |
| fut_cochlea_contrast_med | 5 | 0.263 | 31% | -0.0003 [-0.0006, -0.0000] | -0.0% |
| hnr | 2 | 0.050 | 6% | -0.0004 [-0.0006, -0.0002] | -0.0% |
| topbot | 10 | 0.409 | 49% | -0.0004 [-0.0007, -0.0002] | -0.1% |
| med_adapt_contrast | 5 | 0.357 | 43% | -0.0005 [-0.0007, -0.0003] | -0.1% |
| cochlea_voiced_gated | 40 | 0.401 | 48% | -0.0005 [-0.0010, -0.0001] | -0.1% |
| spectral_contrast | 24 | 0.273 | 33% | -0.0019 [-0.0030, -0.0008] | -0.2% |
| quad_modbank | 180 | 0.681 | 81% | -0.0021 [-0.0035, -0.0008] | -0.3% |
| slow_quad_modbank_dyn | 180 | 0.443 | 53% | -0.0027 [-0.0041, -0.0012] | -0.3% |
| quad_modbank_dyn | 540 | 0.543 | 65% | -0.0030 [-0.0063, -0.0001] | -0.4% |
| cochleagram | 60 | 0.420 | 50% | -0.0034 [-0.0048, -0.0021] | -0.4% |
| formants | 24 | 0.375 | 45% | -0.0039 [-0.0051, -0.0027] | -0.5% |
| adapt_delta | 300 | 0.383 | 46% | -0.0054 [-0.0086, -0.0026] | -0.6% |
| quad_modbank_acausal | 180 | 0.453 | 54% | -0.0055 [-0.0080, -0.0030] | -0.7% |
| tonic | 300 | 0.425 | 51% | -0.0226 [-0.0311, -0.0132] | -2.7% |

Sum of unique shares: 1.3% → shared (not attributable to any one unit): 98.7%

