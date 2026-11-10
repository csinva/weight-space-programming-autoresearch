# LEAN-v20 variance partitioning — patient2, held-out session 236d910e

318 test chunks (the whole held-out session, never used in training); 500 training chunks from the other sessions; 235 electrodes, grid = first 127. Harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model

- grid: mean r = 0.0643, R² = 0.769% of variance
- all: mean r = 0.0787, R² = 1.006% of variance

## Family level — grid (full R² = 0.769%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.802 | 104% | +0.0369 [+0.0260, +0.0479] | +4.8% |
| Interpretable interactions (v3e) | 405 | 0.585 | 76% | +0.0060 [+0.0007, +0.0113] | +0.8% |
| Wavelet scattering | 287 | 0.605 | 79% | +0.0040 [+0.0002, +0.0078] | +0.5% |
| Pitch, voicing, voice quality | 122 | 0.521 | 68% | +0.0028 [-0.0004, +0.0058] | +0.4% |
| Spectral/cochleagram contrast | 192 | 0.642 | 84% | +0.0009 [-0.0032, +0.0051] | +0.1% |
| Spectral shape, flux, formants | 83 | 0.486 | 63% | -0.0016 [-0.0055, +0.0020] | -0.2% |
| Cochleagram core + slow context + lookahead | 960 | 0.716 | 93% | -0.0507 [-0.0678, -0.0344] | -6.6% |

Sum of unique shares: -0.2% → shared (not attributable to any one unit): 100.2%

## Family level — all (full R² = 1.006%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 1.061 | 105% | +0.0367 [+0.0172, +0.0568] | +3.7% |
| Spectral shape, flux, formants | 83 | 0.563 | 56% | +0.0012 [-0.0037, +0.0061] | +0.1% |
| Pitch, voicing, voice quality | 122 | 0.634 | 63% | +0.0004 [-0.0045, +0.0051] | +0.0% |
| Interpretable interactions (v3e) | 405 | 0.706 | 70% | -0.0004 [-0.0076, +0.0074] | -0.0% |
| Spectral/cochleagram contrast | 192 | 0.800 | 80% | -0.0029 [-0.0101, +0.0040] | -0.3% |
| Wavelet scattering | 287 | 0.712 | 71% | -0.0036 [-0.0107, +0.0034] | -0.4% |
| Cochleagram core + slow context + lookahead | 960 | 0.994 | 99% | -0.0603 [-0.0913, -0.0324] | -6.0% |

Sum of unique shares: -2.9% → shared (not attributable to any one unit): 102.9%

## Section level — grid (full R² = 0.769%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| E. 1-D temporal modulation | 132 | 0.561 | 73% | +0.0127 [+0.0100, +0.0154] | +1.6% |
| B. Pitch | 19 | 0.204 | 27% | +0.0062 [+0.0049, +0.0077] | +0.8% |
| J. Interactions (v3e) | 405 | 0.585 | 76% | +0.0060 [+0.0007, +0.0113] | +0.8% |
| M. Quadrature modulation bank | 954 | 0.723 | 94% | +0.0058 [+0.0004, +0.0118] | +0.8% |
| P. Wavelet scattering | 287 | 0.605 | 79% | +0.0040 [+0.0002, +0.0078] | +0.5% |
| D. Spectral flux | 36 | 0.349 | 45% | +0.0039 [+0.0022, +0.0058] | +0.5% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.693 | 90% | +0.0018 [-0.0028, +0.0065] | +0.2% |
| N. Voicing gates | 84 | 0.478 | 62% | +0.0016 [+0.0006, +0.0026] | +0.2% |
| L. Spectral/cochleagram contrast | 192 | 0.642 | 84% | +0.0009 [-0.0032, +0.0051] | +0.1% |
| I. Pitch/voicing dynamics | 7 | 0.068 | 9% | +0.0003 [+0.0000, +0.0006] | +0.0% |
| G. Formants | 33 | 0.397 | 52% | -0.0023 [-0.0041, -0.0005] | -0.3% |
| C. Spectral shape | 14 | 0.394 | 51% | -0.0036 [-0.0068, -0.0008] | -0.5% |
| K. Voice quality (CPP, region flatness) | 12 | 0.206 | 27% | -0.0041 [-0.0072, -0.0013] | -0.5% |
| O. Slow quadrature bank | 360 | 0.652 | 85% | -0.0048 [-0.0090, -0.0006] | -0.6% |
| A. Cochleagram core | 360 | 0.502 | 65% | -0.0313 [-0.0386, -0.0243] | -4.1% |
| H. Slow context + lookahead | 600 | 0.762 | 99% | -0.0348 [-0.0477, -0.0212] | -4.5% |

Sum of unique shares: -4.9% → shared (not attributable to any one unit): 104.9%

## Section level — all (full R² = 1.006%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| E. 1-D temporal modulation | 132 | 0.660 | 66% | +0.0108 [+0.0074, +0.0141] | +1.1% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.950 | 94% | +0.0077 [+0.0011, +0.0149] | +0.8% |
| D. Spectral flux | 36 | 0.364 | 36% | +0.0055 [+0.0024, +0.0088] | +0.5% |
| B. Pitch | 19 | 0.210 | 21% | +0.0053 [+0.0034, +0.0073] | +0.5% |
| N. Voicing gates | 84 | 0.603 | 60% | +0.0007 [-0.0003, +0.0017] | +0.1% |
| I. Pitch/voicing dynamics | 7 | 0.082 | 8% | +0.0001 [-0.0002, +0.0004] | +0.0% |
| J. Interactions (v3e) | 405 | 0.706 | 70% | -0.0004 [-0.0076, +0.0074] | -0.0% |
| M. Quadrature modulation bank | 954 | 0.944 | 94% | -0.0012 [-0.0108, +0.0089] | -0.1% |
| C. Spectral shape | 14 | 0.498 | 50% | -0.0019 [-0.0047, +0.0009] | -0.2% |
| G. Formants | 33 | 0.457 | 45% | -0.0023 [-0.0045, -0.0001] | -0.2% |
| L. Spectral/cochleagram contrast | 192 | 0.800 | 80% | -0.0029 [-0.0101, +0.0040] | -0.3% |
| P. Wavelet scattering | 287 | 0.712 | 71% | -0.0036 [-0.0107, +0.0034] | -0.4% |
| K. Voice quality (CPP, region flatness) | 12 | 0.237 | 24% | -0.0048 [-0.0095, -0.0006] | -0.5% |
| O. Slow quadrature bank | 360 | 0.879 | 87% | -0.0167 [-0.0238, -0.0097] | -1.7% |
| H. Slow context + lookahead | 600 | 1.052 | 105% | -0.0335 [-0.0561, -0.0101] | -3.3% |
| A. Cochleagram core | 360 | 0.646 | 64% | -0.0373 [-0.0509, -0.0258] | -3.7% |

Sum of unique shares: -7.4% → shared (not attributable to any one unit): 107.4%

## Group level — grid (full R² = 0.769%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| modulation | 132 | 0.561 | 73% | +0.0127 [+0.0100, +0.0154] | +1.6% |
| topbot | 10 | 0.432 | 56% | +0.0061 [+0.0047, +0.0075] | +0.8% |
| interact_v3e | 405 | 0.585 | 76% | +0.0060 [+0.0007, +0.0113] | +0.8% |
| quad_modbank | 180 | 0.610 | 79% | +0.0055 [+0.0027, +0.0083] | +0.7% |
| scattering | 287 | 0.605 | 79% | +0.0040 [+0.0002, +0.0078] | +0.5% |
| flux | 36 | 0.349 | 45% | +0.0039 [+0.0022, +0.0058] | +0.5% |
| quad_modbank_acausal | 180 | 0.285 | 37% | +0.0036 [-0.0003, +0.0080] | +0.5% |
| topbot_short | 10 | 0.409 | 53% | +0.0036 [+0.0028, +0.0044] | +0.5% |
| pitch_dyn | 6 | 0.067 | 9% | +0.0031 [+0.0022, +0.0040] | +0.4% |
| quad_xrate_energy | 18 | 0.438 | 57% | +0.0030 [+0.0012, +0.0047] | +0.4% |
| pitch_salience | 11 | 0.126 | 16% | +0.0027 [+0.0019, +0.0035] | +0.3% |
| cochlea_voiced_gated | 40 | 0.449 | 58% | +0.0025 [+0.0019, +0.0032] | +0.3% |
| cpp | 4 | 0.128 | 17% | +0.0025 [+0.0018, +0.0033] | +0.3% |
| stmf | 340 | 0.693 | 90% | +0.0022 [-0.0023, +0.0069] | +0.3% |
| med_cochlea_contrast | 20 | 0.330 | 43% | +0.0021 [+0.0010, +0.0031] | +0.3% |
| pitch_summary | 2 | 0.116 | 15% | +0.0017 [+0.0012, +0.0022] | +0.2% |
| future_past | 260 | 0.538 | 70% | +0.0017 [-0.0007, +0.0041] | +0.2% |
| formant_dyn | 9 | 0.061 | 8% | +0.0016 [+0.0011, +0.0021] | +0.2% |
| short_cochlea_contrast | 20 | 0.321 | 42% | +0.0015 [-0.0002, +0.0033] | +0.2% |
| fut_med_contrast_6 | 5 | 0.313 | 41% | +0.0013 [+0.0009, +0.0017] | +0.2% |
| hnr | 2 | 0.199 | 26% | +0.0012 [+0.0008, +0.0016] | +0.2% |
| cochlea_contrast | 20 | 0.343 | 45% | +0.0009 [+0.0001, +0.0018] | +0.1% |
| f0_dyn | 4 | 0.003 | 0% | +0.0008 [+0.0005, +0.0011] | +0.1% |
| contrast_med | 24 | 0.287 | 37% | +0.0007 [-0.0012, +0.0027] | +0.1% |
| multitau_contrast | 10 | 0.420 | 55% | +0.0006 [-0.0004, +0.0016] | +0.1% |
| hnr_dyn | 3 | 0.065 | 8% | +0.0004 [+0.0001, +0.0007] | +0.1% |
| slow_quad_modbank_dyn | 180 | 0.284 | 37% | +0.0004 [-0.0019, +0.0026] | +0.1% |
| mod_depth | 40 | 0.536 | 70% | +0.0003 [-0.0014, +0.0018] | +0.0% |
| spectral_flatness | 1 | 0.211 | 27% | +0.0001 [-0.0008, +0.0011] | +0.0% |
| cube_fut_contrast | 5 | 0.410 | 53% | +0.0001 [-0.0010, +0.0011] | +0.0% |
| gate_signals_dyn | 18 | 0.000 | 0% | +0.0001 [-0.0007, +0.0009] | +0.0% |
| gate_signals | 6 | 0.001 | 0% | +0.0001 [-0.0006, +0.0008] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.318 | 41% | -0.0001 [-0.0002, -0.0000] | -0.0% |
| med_adapt_contrast | 5 | 0.337 | 44% | -0.0002 [-0.0004, -0.0001] | -0.0% |
| fut_cochlea_contrast | 5 | 0.340 | 44% | -0.0003 [-0.0006, -0.0000] | -0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.104 | 13% | -0.0004 [-0.0006, -0.0002] | -0.1% |
| fut_minus_cur_contrast | 5 | 0.027 | 4% | -0.0004 [-0.0009, +0.0001] | -0.1% |
| phonetic | 5 | 0.359 | 47% | -0.0004 [-0.0028, +0.0017] | -0.1% |
| spectral_contrast | 24 | 0.309 | 40% | -0.0007 [-0.0025, +0.0010] | -0.1% |
| spectral_rolloff | 2 | 0.086 | 11% | -0.0008 [-0.0012, -0.0004] | -0.1% |
| slow_quad_modbank | 180 | 0.621 | 81% | -0.0009 [-0.0039, +0.0020] | -0.1% |
| quad_modbank_dyn | 540 | 0.330 | 43% | -0.0012 [-0.0042, +0.0016] | -0.2% |
| region_flat | 8 | 0.192 | 25% | -0.0013 [-0.0041, +0.0013] | -0.2% |
| quad_xrate_energy_dyn | 36 | 0.132 | 17% | -0.0020 [-0.0033, -0.0008] | -0.3% |
| formants | 24 | 0.372 | 48% | -0.0021 [-0.0040, -0.0004] | -0.3% |
| spectral_moments | 4 | 0.243 | 32% | -0.0027 [-0.0043, -0.0012] | -0.4% |
| stmf_mean_gabor | 17 | 0.107 | 14% | -0.0033 [-0.0048, -0.0019] | -0.4% |
| cochleagram | 60 | 0.458 | 60% | -0.0046 [-0.0070, -0.0022] | -0.6% |
| contrast_short | 24 | 0.257 | 33% | -0.0055 [-0.0077, -0.0032] | -0.7% |
| adapt_delta | 300 | 0.275 | 36% | -0.0286 [-0.0360, -0.0216] | -3.7% |
| tonic | 300 | 0.574 | 75% | -0.0445 [-0.0570, -0.0313] | -5.8% |

Sum of unique shares: -3.0% → shared (not attributable to any one unit): 103.0%

## Group level — all (full R² = 1.006%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| modulation | 132 | 0.660 | 66% | +0.0108 [+0.0074, +0.0141] | +1.1% |
| stmf | 340 | 0.952 | 95% | +0.0086 [+0.0019, +0.0159] | +0.9% |
| future_past | 260 | 0.698 | 69% | +0.0083 [+0.0043, +0.0124] | +0.8% |
| flux | 36 | 0.364 | 36% | +0.0055 [+0.0024, +0.0088] | +0.5% |
| topbot | 10 | 0.541 | 54% | +0.0055 [+0.0038, +0.0072] | +0.5% |
| quad_modbank | 180 | 0.784 | 78% | +0.0049 [+0.0009, +0.0090] | +0.5% |
| med_cochlea_contrast | 20 | 0.373 | 37% | +0.0033 [+0.0015, +0.0052] | +0.3% |
| quad_xrate_energy | 18 | 0.519 | 52% | +0.0028 [+0.0008, +0.0047] | +0.3% |
| pitch_salience | 11 | 0.118 | 12% | +0.0023 [+0.0013, +0.0032] | +0.2% |
| topbot_short | 10 | 0.523 | 52% | +0.0022 [+0.0014, +0.0030] | +0.2% |
| pitch_dyn | 6 | 0.039 | 4% | +0.0021 [+0.0007, +0.0036] | +0.2% |
| short_cochlea_contrast | 20 | 0.374 | 37% | +0.0020 [-0.0002, +0.0043] | +0.2% |
| cpp | 4 | 0.135 | 13% | +0.0017 [+0.0012, +0.0023] | +0.2% |
| contrast_med | 24 | 0.318 | 32% | +0.0017 [-0.0004, +0.0038] | +0.2% |
| cochlea_voiced_gated | 40 | 0.573 | 57% | +0.0014 [+0.0008, +0.0019] | +0.1% |
| quad_modbank_acausal | 180 | 0.290 | 29% | +0.0012 [-0.0052, +0.0077] | +0.1% |
| pitch_summary | 2 | 0.109 | 11% | +0.0011 [+0.0005, +0.0017] | +0.1% |
| hnr | 2 | 0.223 | 22% | +0.0010 [+0.0007, +0.0014] | +0.1% |
| fut_med_contrast_6 | 5 | 0.455 | 45% | +0.0010 [+0.0006, +0.0014] | +0.1% |
| mod_depth | 40 | 0.692 | 69% | +0.0007 [-0.0022, +0.0033] | +0.1% |
| cube_fut_contrast | 5 | 0.539 | 54% | +0.0007 [-0.0004, +0.0018] | +0.1% |
| formant_dyn | 9 | 0.071 | 7% | +0.0005 [+0.0002, +0.0009] | +0.1% |
| multitau_contrast | 10 | 0.499 | 50% | +0.0004 [-0.0003, +0.0011] | +0.0% |
| f0_dyn | 4 | 0.005 | 1% | +0.0004 [+0.0001, +0.0006] | +0.0% |
| spectral_flatness | 1 | 0.220 | 22% | +0.0003 [-0.0005, +0.0010] | +0.0% |
| cochlea_contrast | 20 | 0.384 | 38% | +0.0002 [-0.0016, +0.0019] | +0.0% |
| hnr_dyn | 3 | 0.077 | 8% | +0.0002 [-0.0001, +0.0005] | +0.0% |
| fut_minus_cur_contrast | 5 | 0.026 | 3% | +0.0001 [-0.0006, +0.0007] | +0.0% |
| gate_signals | 6 | 0.002 | 0% | +0.0000 [-0.0005, +0.0006] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.406 | 40% | -0.0002 [-0.0003, -0.0001] | -0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.081 | 8% | -0.0002 [-0.0003, -0.0000] | -0.0% |
| gate_signals_dyn | 18 | 0.000 | 0% | -0.0003 [-0.0010, +0.0005] | -0.0% |
| med_adapt_contrast | 5 | 0.394 | 39% | -0.0003 [-0.0005, -0.0001] | -0.0% |
| interact_v3e | 405 | 0.706 | 70% | -0.0004 [-0.0076, +0.0074] | -0.0% |
| fut_cochlea_contrast | 5 | 0.435 | 43% | -0.0005 [-0.0009, -0.0000] | -0.0% |
| spectral_rolloff | 2 | 0.087 | 9% | -0.0005 [-0.0007, -0.0003] | -0.0% |
| spectral_moments | 4 | 0.299 | 30% | -0.0012 [-0.0022, -0.0002] | -0.1% |
| phonetic | 5 | 0.464 | 46% | -0.0016 [-0.0043, +0.0011] | -0.2% |
| formants | 24 | 0.434 | 43% | -0.0018 [-0.0040, +0.0003] | -0.2% |
| quad_modbank_dyn | 540 | 0.292 | 29% | -0.0019 [-0.0071, +0.0033] | -0.2% |
| quad_xrate_energy_dyn | 36 | 0.107 | 11% | -0.0022 [-0.0039, -0.0007] | -0.2% |
| stmf_mean_gabor | 17 | 0.164 | 16% | -0.0025 [-0.0039, -0.0011] | -0.2% |
| spectral_contrast | 24 | 0.314 | 31% | -0.0026 [-0.0049, -0.0004] | -0.3% |
| scattering | 287 | 0.712 | 71% | -0.0036 [-0.0107, +0.0034] | -0.4% |
| region_flat | 8 | 0.237 | 24% | -0.0040 [-0.0085, +0.0001] | -0.4% |
| slow_quad_modbank_dyn | 180 | 0.269 | 27% | -0.0049 [-0.0094, -0.0005] | -0.5% |
| slow_quad_modbank | 180 | 0.830 | 82% | -0.0052 [-0.0103, -0.0003] | -0.5% |
| contrast_short | 24 | 0.280 | 28% | -0.0088 [-0.0127, -0.0048] | -0.9% |
| cochleagram | 60 | 0.580 | 58% | -0.0098 [-0.0149, -0.0049] | -1.0% |
| adapt_delta | 300 | 0.256 | 25% | -0.0360 [-0.0503, -0.0242] | -3.6% |
| tonic | 300 | 0.827 | 82% | -0.0469 [-0.0686, -0.0243] | -4.7% |

Sum of unique shares: -6.4% → shared (not attributable to any one unit): 106.4%

