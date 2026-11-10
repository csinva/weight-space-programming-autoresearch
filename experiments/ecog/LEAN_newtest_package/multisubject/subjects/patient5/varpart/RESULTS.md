# LEAN-v20 variance partitioning — patient5, held-out session 5e416143

275 test chunks (the whole held-out session, never used in training); 500 training chunks from the other sessions; 188 electrodes, grid = first 124. Harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model

- grid: mean r = 0.0637, R² = 0.628% of variance
- all: mean r = 0.0795, R² = 0.992% of variance

## Family level — grid (full R² = 0.628%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.628 | 100% | +0.0125 [+0.0034, +0.0216] | +2.0% |
| Spectral/cochleagram contrast | 192 | 0.464 | 74% | +0.0068 [+0.0028, +0.0109] | +1.1% |
| Pitch, voicing, voice quality | 122 | 0.338 | 54% | +0.0001 [-0.0027, +0.0029] | +0.0% |
| Wavelet scattering | 287 | 0.482 | 77% | -0.0028 [-0.0088, +0.0028] | -0.4% |
| Interpretable interactions (v3e) | 405 | 0.480 | 76% | -0.0033 [-0.0081, +0.0013] | -0.5% |
| Spectral shape, flux, formants | 83 | 0.405 | 64% | -0.0046 [-0.0071, -0.0022] | -0.7% |
| Cochleagram core + slow context + lookahead | 960 | 0.583 | 93% | -0.0128 [-0.0234, -0.0025] | -2.0% |

Sum of unique shares: -0.6% → shared (not attributable to any one unit): 100.6%

## Family level — all (full R² = 0.992%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.975 | 98% | +0.0184 [+0.0036, +0.0331] | +1.9% |
| Spectral/cochleagram contrast | 192 | 0.752 | 76% | +0.0139 [+0.0068, +0.0214] | +1.4% |
| Wavelet scattering | 287 | 0.781 | 79% | -0.0004 [-0.0091, +0.0081] | -0.0% |
| Pitch, voicing, voice quality | 122 | 0.567 | 57% | -0.0034 [-0.0081, +0.0012] | -0.3% |
| Spectral shape, flux, formants | 83 | 0.607 | 61% | -0.0089 [-0.0137, -0.0046] | -0.9% |
| Cochleagram core + slow context + lookahead | 960 | 0.929 | 94% | -0.0091 [-0.0328, +0.0149] | -0.9% |
| Interpretable interactions (v3e) | 405 | 0.726 | 73% | -0.0120 [-0.0211, -0.0038] | -1.2% |

Sum of unique shares: -0.2% → shared (not attributable to any one unit): 100.2%

## Section level — grid (full R² = 0.628%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| M. Quadrature modulation bank | 954 | 0.607 | 97% | +0.0084 [+0.0044, +0.0130] | +1.3% |
| L. Spectral/cochleagram contrast | 192 | 0.464 | 74% | +0.0068 [+0.0028, +0.0109] | +1.1% |
| O. Slow quadrature bank | 360 | 0.555 | 88% | +0.0065 [+0.0028, +0.0103] | +1.0% |
| N. Voicing gates | 84 | 0.357 | 57% | +0.0020 [+0.0003, +0.0040] | +0.3% |
| B. Pitch | 19 | 0.115 | 18% | +0.0014 [-0.0004, +0.0034] | +0.2% |
| I. Pitch/voicing dynamics | 7 | 0.007 | 1% | +0.0001 [+0.0000, +0.0003] | +0.0% |
| D. Spectral flux | 36 | 0.296 | 47% | -0.0001 [-0.0018, +0.0015] | -0.0% |
| E. 1-D temporal modulation | 132 | 0.388 | 62% | -0.0009 [-0.0046, +0.0029] | -0.2% |
| K. Voice quality (CPP, region flatness) | 12 | 0.115 | 18% | -0.0027 [-0.0051, -0.0004] | -0.4% |
| P. Wavelet scattering | 287 | 0.482 | 77% | -0.0028 [-0.0088, +0.0028] | -0.4% |
| J. Interactions (v3e) | 405 | 0.480 | 76% | -0.0033 [-0.0081, +0.0013] | -0.5% |
| C. Spectral shape | 14 | 0.227 | 36% | -0.0051 [-0.0066, -0.0038] | -0.8% |
| G. Formants | 33 | 0.339 | 54% | -0.0067 [-0.0087, -0.0050] | -1.1% |
| H. Slow context + lookahead | 600 | 0.574 | 91% | -0.0090 [-0.0200, +0.0017] | -1.4% |
| A. Cochleagram core | 360 | 0.372 | 59% | -0.0092 [-0.0153, -0.0037] | -1.5% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.499 | 79% | -0.0175 [-0.0232, -0.0122] | -2.8% |

Sum of unique shares: -5.1% → shared (not attributable to any one unit): 105.1%

## Section level — all (full R² = 0.992%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| L. Spectral/cochleagram contrast | 192 | 0.752 | 76% | +0.0139 [+0.0068, +0.0214] | +1.4% |
| M. Quadrature modulation bank | 954 | 0.883 | 89% | +0.0126 [+0.0047, +0.0210] | +1.3% |
| O. Slow quadrature bank | 360 | 0.842 | 85% | +0.0074 [+0.0023, +0.0127] | +0.7% |
| N. Voicing gates | 84 | 0.582 | 59% | +0.0020 [+0.0001, +0.0041] | +0.2% |
| E. 1-D temporal modulation | 132 | 0.582 | 59% | +0.0018 [-0.0038, +0.0077] | +0.2% |
| B. Pitch | 19 | 0.192 | 19% | +0.0017 [+0.0001, +0.0035] | +0.2% |
| A. Cochleagram core | 360 | 0.634 | 64% | +0.0010 [-0.0107, +0.0126] | +0.1% |
| I. Pitch/voicing dynamics | 7 | 0.012 | 1% | +0.0001 [-0.0002, +0.0003] | +0.0% |
| P. Wavelet scattering | 287 | 0.781 | 79% | -0.0004 [-0.0091, +0.0081] | -0.0% |
| K. Voice quality (CPP, region flatness) | 12 | 0.167 | 17% | -0.0022 [-0.0063, +0.0018] | -0.2% |
| D. Spectral flux | 36 | 0.388 | 39% | -0.0029 [-0.0059, +0.0000] | -0.3% |
| H. Slow context + lookahead | 600 | 0.910 | 92% | -0.0029 [-0.0269, +0.0209] | -0.3% |
| C. Spectral shape | 14 | 0.394 | 40% | -0.0040 [-0.0059, -0.0022] | -0.4% |
| G. Formants | 33 | 0.529 | 53% | -0.0063 [-0.0091, -0.0035] | -0.6% |
| J. Interactions (v3e) | 405 | 0.726 | 73% | -0.0120 [-0.0211, -0.0038] | -1.2% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.768 | 77% | -0.0180 [-0.0268, -0.0101] | -1.8% |

Sum of unique shares: -0.8% → shared (not attributable to any one unit): 100.8%

## Group level — grid (full R² = 0.628%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| future_past | 260 | 0.360 | 57% | +0.0084 [+0.0053, +0.0116] | +1.3% |
| mod_depth | 40 | 0.356 | 57% | +0.0069 [+0.0036, +0.0104] | +1.1% |
| topbot | 10 | 0.274 | 44% | +0.0020 [+0.0006, +0.0038] | +0.3% |
| contrast_med | 24 | 0.243 | 39% | +0.0017 [-0.0004, +0.0040] | +0.3% |
| slow_quad_modbank_dyn | 180 | 0.192 | 31% | +0.0015 [-0.0016, +0.0047] | +0.2% |
| gate_signals_dyn | 18 | 0.001 | 0% | +0.0014 [-0.0002, +0.0032] | +0.2% |
| med_cochlea_contrast | 20 | 0.265 | 42% | +0.0013 [+0.0005, +0.0022] | +0.2% |
| quad_modbank | 180 | 0.459 | 73% | +0.0012 [-0.0010, +0.0034] | +0.2% |
| cochlea_contrast | 20 | 0.278 | 44% | +0.0009 [+0.0003, +0.0016] | +0.1% |
| quad_xrate_energy | 18 | 0.284 | 45% | +0.0009 [-0.0001, +0.0019] | +0.1% |
| cube_fut_contrast | 5 | 0.237 | 38% | +0.0009 [+0.0004, +0.0014] | +0.1% |
| topbot_short | 10 | 0.241 | 38% | +0.0006 [+0.0004, +0.0009] | +0.1% |
| fut_med_contrast_6 | 5 | 0.139 | 22% | +0.0006 [+0.0004, +0.0008] | +0.1% |
| cochleagram | 60 | 0.270 | 43% | +0.0006 [-0.0009, +0.0020] | +0.1% |
| med_adapt_contrast | 5 | 0.177 | 28% | +0.0004 [+0.0001, +0.0008] | +0.1% |
| gate_signals | 6 | 0.002 | 0% | +0.0003 [-0.0003, +0.0008] | +0.0% |
| multitau_contrast | 10 | 0.236 | 38% | +0.0002 [-0.0002, +0.0007] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.146 | 23% | +0.0002 [+0.0001, +0.0003] | +0.0% |
| formant_dyn | 9 | 0.052 | 8% | +0.0001 [-0.0001, +0.0004] | +0.0% |
| f0_dyn | 4 | 0.005 | 1% | +0.0001 [+0.0000, +0.0002] | +0.0% |
| hnr_dyn | 3 | 0.002 | 0% | +0.0000 [-0.0001, +0.0001] | +0.0% |
| fut_minus_cur_contrast | 5 | 0.014 | 2% | +0.0000 [-0.0001, +0.0001] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.051 | 8% | +0.0000 [-0.0003, +0.0003] | +0.0% |
| hnr | 2 | 0.088 | 14% | +0.0000 [-0.0001, +0.0001] | +0.0% |
| spectral_rolloff | 2 | 0.029 | 5% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| pitch_summary | 2 | 0.091 | 14% | -0.0000 [-0.0002, +0.0001] | -0.0% |
| spectral_flatness | 1 | 0.114 | 18% | -0.0001 [-0.0005, +0.0003] | -0.0% |
| flux | 36 | 0.296 | 47% | -0.0001 [-0.0018, +0.0015] | -0.0% |
| slow_quad_modbank | 180 | 0.526 | 84% | -0.0002 [-0.0035, +0.0030] | -0.0% |
| fut_cochlea_contrast | 5 | 0.166 | 26% | -0.0003 [-0.0005, -0.0001] | -0.1% |
| stmf_mean_gabor | 17 | 0.073 | 12% | -0.0004 [-0.0015, +0.0008] | -0.1% |
| contrast_short | 24 | 0.162 | 26% | -0.0006 [-0.0023, +0.0012] | -0.1% |
| short_cochlea_contrast | 20 | 0.240 | 38% | -0.0008 [-0.0020, +0.0004] | -0.1% |
| cochlea_voiced_gated | 40 | 0.267 | 42% | -0.0008 [-0.0018, +0.0001] | -0.1% |
| modulation | 132 | 0.388 | 62% | -0.0009 [-0.0046, +0.0029] | -0.2% |
| pitch_dyn | 6 | 0.055 | 9% | -0.0015 [-0.0022, -0.0008] | -0.2% |
| cpp | 4 | 0.112 | 18% | -0.0016 [-0.0027, -0.0006] | -0.3% |
| quad_modbank_dyn | 540 | 0.281 | 45% | -0.0018 [-0.0040, +0.0005] | -0.3% |
| pitch_salience | 11 | 0.092 | 15% | -0.0022 [-0.0036, -0.0008] | -0.3% |
| quad_xrate_energy_dyn | 36 | 0.085 | 14% | -0.0025 [-0.0038, -0.0013] | -0.4% |
| adapt_delta | 300 | 0.228 | 36% | -0.0027 [-0.0083, +0.0025] | -0.4% |
| spectral_contrast | 24 | 0.243 | 39% | -0.0027 [-0.0044, -0.0010] | -0.4% |
| scattering | 287 | 0.482 | 77% | -0.0028 [-0.0088, +0.0028] | -0.4% |
| spectral_moments | 4 | 0.112 | 18% | -0.0028 [-0.0039, -0.0019] | -0.5% |
| region_flat | 8 | 0.026 | 4% | -0.0029 [-0.0052, -0.0006] | -0.5% |
| interact_v3e | 405 | 0.480 | 76% | -0.0033 [-0.0081, +0.0013] | -0.5% |
| quad_modbank_acausal | 180 | 0.196 | 31% | -0.0033 [-0.0074, +0.0007] | -0.5% |
| phonetic | 5 | 0.221 | 35% | -0.0043 [-0.0055, -0.0032] | -0.7% |
| formants | 24 | 0.333 | 53% | -0.0090 [-0.0113, -0.0070] | -1.4% |
| stmf | 340 | 0.487 | 78% | -0.0125 [-0.0181, -0.0076] | -2.0% |
| tonic | 300 | 0.418 | 67% | -0.0126 [-0.0227, -0.0018] | -2.0% |

Sum of unique shares: -6.8% → shared (not attributable to any one unit): 106.8%

## Group level — all (full R² = 0.992%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| future_past | 260 | 0.545 | 55% | +0.0116 [+0.0071, +0.0168] | +1.2% |
| mod_depth | 40 | 0.493 | 50% | +0.0078 [+0.0033, +0.0124] | +0.8% |
| quad_modbank | 180 | 0.679 | 68% | +0.0051 [+0.0013, +0.0089] | +0.5% |
| contrast_short | 24 | 0.252 | 25% | +0.0044 [+0.0012, +0.0078] | +0.4% |
| adapt_delta | 300 | 0.373 | 38% | +0.0041 [-0.0074, +0.0157] | +0.4% |
| topbot | 10 | 0.464 | 47% | +0.0038 [+0.0020, +0.0059] | +0.4% |
| contrast_med | 24 | 0.371 | 37% | +0.0029 [+0.0007, +0.0053] | +0.3% |
| quad_xrate_energy | 18 | 0.422 | 43% | +0.0027 [+0.0009, +0.0044] | +0.3% |
| fut_minus_cur_contrast | 5 | 0.022 | 2% | +0.0021 [+0.0011, +0.0031] | +0.2% |
| cube_fut_contrast | 5 | 0.414 | 42% | +0.0020 [+0.0011, +0.0029] | +0.2% |
| spectral_contrast | 24 | 0.349 | 35% | +0.0019 [-0.0010, +0.0047] | +0.2% |
| med_cochlea_contrast | 20 | 0.399 | 40% | +0.0018 [+0.0007, +0.0031] | +0.2% |
| modulation | 132 | 0.582 | 59% | +0.0018 [-0.0038, +0.0077] | +0.2% |
| multitau_contrast | 10 | 0.335 | 34% | +0.0015 [+0.0008, +0.0023] | +0.2% |
| med_adapt_contrast | 5 | 0.251 | 25% | +0.0014 [+0.0007, +0.0021] | +0.1% |
| spectral_rolloff | 2 | 0.047 | 5% | +0.0013 [+0.0007, +0.0020] | +0.1% |
| pitch_summary | 2 | 0.161 | 16% | +0.0013 [+0.0006, +0.0020] | +0.1% |
| quad_modbank_dyn | 540 | 0.402 | 41% | +0.0013 [-0.0024, +0.0049] | +0.1% |
| hnr_dyn | 3 | 0.004 | 0% | +0.0012 [+0.0006, +0.0019] | +0.1% |
| cochlea_contrast | 20 | 0.423 | 43% | +0.0012 [+0.0002, +0.0023] | +0.1% |
| short_cochlea_contrast | 20 | 0.357 | 36% | +0.0011 [-0.0003, +0.0026] | +0.1% |
| fut_med_contrast_6 | 5 | 0.231 | 23% | +0.0011 [+0.0005, +0.0017] | +0.1% |
| gate_signals_dyn | 18 | 0.001 | 0% | +0.0011 [-0.0006, +0.0028] | +0.1% |
| cochleagram | 60 | 0.472 | 48% | +0.0010 [-0.0024, +0.0045] | +0.1% |
| spectral_flatness | 1 | 0.189 | 19% | +0.0010 [+0.0000, +0.0020] | +0.1% |
| stmf_mean_gabor | 17 | 0.111 | 11% | +0.0008 [-0.0011, +0.0026] | +0.1% |
| cpp | 4 | 0.158 | 16% | +0.0007 [-0.0010, +0.0021] | +0.1% |
| slow_quad_modbank_dyn | 180 | 0.298 | 30% | +0.0006 [-0.0039, +0.0049] | +0.1% |
| slow_quad_modbank | 180 | 0.811 | 82% | +0.0006 [-0.0037, +0.0048] | +0.1% |
| quad_modbank_acausal | 180 | 0.286 | 29% | +0.0005 [-0.0056, +0.0068] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.247 | 25% | +0.0004 [+0.0002, +0.0006] | +0.0% |
| cochlea_voiced_gated | 40 | 0.460 | 46% | +0.0004 [-0.0004, +0.0011] | +0.0% |
| topbot_short | 10 | 0.412 | 42% | +0.0004 [-0.0001, +0.0008] | +0.0% |
| formant_dyn | 9 | 0.072 | 7% | +0.0000 [-0.0002, +0.0003] | +0.0% |
| gate_signals | 6 | 0.002 | 0% | -0.0000 [-0.0010, +0.0009] | -0.0% |
| f0_dyn | 4 | 0.009 | 1% | -0.0000 [-0.0001, +0.0001] | -0.0% |
| hnr | 2 | 0.143 | 14% | -0.0001 [-0.0003, +0.0000] | -0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.096 | 10% | -0.0003 [-0.0007, +0.0001] | -0.0% |
| scattering | 287 | 0.781 | 79% | -0.0004 [-0.0091, +0.0081] | -0.0% |
| quad_xrate_energy_dyn | 36 | 0.122 | 12% | -0.0004 [-0.0019, +0.0010] | -0.0% |
| fut_cochlea_contrast | 5 | 0.279 | 28% | -0.0005 [-0.0009, -0.0002] | -0.1% |
| pitch_salience | 11 | 0.163 | 16% | -0.0011 [-0.0023, +0.0001] | -0.1% |
| pitch_dyn | 6 | 0.087 | 9% | -0.0013 [-0.0022, -0.0005] | -0.1% |
| spectral_moments | 4 | 0.183 | 18% | -0.0024 [-0.0036, -0.0014] | -0.2% |
| flux | 36 | 0.388 | 39% | -0.0029 [-0.0059, +0.0000] | -0.3% |
| phonetic | 5 | 0.394 | 40% | -0.0036 [-0.0052, -0.0022] | -0.4% |
| region_flat | 8 | 0.035 | 4% | -0.0044 [-0.0087, -0.0001] | -0.4% |
| formants | 24 | 0.514 | 52% | -0.0078 [-0.0108, -0.0050] | -0.8% |
| interact_v3e | 405 | 0.726 | 73% | -0.0120 [-0.0211, -0.0038] | -1.2% |
| tonic | 300 | 0.712 | 72% | -0.0169 [-0.0392, +0.0060] | -1.7% |
| stmf | 340 | 0.737 | 74% | -0.0175 [-0.0264, -0.0094] | -1.8% |

Sum of unique shares: -0.1% → shared (not attributable to any one unit): 100.1%

