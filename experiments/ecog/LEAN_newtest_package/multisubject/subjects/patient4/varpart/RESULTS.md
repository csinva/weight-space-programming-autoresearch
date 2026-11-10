# LEAN-v20 variance partitioning — patient4, held-out session 2eeb170d

107 test chunks (the whole held-out session, never used in training); 500 training chunks from the other sessions; 155 electrodes, grid = first 96. Harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model

- grid: mean r = 0.0578, R² = 0.475% of variance
- all: mean r = 0.0613, R² = 0.521% of variance

## Family level — grid (full R² = 0.475%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.479 | 101% | +0.0260 [+0.0147, +0.0388] | +5.5% |
| Wavelet scattering | 287 | 0.343 | 72% | +0.0043 [+0.0013, +0.0076] | +0.9% |
| Spectral/cochleagram contrast | 192 | 0.380 | 80% | +0.0016 [-0.0022, +0.0057] | +0.3% |
| Pitch, voicing, voice quality | 122 | 0.292 | 62% | -0.0000 [-0.0024, +0.0025] | -0.0% |
| Interpretable interactions (v3e) | 405 | 0.372 | 78% | -0.0053 [-0.0134, +0.0017] | -1.1% |
| Spectral shape, flux, formants | 83 | 0.344 | 72% | -0.0074 [-0.0112, -0.0041] | -1.6% |
| Cochleagram core + slow context + lookahead | 960 | 0.424 | 89% | -0.0077 [-0.0191, +0.0041] | -1.6% |

Sum of unique shares: 2.4% → shared (not attributable to any one unit): 97.6%

## Family level — all (full R² = 0.521%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.513 | 98% | +0.0385 [+0.0225, +0.0568] | +7.4% |
| Wavelet scattering | 287 | 0.302 | 58% | +0.0082 [+0.0033, +0.0141] | +1.6% |
| Spectral/cochleagram contrast | 192 | 0.369 | 71% | +0.0055 [-0.0000, +0.0115] | +1.1% |
| Cochleagram core + slow context + lookahead | 960 | 0.437 | 84% | +0.0008 [-0.0158, +0.0214] | +0.2% |
| Pitch, voicing, voice quality | 122 | 0.258 | 50% | -0.0000 [-0.0031, +0.0030] | -0.0% |
| Spectral shape, flux, formants | 83 | 0.357 | 68% | -0.0040 [-0.0066, -0.0018] | -0.8% |
| Interpretable interactions (v3e) | 405 | 0.366 | 70% | -0.0044 [-0.0130, +0.0030] | -0.8% |

Sum of unique shares: 8.6% → shared (not attributable to any one unit): 91.4%

## Section level — grid (full R² = 0.475%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| F. Spectro-temporal modulation (STMF) | 357 | 0.391 | 82% | +0.0121 [+0.0066, +0.0181] | +2.5% |
| E. 1-D temporal modulation | 132 | 0.371 | 78% | +0.0055 [+0.0011, +0.0105] | +1.2% |
| P. Wavelet scattering | 287 | 0.343 | 72% | +0.0043 [+0.0013, +0.0076] | +0.9% |
| O. Slow quadrature bank | 360 | 0.424 | 89% | +0.0037 [-0.0008, +0.0085] | +0.8% |
| L. Spectral/cochleagram contrast | 192 | 0.380 | 80% | +0.0016 [-0.0022, +0.0057] | +0.3% |
| A. Cochleagram core | 360 | 0.293 | 62% | +0.0014 [-0.0022, +0.0050] | +0.3% |
| I. Pitch/voicing dynamics | 7 | 0.025 | 5% | +0.0009 [+0.0003, +0.0015] | +0.2% |
| G. Formants | 33 | 0.280 | 59% | -0.0001 [-0.0023, +0.0023] | -0.0% |
| N. Voicing gates | 84 | 0.280 | 59% | -0.0001 [-0.0012, +0.0010] | -0.0% |
| B. Pitch | 19 | 0.118 | 25% | -0.0003 [-0.0013, +0.0008] | -0.1% |
| K. Voice quality (CPP, region flatness) | 12 | 0.119 | 25% | -0.0003 [-0.0023, +0.0019] | -0.1% |
| D. Spectral flux | 36 | 0.288 | 61% | -0.0032 [-0.0053, -0.0013] | -0.7% |
| C. Spectral shape | 14 | 0.289 | 61% | -0.0038 [-0.0072, -0.0010] | -0.8% |
| J. Interactions (v3e) | 405 | 0.372 | 78% | -0.0053 [-0.0134, +0.0017] | -1.1% |
| M. Quadrature modulation bank | 954 | 0.454 | 96% | -0.0071 [-0.0147, -0.0007] | -1.5% |
| H. Slow context + lookahead | 600 | 0.410 | 86% | -0.0106 [-0.0211, +0.0006] | -2.2% |

Sum of unique shares: -0.3% → shared (not attributable to any one unit): 100.3%

## Section level — all (full R² = 0.521%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| F. Spectro-temporal modulation (STMF) | 357 | 0.378 | 73% | +0.0144 [+0.0068, +0.0231] | +2.8% |
| O. Slow quadrature bank | 360 | 0.440 | 84% | +0.0084 [+0.0017, +0.0158] | +1.6% |
| P. Wavelet scattering | 287 | 0.302 | 58% | +0.0082 [+0.0033, +0.0141] | +1.6% |
| E. 1-D temporal modulation | 132 | 0.363 | 70% | +0.0075 [+0.0021, +0.0138] | +1.4% |
| L. Spectral/cochleagram contrast | 192 | 0.369 | 71% | +0.0055 [-0.0000, +0.0115] | +1.1% |
| A. Cochleagram core | 360 | 0.261 | 50% | +0.0046 [+0.0003, +0.0093] | +0.9% |
| I. Pitch/voicing dynamics | 7 | 0.020 | 4% | +0.0005 [+0.0001, +0.0009] | +0.1% |
| K. Voice quality (CPP, region flatness) | 12 | 0.133 | 26% | -0.0001 [-0.0024, +0.0023] | -0.0% |
| N. Voicing gates | 84 | 0.233 | 45% | -0.0001 [-0.0010, +0.0009] | -0.0% |
| B. Pitch | 19 | 0.112 | 22% | -0.0003 [-0.0018, +0.0013] | -0.1% |
| G. Formants | 33 | 0.266 | 51% | -0.0003 [-0.0022, +0.0015] | -0.1% |
| M. Quadrature modulation bank | 954 | 0.468 | 90% | -0.0009 [-0.0074, +0.0047] | -0.2% |
| D. Spectral flux | 36 | 0.286 | 55% | -0.0016 [-0.0032, -0.0001] | -0.3% |
| C. Spectral shape | 14 | 0.245 | 47% | -0.0023 [-0.0045, -0.0004] | -0.4% |
| H. Slow context + lookahead | 600 | 0.418 | 80% | -0.0032 [-0.0189, +0.0152] | -0.6% |
| J. Interactions (v3e) | 405 | 0.366 | 70% | -0.0044 [-0.0130, +0.0030] | -0.8% |

Sum of unique shares: 6.9% → shared (not attributable to any one unit): 93.1%

## Group level — grid (full R² = 0.475%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| stmf | 340 | 0.387 | 82% | +0.0105 [+0.0056, +0.0158] | +2.2% |
| future_past | 260 | 0.308 | 65% | +0.0055 [+0.0018, +0.0099] | +1.2% |
| modulation | 132 | 0.371 | 78% | +0.0055 [+0.0011, +0.0105] | +1.2% |
| scattering | 287 | 0.343 | 72% | +0.0043 [+0.0013, +0.0076] | +0.9% |
| quad_xrate_energy_dyn | 36 | 0.137 | 29% | +0.0020 [+0.0006, +0.0037] | +0.4% |
| slow_quad_modbank | 180 | 0.415 | 87% | +0.0018 [-0.0023, +0.0062] | +0.4% |
| short_cochlea_contrast | 20 | 0.176 | 37% | +0.0017 [+0.0007, +0.0028] | +0.4% |
| topbot | 10 | 0.271 | 57% | +0.0011 [+0.0005, +0.0020] | +0.2% |
| pitch_salience | 11 | 0.059 | 12% | +0.0011 [-0.0001, +0.0023] | +0.2% |
| multitau_contrast | 10 | 0.269 | 57% | +0.0009 [-0.0003, +0.0021] | +0.2% |
| cpp | 4 | 0.114 | 24% | +0.0007 [-0.0007, +0.0021] | +0.1% |
| cube_fut_contrast | 5 | 0.230 | 48% | +0.0003 [-0.0008, +0.0014] | +0.1% |
| contrast_short | 24 | 0.165 | 35% | +0.0001 [-0.0015, +0.0020] | +0.0% |
| spectral_flatness | 1 | 0.110 | 23% | +0.0000 [-0.0001, +0.0002] | +0.0% |
| contrast_med | 24 | 0.200 | 42% | +0.0000 [-0.0021, +0.0021] | +0.0% |
| hnr_dyn | 3 | 0.009 | 2% | +0.0000 [-0.0002, +0.0003] | +0.0% |
| f0_dyn | 4 | 0.022 | 5% | +0.0000 [-0.0001, +0.0001] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.143 | 30% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| med_cochlea_contrast | 20 | 0.188 | 39% | -0.0000 [-0.0013, +0.0012] | -0.0% |
| gate_signals | 6 | 0.001 | 0% | -0.0001 [-0.0007, +0.0004] | -0.0% |
| spectral_moments | 4 | 0.180 | 38% | -0.0001 [-0.0009, +0.0007] | -0.0% |
| stmf_mean_gabor | 17 | 0.120 | 25% | -0.0002 [-0.0019, +0.0014] | -0.0% |
| adapt_delta | 300 | 0.180 | 38% | -0.0003 [-0.0042, +0.0034] | -0.1% |
| gate_signals_dyn | 18 | 0.001 | 0% | -0.0004 [-0.0010, +0.0002] | -0.1% |
| pitch_summary | 2 | 0.102 | 21% | -0.0005 [-0.0011, +0.0000] | -0.1% |
| fut_med_contrast_6 | 5 | 0.190 | 40% | -0.0005 [-0.0017, +0.0005] | -0.1% |
| topbot_short | 10 | 0.263 | 55% | -0.0007 [-0.0019, +0.0006] | -0.1% |
| quad_xrate_energy | 18 | 0.310 | 65% | -0.0007 [-0.0018, +0.0004] | -0.2% |
| quad_modbank | 180 | 0.389 | 82% | -0.0008 [-0.0043, +0.0023] | -0.2% |
| formant_dyn | 9 | 0.063 | 13% | -0.0009 [-0.0028, +0.0009] | -0.2% |
| fut_minus_cur_contrast | 5 | 0.010 | 2% | -0.0010 [-0.0016, -0.0004] | -0.2% |
| slow_quad_modbank_dyn | 180 | 0.195 | 41% | -0.0010 [-0.0048, +0.0023] | -0.2% |
| cochlea_voiced_gated | 40 | 0.279 | 59% | -0.0011 [-0.0030, +0.0007] | -0.2% |
| region_flat | 8 | 0.059 | 12% | -0.0011 [-0.0027, +0.0004] | -0.2% |
| pitch_dyn | 6 | 0.065 | 14% | -0.0012 [-0.0031, +0.0005] | -0.3% |
| cochlea_contrast | 20 | 0.193 | 41% | -0.0013 [-0.0033, +0.0009] | -0.3% |
| med_adapt_contrast | 5 | 0.209 | 44% | -0.0015 [-0.0035, +0.0004] | -0.3% |
| spectral_rolloff | 2 | 0.136 | 29% | -0.0016 [-0.0034, +0.0002] | -0.3% |
| fut_cochlea_contrast | 5 | 0.200 | 42% | -0.0016 [-0.0035, +0.0001] | -0.3% |
| fut_cochlea_contrast_med | 5 | 0.198 | 42% | -0.0016 [-0.0034, +0.0001] | -0.3% |
| quad_modbank_dyn | 540 | 0.215 | 45% | -0.0016 [-0.0057, +0.0019] | -0.3% |
| hnr | 2 | 0.129 | 27% | -0.0017 [-0.0037, -0.0000] | -0.4% |
| quad_modbank_acausal | 180 | 0.181 | 38% | -0.0018 [-0.0063, +0.0026] | -0.4% |
| cochleagram | 60 | 0.279 | 59% | -0.0019 [-0.0044, +0.0003] | -0.4% |
| formants | 24 | 0.276 | 58% | -0.0028 [-0.0051, -0.0006] | -0.6% |
| spectral_contrast | 24 | 0.239 | 50% | -0.0029 [-0.0054, -0.0008] | -0.6% |
| phonetic | 5 | 0.265 | 56% | -0.0031 [-0.0054, -0.0009] | -0.6% |
| flux | 36 | 0.288 | 61% | -0.0032 [-0.0053, -0.0013] | -0.7% |
| interact_v3e | 405 | 0.372 | 78% | -0.0053 [-0.0134, +0.0017] | -1.1% |
| mod_depth | 40 | 0.272 | 57% | -0.0054 [-0.0095, -0.0018] | -1.1% |
| tonic | 300 | 0.331 | 70% | -0.0130 [-0.0229, -0.0017] | -2.7% |

Sum of unique shares: -5.3% → shared (not attributable to any one unit): 105.3%

## Group level — all (full R² = 0.521%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| stmf | 340 | 0.376 | 72% | +0.0147 [+0.0074, +0.0231] | +2.8% |
| future_past | 260 | 0.285 | 55% | +0.0113 [+0.0060, +0.0176] | +2.2% |
| scattering | 287 | 0.302 | 58% | +0.0082 [+0.0033, +0.0141] | +1.6% |
| modulation | 132 | 0.363 | 70% | +0.0075 [+0.0021, +0.0138] | +1.4% |
| slow_quad_modbank | 180 | 0.424 | 81% | +0.0037 [-0.0014, +0.0096] | +0.7% |
| adapt_delta | 300 | 0.120 | 23% | +0.0021 [-0.0018, +0.0060] | +0.4% |
| quad_xrate_energy_dyn | 36 | 0.091 | 17% | +0.0016 [+0.0007, +0.0025] | +0.3% |
| short_cochlea_contrast | 20 | 0.219 | 42% | +0.0011 [+0.0001, +0.0022] | +0.2% |
| cpp | 4 | 0.108 | 21% | +0.0010 [-0.0002, +0.0023] | +0.2% |
| topbot | 10 | 0.222 | 43% | +0.0010 [+0.0005, +0.0017] | +0.2% |
| pitch_salience | 11 | 0.042 | 8% | +0.0006 [-0.0011, +0.0024] | +0.1% |
| contrast_med | 24 | 0.226 | 43% | +0.0006 [-0.0008, +0.0021] | +0.1% |
| spectral_moments | 4 | 0.144 | 28% | +0.0006 [-0.0000, +0.0013] | +0.1% |
| multitau_contrast | 10 | 0.239 | 46% | +0.0004 [-0.0006, +0.0014] | +0.1% |
| contrast_short | 24 | 0.148 | 28% | +0.0002 [-0.0013, +0.0018] | +0.0% |
| med_cochlea_contrast | 20 | 0.222 | 43% | +0.0001 [-0.0007, +0.0009] | +0.0% |
| spectral_flatness | 1 | 0.090 | 17% | +0.0000 [-0.0001, +0.0002] | +0.0% |
| hnr_dyn | 3 | 0.007 | 1% | +0.0000 [-0.0001, +0.0002] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.092 | 18% | -0.0000 [-0.0002, +0.0001] | -0.0% |
| f0_dyn | 4 | 0.018 | 3% | -0.0001 [-0.0001, -0.0000] | -0.0% |
| cube_fut_contrast | 5 | 0.203 | 39% | -0.0001 [-0.0009, +0.0007] | -0.0% |
| gate_signals | 6 | 0.000 | 0% | -0.0001 [-0.0009, +0.0007] | -0.0% |
| topbot_short | 10 | 0.214 | 41% | -0.0002 [-0.0011, +0.0008] | -0.0% |
| quad_modbank_acausal | 180 | 0.132 | 25% | -0.0002 [-0.0041, +0.0036] | -0.0% |
| cochlea_contrast | 20 | 0.222 | 43% | -0.0002 [-0.0016, +0.0014] | -0.0% |
| stmf_mean_gabor | 17 | 0.133 | 25% | -0.0003 [-0.0023, +0.0016] | -0.0% |
| pitch_summary | 2 | 0.082 | 16% | -0.0003 [-0.0007, +0.0000] | -0.1% |
| gate_signals_dyn | 18 | 0.001 | 0% | -0.0004 [-0.0010, +0.0002] | -0.1% |
| quad_xrate_energy | 18 | 0.271 | 52% | -0.0004 [-0.0011, +0.0005] | -0.1% |
| fut_med_contrast_6 | 5 | 0.202 | 39% | -0.0004 [-0.0012, +0.0003] | -0.1% |
| slow_quad_modbank_dyn | 180 | 0.143 | 27% | -0.0005 [-0.0054, +0.0038] | -0.1% |
| fut_minus_cur_contrast | 5 | 0.009 | 2% | -0.0006 [-0.0012, +0.0001] | -0.1% |
| cochlea_voiced_gated | 40 | 0.228 | 44% | -0.0006 [-0.0018, +0.0005] | -0.1% |
| quad_modbank_dyn | 540 | 0.156 | 30% | -0.0006 [-0.0037, +0.0022] | -0.1% |
| cochleagram | 60 | 0.229 | 44% | -0.0007 [-0.0025, +0.0011] | -0.1% |
| pitch_dyn | 6 | 0.043 | 8% | -0.0008 [-0.0018, +0.0003] | -0.1% |
| formant_dyn | 9 | 0.049 | 9% | -0.0008 [-0.0019, +0.0003] | -0.2% |
| quad_modbank | 180 | 0.361 | 69% | -0.0009 [-0.0036, +0.0014] | -0.2% |
| fut_cochlea_contrast | 5 | 0.189 | 36% | -0.0009 [-0.0020, +0.0001] | -0.2% |
| spectral_rolloff | 2 | 0.114 | 22% | -0.0010 [-0.0021, -0.0000] | -0.2% |
| hnr | 2 | 0.111 | 21% | -0.0011 [-0.0023, -0.0000] | -0.2% |
| med_adapt_contrast | 5 | 0.185 | 35% | -0.0011 [-0.0026, +0.0003] | -0.2% |
| fut_cochlea_contrast_med | 5 | 0.188 | 36% | -0.0012 [-0.0023, -0.0001] | -0.2% |
| flux | 36 | 0.286 | 55% | -0.0016 [-0.0032, -0.0001] | -0.3% |
| region_flat | 8 | 0.053 | 10% | -0.0017 [-0.0035, +0.0000] | -0.3% |
| formants | 24 | 0.269 | 52% | -0.0019 [-0.0037, -0.0003] | -0.4% |
| spectral_contrast | 24 | 0.265 | 51% | -0.0019 [-0.0036, -0.0004] | -0.4% |
| phonetic | 5 | 0.218 | 42% | -0.0020 [-0.0036, -0.0006] | -0.4% |
| mod_depth | 40 | 0.278 | 53% | -0.0026 [-0.0075, +0.0021] | -0.5% |
| interact_v3e | 405 | 0.366 | 70% | -0.0044 [-0.0130, +0.0030] | -0.8% |
| tonic | 300 | 0.350 | 67% | -0.0135 [-0.0297, +0.0042] | -2.6% |

Sum of unique shares: 2.3% → shared (not attributable to any one unit): 97.7%

