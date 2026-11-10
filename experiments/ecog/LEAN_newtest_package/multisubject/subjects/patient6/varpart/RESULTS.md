# LEAN-v20 variance partitioning — patient6, held-out session 5d2842ef

222 test chunks (the whole held-out session, never used in training); 500 training chunks from the other sessions; 167 electrodes, grid = first 127. Harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model

- grid: mean r = 0.0614, R² = 0.654% of variance
- all: mean r = 0.0483, R² = 0.499% of variance

## Family level — grid (full R² = 0.654%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.595 | 91% | +0.0341 [+0.0256, +0.0427] | +5.2% |
| Cochleagram core + slow context + lookahead | 960 | 0.548 | 84% | +0.0225 [+0.0102, +0.0373] | +3.4% |
| Interpretable interactions (v3e) | 405 | 0.538 | 82% | -0.0006 [-0.0044, +0.0030] | -0.1% |
| Pitch, voicing, voice quality | 122 | 0.404 | 62% | -0.0016 [-0.0033, -0.0004] | -0.2% |
| Wavelet scattering | 287 | 0.554 | 85% | -0.0017 [-0.0050, +0.0012] | -0.3% |
| Spectral shape, flux, formants | 83 | 0.501 | 77% | -0.0020 [-0.0031, -0.0010] | -0.3% |
| Spectral/cochleagram contrast | 192 | 0.493 | 75% | -0.0040 [-0.0070, -0.0015] | -0.6% |

Sum of unique shares: 7.1% → shared (not attributable to any one unit): 92.9%

## Family level — all (full R² = 0.499%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 0.454 | 91% | +0.0258 [+0.0193, +0.0322] | +5.2% |
| Cochleagram core + slow context + lookahead | 960 | 0.418 | 84% | +0.0170 [+0.0075, +0.0281] | +3.4% |
| Interpretable interactions (v3e) | 405 | 0.410 | 82% | -0.0004 [-0.0033, +0.0024] | -0.1% |
| Wavelet scattering | 287 | 0.423 | 85% | -0.0010 [-0.0036, +0.0013] | -0.2% |
| Pitch, voicing, voice quality | 122 | 0.309 | 62% | -0.0011 [-0.0025, -0.0002] | -0.2% |
| Spectral shape, flux, formants | 83 | 0.382 | 77% | -0.0015 [-0.0024, -0.0008] | -0.3% |
| Spectral/cochleagram contrast | 192 | 0.376 | 75% | -0.0030 [-0.0053, -0.0011] | -0.6% |

Sum of unique shares: 7.1% → shared (not attributable to any one unit): 92.9%

## Section level — grid (full R² = 0.654%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| H. Slow context + lookahead | 600 | 0.558 | 85% | +0.0168 [+0.0070, +0.0288] | +2.6% |
| E. 1-D temporal modulation | 132 | 0.456 | 70% | +0.0090 [+0.0051, +0.0131] | +1.4% |
| G. Formants | 33 | 0.364 | 56% | +0.0008 [+0.0004, +0.0013] | +0.1% |
| O. Slow quadrature bank | 360 | 0.457 | 70% | +0.0004 [-0.0012, +0.0020] | +0.1% |
| K. Voice quality (CPP, region flatness) | 12 | 0.145 | 22% | +0.0001 [-0.0002, +0.0005] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.012 | 2% | -0.0003 [-0.0004, -0.0002] | -0.0% |
| J. Interactions (v3e) | 405 | 0.538 | 82% | -0.0006 [-0.0044, +0.0030] | -0.1% |
| C. Spectral shape | 14 | 0.334 | 51% | -0.0009 [-0.0014, -0.0004] | -0.1% |
| D. Spectral flux | 36 | 0.385 | 59% | -0.0017 [-0.0024, -0.0010] | -0.3% |
| P. Wavelet scattering | 287 | 0.554 | 85% | -0.0017 [-0.0050, +0.0012] | -0.3% |
| B. Pitch | 19 | 0.120 | 18% | -0.0020 [-0.0026, -0.0015] | -0.3% |
| N. Voicing gates | 84 | 0.410 | 63% | -0.0024 [-0.0041, -0.0014] | -0.4% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.481 | 74% | -0.0031 [-0.0060, -0.0005] | -0.5% |
| L. Spectral/cochleagram contrast | 192 | 0.493 | 75% | -0.0040 [-0.0070, -0.0015] | -0.6% |
| M. Quadrature modulation bank | 954 | 0.582 | 89% | -0.0107 [-0.0157, -0.0062] | -1.6% |
| A. Cochleagram core | 360 | 0.472 | 72% | -0.0128 [-0.0168, -0.0091] | -2.0% |

Sum of unique shares: -2.0% → shared (not attributable to any one unit): 102.0%

## Section level — all (full R² = 0.499%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| H. Slow context + lookahead | 600 | 0.425 | 85% | +0.0125 [+0.0052, +0.0217] | +2.5% |
| E. 1-D temporal modulation | 132 | 0.347 | 70% | +0.0068 [+0.0038, +0.0099] | +1.4% |
| G. Formants | 33 | 0.277 | 56% | +0.0006 [+0.0002, +0.0009] | +0.1% |
| O. Slow quadrature bank | 360 | 0.349 | 70% | +0.0004 [-0.0009, +0.0015] | +0.1% |
| K. Voice quality (CPP, region flatness) | 12 | 0.111 | 22% | +0.0001 [-0.0002, +0.0004] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.009 | 2% | -0.0002 [-0.0003, -0.0002] | -0.0% |
| J. Interactions (v3e) | 405 | 0.410 | 82% | -0.0004 [-0.0033, +0.0024] | -0.1% |
| C. Spectral shape | 14 | 0.255 | 51% | -0.0007 [-0.0011, -0.0003] | -0.1% |
| P. Wavelet scattering | 287 | 0.423 | 85% | -0.0010 [-0.0036, +0.0013] | -0.2% |
| D. Spectral flux | 36 | 0.293 | 59% | -0.0012 [-0.0018, -0.0008] | -0.2% |
| B. Pitch | 19 | 0.093 | 19% | -0.0015 [-0.0019, -0.0011] | -0.3% |
| N. Voicing gates | 84 | 0.313 | 63% | -0.0018 [-0.0031, -0.0011] | -0.4% |
| F. Spectro-temporal modulation (STMF) | 357 | 0.367 | 74% | -0.0024 [-0.0046, -0.0004] | -0.5% |
| L. Spectral/cochleagram contrast | 192 | 0.376 | 75% | -0.0030 [-0.0053, -0.0011] | -0.6% |
| M. Quadrature modulation bank | 954 | 0.443 | 89% | -0.0082 [-0.0120, -0.0047] | -1.6% |
| A. Cochleagram core | 360 | 0.360 | 72% | -0.0097 [-0.0128, -0.0069] | -2.0% |

Sum of unique shares: -2.0% → shared (not attributable to any one unit): 102.0%

## Group level — grid (full R² = 0.654%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| modulation | 132 | 0.456 | 70% | +0.0090 [+0.0051, +0.0131] | +1.4% |
| tonic | 300 | 0.488 | 75% | +0.0052 [-0.0039, +0.0158] | +0.8% |
| slow_quad_modbank_dyn | 180 | 0.170 | 26% | +0.0012 [+0.0000, +0.0024] | +0.2% |
| slow_quad_modbank | 180 | 0.435 | 67% | +0.0010 [-0.0000, +0.0019] | +0.2% |
| stmf | 340 | 0.480 | 73% | +0.0008 [-0.0015, +0.0028] | +0.1% |
| future_past | 260 | 0.362 | 55% | +0.0007 [-0.0002, +0.0017] | +0.1% |
| fut_cochlea_contrast_med | 5 | 0.148 | 23% | +0.0005 [+0.0002, +0.0007] | +0.1% |
| spectral_moments | 4 | 0.190 | 29% | +0.0004 [+0.0002, +0.0008] | +0.1% |
| fut_minus_cur_contrast | 5 | 0.006 | 1% | +0.0003 [+0.0001, +0.0006] | +0.0% |
| spectral_flatness | 1 | 0.071 | 11% | +0.0003 [+0.0001, +0.0005] | +0.0% |
| formant_dyn | 9 | 0.101 | 15% | +0.0003 [+0.0000, +0.0005] | +0.0% |
| pitch_dyn | 6 | 0.056 | 9% | +0.0002 [-0.0000, +0.0005] | +0.0% |
| pitch_summary | 2 | 0.076 | 12% | +0.0000 [-0.0002, +0.0003] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.068 | 10% | -0.0001 [-0.0004, +0.0001] | -0.0% |
| hnr_dyn | 3 | 0.001 | 0% | -0.0002 [-0.0002, -0.0001] | -0.0% |
| topbot | 10 | 0.353 | 54% | -0.0002 [-0.0005, +0.0001] | -0.0% |
| med_adapt_contrast | 5 | 0.259 | 40% | -0.0002 [-0.0004, -0.0001] | -0.0% |
| cpp | 4 | 0.161 | 25% | -0.0002 [-0.0004, -0.0001] | -0.0% |
| gate_signals | 6 | 0.001 | 0% | -0.0003 [-0.0005, -0.0001] | -0.0% |
| f0_dyn | 4 | 0.011 | 2% | -0.0003 [-0.0004, -0.0002] | -0.0% |
| mod_depth | 40 | 0.417 | 64% | -0.0003 [-0.0021, +0.0013] | -0.1% |
| formants | 24 | 0.341 | 52% | -0.0004 [-0.0008, +0.0001] | -0.1% |
| spectral_contrast | 24 | 0.330 | 50% | -0.0005 [-0.0013, +0.0003] | -0.1% |
| interact_v3e | 405 | 0.538 | 82% | -0.0006 [-0.0044, +0.0030] | -0.1% |
| cube_fut_contrast | 5 | 0.218 | 33% | -0.0006 [-0.0008, -0.0003] | -0.1% |
| med_cochlea_contrast | 20 | 0.287 | 44% | -0.0006 [-0.0011, -0.0002] | -0.1% |
| topbot_short | 10 | 0.326 | 50% | -0.0009 [-0.0012, -0.0005] | -0.1% |
| quad_xrate_energy | 18 | 0.383 | 59% | -0.0009 [-0.0015, -0.0003] | -0.1% |
| hnr | 2 | 0.110 | 17% | -0.0009 [-0.0012, -0.0007] | -0.1% |
| cochlea_voiced_gated | 40 | 0.361 | 55% | -0.0009 [-0.0027, -0.0000] | -0.1% |
| gate_signals_dyn | 18 | 0.000 | 0% | -0.0011 [-0.0016, -0.0006] | -0.2% |
| phonetic | 5 | 0.329 | 50% | -0.0012 [-0.0015, -0.0009] | -0.2% |
| spectral_rolloff | 2 | 0.138 | 21% | -0.0012 [-0.0015, -0.0009] | -0.2% |
| stmf_mean_gabor | 17 | 0.061 | 9% | -0.0013 [-0.0020, -0.0006] | -0.2% |
| cochleagram | 60 | 0.366 | 56% | -0.0014 [-0.0023, -0.0004] | -0.2% |
| multitau_contrast | 10 | 0.329 | 50% | -0.0014 [-0.0019, -0.0008] | -0.2% |
| cochlea_contrast | 20 | 0.280 | 43% | -0.0014 [-0.0019, -0.0010] | -0.2% |
| region_flat | 8 | 0.041 | 6% | -0.0014 [-0.0019, -0.0009] | -0.2% |
| fut_med_contrast_6 | 5 | 0.129 | 20% | -0.0016 [-0.0020, -0.0012] | -0.2% |
| flux | 36 | 0.385 | 59% | -0.0017 [-0.0024, -0.0010] | -0.3% |
| scattering | 287 | 0.554 | 85% | -0.0017 [-0.0050, +0.0012] | -0.3% |
| fut_cochlea_contrast | 5 | 0.157 | 24% | -0.0017 [-0.0021, -0.0014] | -0.3% |
| pitch_salience | 11 | 0.065 | 10% | -0.0018 [-0.0023, -0.0014] | -0.3% |
| short_cochlea_contrast | 20 | 0.287 | 44% | -0.0021 [-0.0027, -0.0015] | -0.3% |
| contrast_med | 24 | 0.313 | 48% | -0.0026 [-0.0033, -0.0019] | -0.4% |
| quad_modbank | 180 | 0.459 | 70% | -0.0037 [-0.0055, -0.0022] | -0.6% |
| quad_xrate_energy_dyn | 36 | 0.097 | 15% | -0.0037 [-0.0048, -0.0028] | -0.6% |
| contrast_short | 24 | 0.262 | 40% | -0.0038 [-0.0051, -0.0027] | -0.6% |
| quad_modbank_acausal | 180 | 0.185 | 28% | -0.0103 [-0.0136, -0.0074] | -1.6% |
| adapt_delta | 300 | 0.201 | 31% | -0.0114 [-0.0152, -0.0080] | -1.7% |
| quad_modbank_dyn | 540 | 0.218 | 33% | -0.0135 [-0.0171, -0.0103] | -2.1% |

Sum of unique shares: -8.9% → shared (not attributable to any one unit): 108.9%

## Group level — all (full R² = 0.499%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| modulation | 132 | 0.347 | 70% | +0.0068 [+0.0038, +0.0099] | +1.4% |
| tonic | 300 | 0.372 | 75% | +0.0037 [-0.0032, +0.0116] | +0.7% |
| slow_quad_modbank_dyn | 180 | 0.129 | 26% | +0.0010 [+0.0001, +0.0019] | +0.2% |
| slow_quad_modbank | 180 | 0.332 | 67% | +0.0007 [-0.0000, +0.0015] | +0.1% |
| stmf | 340 | 0.367 | 73% | +0.0005 [-0.0012, +0.0021] | +0.1% |
| future_past | 260 | 0.275 | 55% | +0.0005 [-0.0003, +0.0012] | +0.1% |
| fut_cochlea_contrast_med | 5 | 0.113 | 23% | +0.0003 [+0.0002, +0.0005] | +0.1% |
| spectral_moments | 4 | 0.146 | 29% | +0.0003 [+0.0001, +0.0006] | +0.1% |
| fut_minus_cur_contrast | 5 | 0.005 | 1% | +0.0002 [+0.0001, +0.0004] | +0.0% |
| spectral_flatness | 1 | 0.055 | 11% | +0.0002 [+0.0001, +0.0004] | +0.0% |
| formant_dyn | 9 | 0.077 | 15% | +0.0002 [+0.0000, +0.0004] | +0.0% |
| pitch_dyn | 6 | 0.043 | 9% | +0.0002 [-0.0000, +0.0004] | +0.0% |
| pitch_summary | 2 | 0.059 | 12% | +0.0000 [-0.0002, +0.0002] | +0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.051 | 10% | -0.0001 [-0.0003, +0.0001] | -0.0% |
| topbot | 10 | 0.269 | 54% | -0.0001 [-0.0004, +0.0001] | -0.0% |
| hnr_dyn | 3 | 0.001 | 0% | -0.0001 [-0.0002, -0.0001] | -0.0% |
| med_adapt_contrast | 5 | 0.198 | 40% | -0.0002 [-0.0003, -0.0001] | -0.0% |
| cpp | 4 | 0.123 | 25% | -0.0002 [-0.0003, -0.0001] | -0.0% |
| gate_signals | 6 | 0.000 | 0% | -0.0002 [-0.0003, -0.0000] | -0.0% |
| mod_depth | 40 | 0.318 | 64% | -0.0002 [-0.0015, +0.0010] | -0.0% |
| f0_dyn | 4 | 0.009 | 2% | -0.0002 [-0.0003, -0.0002] | -0.0% |
| formants | 24 | 0.260 | 52% | -0.0003 [-0.0006, +0.0000] | -0.1% |
| interact_v3e | 405 | 0.410 | 82% | -0.0004 [-0.0033, +0.0024] | -0.1% |
| spectral_contrast | 24 | 0.252 | 51% | -0.0004 [-0.0010, +0.0002] | -0.1% |
| cube_fut_contrast | 5 | 0.166 | 33% | -0.0004 [-0.0006, -0.0002] | -0.1% |
| med_cochlea_contrast | 20 | 0.219 | 44% | -0.0005 [-0.0008, -0.0001] | -0.1% |
| topbot_short | 10 | 0.249 | 50% | -0.0007 [-0.0009, -0.0004] | -0.1% |
| quad_xrate_energy | 18 | 0.292 | 58% | -0.0007 [-0.0011, -0.0002] | -0.1% |
| cochlea_voiced_gated | 40 | 0.275 | 55% | -0.0007 [-0.0021, -0.0000] | -0.1% |
| hnr | 2 | 0.084 | 17% | -0.0007 [-0.0009, -0.0005] | -0.1% |
| gate_signals_dyn | 18 | 0.000 | 0% | -0.0008 [-0.0012, -0.0005] | -0.2% |
| phonetic | 5 | 0.251 | 50% | -0.0009 [-0.0012, -0.0007] | -0.2% |
| spectral_rolloff | 2 | 0.106 | 21% | -0.0009 [-0.0012, -0.0007] | -0.2% |
| stmf_mean_gabor | 17 | 0.047 | 9% | -0.0010 [-0.0015, -0.0005] | -0.2% |
| cochleagram | 60 | 0.279 | 56% | -0.0010 [-0.0017, -0.0003] | -0.2% |
| scattering | 287 | 0.423 | 85% | -0.0010 [-0.0036, +0.0013] | -0.2% |
| multitau_contrast | 10 | 0.251 | 50% | -0.0010 [-0.0014, -0.0006] | -0.2% |
| cochlea_contrast | 20 | 0.214 | 43% | -0.0011 [-0.0015, -0.0007] | -0.2% |
| region_flat | 8 | 0.032 | 6% | -0.0011 [-0.0015, -0.0007] | -0.2% |
| fut_med_contrast_6 | 5 | 0.099 | 20% | -0.0012 [-0.0015, -0.0009] | -0.2% |
| flux | 36 | 0.293 | 59% | -0.0012 [-0.0018, -0.0008] | -0.2% |
| pitch_salience | 11 | 0.051 | 10% | -0.0013 [-0.0017, -0.0010] | -0.3% |
| fut_cochlea_contrast | 5 | 0.120 | 24% | -0.0013 [-0.0016, -0.0011] | -0.3% |
| short_cochlea_contrast | 20 | 0.219 | 44% | -0.0016 [-0.0021, -0.0011] | -0.3% |
| contrast_med | 24 | 0.239 | 48% | -0.0020 [-0.0025, -0.0015] | -0.4% |
| quad_xrate_energy_dyn | 36 | 0.074 | 15% | -0.0028 [-0.0037, -0.0021] | -0.6% |
| quad_modbank | 180 | 0.350 | 70% | -0.0028 [-0.0042, -0.0017] | -0.6% |
| contrast_short | 24 | 0.200 | 40% | -0.0029 [-0.0039, -0.0021] | -0.6% |
| quad_modbank_acausal | 180 | 0.141 | 28% | -0.0079 [-0.0105, -0.0057] | -1.6% |
| adapt_delta | 300 | 0.154 | 31% | -0.0087 [-0.0116, -0.0062] | -1.8% |
| quad_modbank_dyn | 540 | 0.166 | 33% | -0.0104 [-0.0131, -0.0080] | -2.1% |

Sum of unique shares: -8.9% → shared (not attributable to any one unit): 108.9%

