# LEAN-v20 variance partitioning

Fixed 23-chunk test set, 500 training chunks, harness ridge (per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the canonical per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test chunks.

## Full model (validation)

- grid: mean r = 0.0904 (canonical 0.0904), R² = 1.894% of variance
- all166: mean r = 0.0808 (canonical 0.0808), R² = 1.586% of variance
- per-electrode r vs canonical diag_v20: corr 1.00000, max |diff| 0.00000

## Family level — grid (full R² = 1.894%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 1.715 | 91% | +0.0766 [+0.0318, +0.1084] | +4.0% |
| Cochleagram core + slow context + lookahead | 960 | 1.588 | 84% | +0.0581 [+0.0121, +0.1372] | +3.1% |
| Wavelet scattering | 287 | 1.467 | 77% | +0.0303 [+0.0167, +0.0494] | +1.6% |
| Interpretable interactions (v3e) | 405 | 1.554 | 82% | +0.0278 [+0.0141, +0.0400] | +1.5% |
| Spectral/cochleagram contrast | 192 | 1.360 | 72% | +0.0183 [+0.0052, +0.0342] | +1.0% |
| Spectral shape, flux, formants | 83 | 1.314 | 69% | +0.0070 [+0.0023, +0.0113] | +0.4% |
| Pitch, voicing, voice quality | 122 | 1.149 | 61% | +0.0067 [+0.0033, +0.0107] | +0.4% |

Sum of unique shares: 11.9% → shared (not attributable to any one unit): 88.1%

## Family level — all166 (full R² = 1.586%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| Modulation filters (1-D, STMF, quadrature, slow quadrature) | 1803 | 1.420 | 90% | +0.0615 [+0.0214, +0.0900] | +3.9% |
| Cochleagram core + slow context + lookahead | 960 | 1.338 | 84% | +0.0568 [+0.0150, +0.1304] | +3.6% |
| Interpretable interactions (v3e) | 405 | 1.271 | 80% | +0.0356 [+0.0236, +0.0470] | +2.2% |
| Wavelet scattering | 287 | 1.210 | 76% | +0.0260 [+0.0152, +0.0417] | +1.6% |
| Spectral/cochleagram contrast | 192 | 1.127 | 71% | +0.0195 [+0.0095, +0.0321] | +1.2% |
| Pitch, voicing, voice quality | 122 | 0.946 | 60% | +0.0118 [+0.0082, +0.0162] | +0.7% |
| Spectral shape, flux, formants | 83 | 1.072 | 68% | +0.0048 [+0.0009, +0.0080] | +0.3% |

Sum of unique shares: 13.6% → shared (not attributable to any one unit): 86.4%

## Section level — grid (full R² = 1.894%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| H. Slow context + lookahead | 600 | 1.532 | 81% | +0.0456 [+0.0021, +0.1231] | +2.4% |
| P. Wavelet scattering | 287 | 1.467 | 77% | +0.0303 [+0.0167, +0.0494] | +1.6% |
| E. 1-D temporal modulation | 132 | 1.085 | 57% | +0.0295 [+0.0069, +0.0494] | +1.6% |
| J. Interactions (v3e) | 405 | 1.554 | 82% | +0.0278 [+0.0141, +0.0400] | +1.5% |
| L. Spectral/cochleagram contrast | 192 | 1.360 | 72% | +0.0183 [+0.0052, +0.0342] | +1.0% |
| M. Quadrature modulation bank | 954 | 1.610 | 85% | +0.0176 [+0.0020, +0.0309] | +0.9% |
| D. Spectral flux | 36 | 1.039 | 55% | +0.0093 [+0.0062, +0.0122] | +0.5% |
| A. Cochleagram core | 360 | 1.207 | 64% | +0.0057 [-0.0008, +0.0145] | +0.3% |
| B. Pitch | 19 | 0.758 | 40% | +0.0051 [+0.0030, +0.0070] | +0.3% |
| O. Slow quadrature bank | 360 | 1.380 | 73% | +0.0021 [-0.0042, +0.0088] | +0.1% |
| K. Voice quality (CPP, region flatness) | 12 | 0.410 | 22% | +0.0021 [+0.0006, +0.0037] | +0.1% |
| N. Voicing gates | 84 | 1.073 | 57% | +0.0013 [-0.0019, +0.0046] | +0.1% |
| G. Formants | 33 | 0.991 | 52% | +0.0003 [-0.0016, +0.0017] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.122 | 6% | -0.0007 [-0.0016, +0.0001] | -0.0% |
| C. Spectral shape | 14 | 1.011 | 53% | -0.0015 [-0.0037, +0.0007] | -0.1% |
| F. Spectro-temporal modulation (STMF) | 357 | 1.395 | 74% | -0.0015 [-0.0194, +0.0126] | -0.1% |

Sum of unique shares: 10.1% → shared (not attributable to any one unit): 89.9%

## Section level — all166 (full R² = 1.586%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| H. Slow context + lookahead | 600 | 1.291 | 81% | +0.0423 [+0.0027, +0.1150] | +2.7% |
| J. Interactions (v3e) | 405 | 1.271 | 80% | +0.0356 [+0.0236, +0.0470] | +2.2% |
| E. 1-D temporal modulation | 132 | 0.870 | 55% | +0.0298 [+0.0101, +0.0470] | +1.9% |
| P. Wavelet scattering | 287 | 1.210 | 76% | +0.0260 [+0.0152, +0.0417] | +1.6% |
| L. Spectral/cochleagram contrast | 192 | 1.127 | 71% | +0.0195 [+0.0095, +0.0321] | +1.2% |
| M. Quadrature modulation bank | 954 | 1.343 | 85% | +0.0132 [-0.0005, +0.0246] | +0.8% |
| B. Pitch | 19 | 0.612 | 39% | +0.0090 [+0.0067, +0.0110] | +0.6% |
| A. Cochleagram core | 360 | 0.995 | 63% | +0.0073 [+0.0018, +0.0140] | +0.5% |
| O. Slow quadrature bank | 360 | 1.161 | 73% | +0.0070 [+0.0009, +0.0126] | +0.4% |
| D. Spectral flux | 36 | 0.850 | 54% | +0.0068 [+0.0044, +0.0090] | +0.4% |
| N. Voicing gates | 84 | 0.891 | 56% | +0.0031 [+0.0007, +0.0059] | +0.2% |
| K. Voice quality (CPP, region flatness) | 12 | 0.326 | 21% | +0.0018 [+0.0006, +0.0034] | +0.1% |
| F. Spectro-temporal modulation (STMF) | 357 | 1.162 | 73% | +0.0005 [-0.0154, +0.0115] | +0.0% |
| G. Formants | 33 | 0.810 | 51% | +0.0002 [-0.0013, +0.0015] | +0.0% |
| I. Pitch/voicing dynamics | 7 | 0.096 | 6% | -0.0008 [-0.0014, -0.0001] | -0.0% |
| C. Spectral shape | 14 | 0.837 | 53% | -0.0010 [-0.0029, +0.0008] | -0.1% |

Sum of unique shares: 12.6% → shared (not attributable to any one unit): 87.4%

## Group level — grid (full R² = 1.894%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| tonic | 300 | 1.204 | 64% | +0.0463 [+0.0040, +0.1134] | +2.4% |
| scattering | 287 | 1.467 | 77% | +0.0303 [+0.0167, +0.0494] | +1.6% |
| modulation | 132 | 1.085 | 57% | +0.0295 [+0.0069, +0.0494] | +1.6% |
| interact_v3e | 405 | 1.554 | 82% | +0.0278 [+0.0141, +0.0400] | +1.5% |
| mod_depth | 40 | 0.999 | 53% | +0.0144 [+0.0075, +0.0218] | +0.8% |
| quad_modbank_acausal | 180 | 1.158 | 61% | +0.0098 [+0.0021, +0.0180] | +0.5% |
| flux | 36 | 1.039 | 55% | +0.0093 [+0.0062, +0.0122] | +0.5% |
| future_past | 260 | 1.034 | 55% | +0.0092 [+0.0016, +0.0169] | +0.5% |
| adapt_delta | 300 | 1.143 | 60% | +0.0074 [+0.0015, +0.0153] | +0.4% |
| quad_modbank_dyn | 540 | 1.240 | 65% | +0.0063 [-0.0025, +0.0127] | +0.3% |
| slow_quad_modbank | 180 | 1.288 | 68% | +0.0057 [+0.0010, +0.0101] | +0.3% |
| stmf_mean_gabor | 17 | 0.111 | 6% | +0.0055 [+0.0005, +0.0104] | +0.3% |
| pitch_salience | 11 | 0.676 | 36% | +0.0042 [+0.0026, +0.0059] | +0.2% |
| contrast_med | 24 | 0.688 | 36% | +0.0041 [+0.0008, +0.0077] | +0.2% |
| spectral_contrast | 24 | 0.764 | 40% | +0.0040 [+0.0012, +0.0067] | +0.2% |
| region_flat | 8 | 0.326 | 17% | +0.0033 [+0.0018, +0.0049] | +0.2% |
| quad_modbank | 180 | 1.339 | 71% | +0.0026 [-0.0020, +0.0056] | +0.1% |
| short_cochlea_contrast | 20 | 0.644 | 34% | +0.0025 [+0.0009, +0.0042] | +0.1% |
| phonetic | 5 | 0.968 | 51% | +0.0021 [+0.0007, +0.0034] | +0.1% |
| gate_signals | 6 | 0.002 | 0% | +0.0020 [-0.0004, +0.0050] | +0.1% |
| contrast_short | 24 | 0.583 | 31% | +0.0019 [+0.0000, +0.0039] | +0.1% |
| cochlea_voiced_gated | 40 | 1.029 | 54% | +0.0018 [+0.0007, +0.0028] | +0.1% |
| formant_dyn | 9 | 0.202 | 11% | +0.0015 [+0.0002, +0.0026] | +0.1% |
| gate_signals_dyn | 18 | 0.004 | 0% | +0.0014 [-0.0006, +0.0040] | +0.1% |
| fut_cochlea_contrast | 5 | 0.663 | 35% | +0.0011 [+0.0001, +0.0022] | +0.1% |
| cube_fut_contrast | 5 | 0.666 | 35% | +0.0010 [-0.0000, +0.0021] | +0.1% |
| cochlea_contrast | 20 | 0.740 | 39% | +0.0009 [-0.0019, +0.0038] | +0.0% |
| fut_cochlea_contrast_med | 5 | 0.648 | 34% | +0.0008 [+0.0002, +0.0013] | +0.0% |
| multitau_contrast | 10 | 0.819 | 43% | +0.0007 [-0.0021, +0.0033] | +0.0% |
| med_cochlea_contrast | 20 | 0.714 | 38% | +0.0006 [-0.0006, +0.0023] | +0.0% |
| pitch_dyn | 6 | 0.491 | 26% | +0.0006 [-0.0002, +0.0014] | +0.0% |
| topbot | 10 | 1.006 | 53% | +0.0006 [-0.0009, +0.0019] | +0.0% |
| cochleagram | 60 | 1.058 | 56% | +0.0003 [-0.0031, +0.0035] | +0.0% |
| formants | 24 | 0.946 | 50% | +0.0003 [-0.0012, +0.0016] | +0.0% |
| med_adapt_contrast | 5 | 0.656 | 35% | +0.0003 [-0.0008, +0.0013] | +0.0% |
| hnr | 2 | 0.552 | 29% | +0.0002 [-0.0002, +0.0005] | +0.0% |
| fut_minus_cur_contrast | 5 | 0.127 | 7% | +0.0001 [-0.0004, +0.0005] | +0.0% |
| quad_xrate_energy_dyn | 36 | 0.752 | 40% | +0.0001 [-0.0019, +0.0021] | +0.0% |
| spectral_flatness | 1 | 0.574 | 30% | +0.0000 [-0.0001, +0.0003] | +0.0% |
| pitch_summary | 2 | 0.558 | 29% | -0.0000 [-0.0004, +0.0002] | -0.0% |
| hnr_dyn | 3 | 0.035 | 2% | -0.0000 [-0.0001, +0.0000] | -0.0% |
| fut_med_contrast_6 | 5 | 0.358 | 19% | -0.0004 [-0.0023, +0.0010] | -0.0% |
| spectral_moments | 4 | 0.730 | 39% | -0.0006 [-0.0018, +0.0006] | -0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.638 | 34% | -0.0006 [-0.0015, +0.0003] | -0.0% |
| topbot_short | 10 | 0.903 | 48% | -0.0006 [-0.0014, +0.0002] | -0.0% |
| f0_dyn | 4 | 0.106 | 6% | -0.0007 [-0.0016, +0.0001] | -0.0% |
| spectral_rolloff | 2 | 0.399 | 21% | -0.0009 [-0.0021, +0.0003] | -0.0% |
| cpp | 4 | 0.353 | 19% | -0.0011 [-0.0024, +0.0002] | -0.1% |
| slow_quad_modbank_dyn | 180 | 1.106 | 58% | -0.0017 [-0.0045, +0.0016] | -0.1% |
| quad_xrate_energy | 18 | 1.026 | 54% | -0.0028 [-0.0052, -0.0006] | -0.1% |
| stmf | 340 | 1.391 | 73% | -0.0047 [-0.0235, +0.0106] | -0.2% |

Sum of unique shares: 12.0% → shared (not attributable to any one unit): 88.0%

## Group level — all166 (full R² = 1.586%)

| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |
|---|---|---|---|---|---|
| tonic | 300 | 0.984 | 62% | +0.0410 [+0.0020, +0.1052] | +2.6% |
| interact_v3e | 405 | 1.271 | 80% | +0.0356 [+0.0236, +0.0470] | +2.2% |
| modulation | 132 | 0.870 | 55% | +0.0298 [+0.0101, +0.0470] | +1.9% |
| scattering | 287 | 1.210 | 76% | +0.0260 [+0.0152, +0.0417] | +1.6% |
| mod_depth | 40 | 0.810 | 51% | +0.0189 [+0.0119, +0.0263] | +1.2% |
| future_past | 260 | 0.866 | 55% | +0.0114 [+0.0051, +0.0175] | +0.7% |
| adapt_delta | 300 | 0.928 | 59% | +0.0088 [+0.0037, +0.0148] | +0.6% |
| contrast_med | 24 | 0.552 | 35% | +0.0080 [+0.0045, +0.0113] | +0.5% |
| flux | 36 | 0.850 | 54% | +0.0068 [+0.0044, +0.0090] | +0.4% |
| quad_modbank_acausal | 180 | 0.966 | 61% | +0.0067 [-0.0000, +0.0134] | +0.4% |
| cochleagram | 60 | 0.873 | 55% | +0.0061 [+0.0039, +0.0083] | +0.4% |
| quad_modbank_dyn | 540 | 1.025 | 65% | +0.0048 [-0.0019, +0.0096] | +0.3% |
| slow_quad_modbank | 180 | 1.085 | 68% | +0.0047 [+0.0010, +0.0083] | +0.3% |
| slow_quad_modbank_dyn | 180 | 0.930 | 59% | +0.0044 [+0.0010, +0.0072] | +0.3% |
| stmf_mean_gabor | 17 | 0.091 | 6% | +0.0035 [-0.0005, +0.0075] | +0.2% |
| pitch_salience | 11 | 0.551 | 35% | +0.0031 [+0.0016, +0.0048] | +0.2% |
| region_flat | 8 | 0.262 | 17% | +0.0030 [+0.0017, +0.0044] | +0.2% |
| quad_modbank | 180 | 1.118 | 71% | +0.0023 [-0.0016, +0.0048] | +0.1% |
| spectral_contrast | 24 | 0.614 | 39% | +0.0022 [-0.0005, +0.0047] | +0.1% |
| phonetic | 5 | 0.803 | 51% | +0.0016 [+0.0004, +0.0027] | +0.1% |
| gate_signals | 6 | 0.002 | 0% | +0.0015 [-0.0004, +0.0037] | +0.1% |
| short_cochlea_contrast | 20 | 0.527 | 33% | +0.0014 [-0.0008, +0.0034] | +0.1% |
| cube_fut_contrast | 5 | 0.570 | 36% | +0.0014 [+0.0007, +0.0023] | +0.1% |
| contrast_short | 24 | 0.469 | 30% | +0.0012 [-0.0004, +0.0029] | +0.1% |
| formant_dyn | 9 | 0.162 | 10% | +0.0012 [+0.0001, +0.0020] | +0.1% |
| gate_signals_dyn | 18 | 0.004 | 0% | +0.0012 [-0.0004, +0.0033] | +0.1% |
| cochlea_voiced_gated | 40 | 0.849 | 54% | +0.0012 [+0.0002, +0.0021] | +0.1% |
| cochlea_contrast | 20 | 0.608 | 38% | +0.0011 [-0.0012, +0.0041] | +0.1% |
| fut_cochlea_contrast | 5 | 0.571 | 36% | +0.0011 [+0.0003, +0.0020] | +0.1% |
| med_cochlea_contrast | 20 | 0.587 | 37% | +0.0008 [-0.0001, +0.0023] | +0.1% |
| fut_cochlea_contrast_med | 5 | 0.555 | 35% | +0.0007 [+0.0001, +0.0011] | +0.0% |
| pitch_dyn | 6 | 0.393 | 25% | +0.0005 [-0.0001, +0.0011] | +0.0% |
| multitau_contrast | 10 | 0.661 | 42% | +0.0003 [-0.0021, +0.0027] | +0.0% |
| formants | 24 | 0.776 | 49% | +0.0003 [-0.0011, +0.0015] | +0.0% |
| topbot | 10 | 0.833 | 53% | +0.0003 [-0.0007, +0.0012] | +0.0% |
| spectral_flatness | 1 | 0.466 | 29% | +0.0001 [-0.0001, +0.0003] | +0.0% |
| hnr | 2 | 0.443 | 28% | +0.0000 [-0.0004, +0.0004] | +0.0% |
| hnr_dyn | 3 | 0.027 | 2% | -0.0000 [-0.0000, +0.0000] | -0.0% |
| pitch_summary | 2 | 0.454 | 29% | -0.0000 [-0.0004, +0.0002] | -0.0% |
| fut_minus_cur_contrast | 5 | 0.108 | 7% | -0.0001 [-0.0004, +0.0002] | -0.0% |
| quad_xrate_energy_dyn | 36 | 0.614 | 39% | -0.0001 [-0.0016, +0.0016] | -0.0% |
| med_adapt_contrast | 5 | 0.531 | 34% | -0.0002 [-0.0009, +0.0006] | -0.0% |
| fut_med_contrast_6 | 5 | 0.324 | 20% | -0.0002 [-0.0016, +0.0009] | -0.0% |
| cochlea_voiced_gated_dyn | 20 | 0.523 | 33% | -0.0003 [-0.0010, +0.0004] | -0.0% |
| topbot_short | 10 | 0.744 | 47% | -0.0004 [-0.0010, +0.0002] | -0.0% |
| f0_dyn | 4 | 0.083 | 5% | -0.0008 [-0.0014, -0.0001] | -0.0% |
| spectral_moments | 4 | 0.603 | 38% | -0.0008 [-0.0019, +0.0002] | -0.1% |
| spectral_rolloff | 2 | 0.332 | 21% | -0.0009 [-0.0018, -0.0001] | -0.1% |
| stmf | 340 | 1.160 | 73% | -0.0013 [-0.0177, +0.0101] | -0.1% |
| cpp | 4 | 0.279 | 18% | -0.0014 [-0.0026, -0.0003] | -0.1% |
| quad_xrate_energy | 18 | 0.847 | 53% | -0.0016 [-0.0036, +0.0003] | -0.1% |

Sum of unique shares: 14.8% → shared (not attributable to any one unit): 85.2%

