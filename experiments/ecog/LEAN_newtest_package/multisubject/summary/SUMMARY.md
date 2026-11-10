# Cross-subject summary (grid electrodes, held-out sessions)

patient1 = all 493 never-used chunks of its held-out session (unfiltered, like the others). Trial scores: 593 Figure 1 trials. "Kept" = the model the search kept on patient1 (LEAN-v35-AUTOLAGZ). Unique/shared as % of R²(Both).

| Subject | Test chunks | v16 | v20 | Kept | v16→v20 | Whisper | Both | Both − Whisper | Unique LEAN [95% CI] | Unique Whisper | Shared | r(trials, patient1 old) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patient1 | 493 | 0.0649 | 0.0690 | 0.0691 | +0.0041 | 0.0752 | 0.0810 | +0.0058 | 5.8% [4.9, 6.8] | 25.3% | 68.9% | 0.98 |
| patient2 | 318 | 0.0661 | 0.0643 | 0.0641 | -0.0018 | 0.0776 | 0.0752 | -0.0023 | -0.1% [-3.3, 3.3] | 26.2% | 74.0% | 0.80 |
| patient3 | 416 | 0.0592 | 0.0604 | 0.0603 | +0.0012 | 0.0710 | 0.0717 | +0.0007 | 1.4% [0.2, 2.7] | 26.4% | 72.2% | 0.84 |
| patient4 | 107 | 0.0575 | 0.0578 | 0.0575 | +0.0003 | 0.0591 | 0.0604 | +0.0014 | 5.8% [-0.8, 13.6] | 9.1% | 85.2% | 0.85 |
| patient5 | 275 | 0.0628 | 0.0637 | 0.0634 | +0.0009 | 0.0678 | 0.0778 | +0.0100 | 17.2% [12.6, 22.6] | 34.6% | 48.2% | 0.94 |
| patient6 | 222 | 0.0606 | 0.0614 | 0.0615 | +0.0008 | 0.0649 | 0.0704 | +0.0055 | 11.8% [7.9, 16.3] | 23.6% | 64.6% | 0.89 |

## LEAN-v20 family unique variance (% of full model)

| Subject | Modulation filters | Interactions (v3e) | Wavelet scattering | Contrast | Pitch and voicing | Shape, flux, formants | Cochleagram + context |
|---|---|---|---|---|---|---|---|
| patient1 | 7.2 | 2.0 | 1.2 | 0.7 | 0.4 | 0.4 | 0.4 |
| patient2 | 4.8 | 0.8 | 0.5 | 0.1 | 0.4 | -0.2 | -6.6 |
| patient3 | 6.2 | 0.3 | 0.7 | -0.2 | 0.7 | -0.2 | -2.2 |
| patient4 | 5.5 | -1.1 | 0.9 | 0.3 | -0.0 | -1.6 | -1.6 |
| patient5 | 2.0 | -0.5 | -0.4 | 1.1 | 0.0 | -0.7 | -2.0 |
| patient6 | 5.2 | -0.1 | -0.3 | -0.6 | -0.2 | -0.3 | 3.4 |
