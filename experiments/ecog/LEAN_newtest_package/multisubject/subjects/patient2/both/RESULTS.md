# patient2: LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning

Held-out session 236d910e (318 chunks); 500 training chunks from the other sessions; 235 electrodes, grid = first 127. Same definitions as runs-wave/lean-whisper-newtest (unique/shared as % of R²(Both), 95% CI from a paired chunk bootstrap).

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.0752 | 0.0643 | 0.0776 | 1.041 | -0.1% [-3.3, 3.3] | 26.2% [23.6, 28.7] | 74.0% [65.4, 84.2] |
| all | 0.0866 | 0.0787 | 0.0826 | 1.206 | 12.9% [7.9, 18.1] | 16.6% [14.7, 18.6] | 70.5% [60.4, 82.7] |
