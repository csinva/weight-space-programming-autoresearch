# patient3: LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning

Held-out session e3d31b95 (416 chunks); 500 training chunks from the other sessions; 116 electrodes, grid = first 63. Same definitions as runs-wave/lean-whisper-newtest (unique/shared as % of R²(Both), 95% CI from a paired chunk bootstrap).

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.0717 | 0.0604 | 0.0710 | 1.532 | 1.4% [0.2, 2.7] | 26.4% [24.1, 28.7] | 72.2% [65.8, 79.0] |
| all | 0.0613 | 0.0523 | 0.0589 | 1.093 | 4.4% [2.6, 6.4] | 23.4% [21.2, 25.6] | 72.2% [65.8, 79.0] |
