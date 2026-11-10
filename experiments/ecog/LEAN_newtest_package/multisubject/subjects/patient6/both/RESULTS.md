# patient6: LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning

Held-out session 5d2842ef (222 chunks); 500 training chunks from the other sessions; 167 electrodes, grid = first 127. Same definitions as runs-wave/lean-whisper-newtest (unique/shared as % of R²(Both), 95% CI from a paired chunk bootstrap).

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.0704 | 0.0614 | 0.0649 | 0.856 | 11.8% [7.9, 16.3] | 23.6% [20.6, 26.9] | 64.6% [56.6, 73.8] |
| all | 0.0550 | 0.0483 | 0.0503 | 0.652 | 11.8% [8.0, 16.3] | 23.5% [20.5, 26.8] | 64.7% [56.6, 73.8] |
