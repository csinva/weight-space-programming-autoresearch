# patient5: LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning

Held-out session 5e416143 (275 chunks); 500 training chunks from the other sessions; 188 electrodes, grid = first 124. Same definitions as runs-wave/lean-whisper-newtest (unique/shared as % of R²(Both), 95% CI from a paired chunk bootstrap).

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.0778 | 0.0637 | 0.0678 | 0.961 | 17.2% [12.6, 22.6] | 34.6% [31.6, 37.6] | 48.2% [42.2, 55.1] |
| all | 0.0924 | 0.0795 | 0.0761 | 1.298 | 27.7% [21.2, 35.4] | 23.5% [21.1, 25.8] | 48.8% [41.8, 57.1] |
