# patient4: LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning

Held-out session 2eeb170d (107 chunks); 500 training chunks from the other sessions; 155 electrodes, grid = first 96. Same definitions as runs-wave/lean-whisper-newtest (unique/shared as % of R²(Both), 95% CI from a paired chunk bootstrap).

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.0604 | 0.0578 | 0.0591 | 0.522 | 5.8% [-0.8, 13.6] | 9.1% [5.2, 12.9] | 85.2% [71.5, 104.2] |
| all | 0.0655 | 0.0613 | 0.0621 | 0.604 | 12.6% [3.5, 23.5] | 13.8% [8.1, 19.5] | 73.6% [57.8, 95.7] |
