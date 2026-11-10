# patient1VAL: LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning

Held-out session 450a36e4 (23 chunks); 500 training chunks from the other sessions; 166 electrodes, grid = first 125. Same definitions as runs-wave/lean-whisper-newtest (unique/shared as % of R²(Both), 95% CI from a paired chunk bootstrap).

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.1038 | 0.0904 | 0.0933 | 2.537 | 6.5% [3.0, 11.0] | 25.4% [21.3, 29.0] | 68.2% [63.3, 73.9] |
| all | 0.0920 | 0.0808 | 0.0817 | 2.092 | 7.8% [4.1, 12.8] | 24.2% [20.5, 27.4] | 68.0% [63.3, 73.5] |
