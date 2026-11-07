# LEAN-v20 + Whisper ("Both") and LEAN-vs-Whisper variance partitioning

Same 500 training chunks and harness ridge as every LEAN fit; Both = LEAN-v20 (3,852) + Whisper (1,280) features in one ridge (4 delays). LEAN-only and Whisper-only are column subsets of the same pass. R² per electrode = r·|r| (r = chunk-averaged per-chunk Pearson r), averaged over electrodes. Unique LEAN = R²(Both) − R²(Whisper); unique Whisper = R²(Both) − R²(LEAN); shared = R²(LEAN) + R²(Whisper) − R²(Both). 95% CIs: paired bootstrap over test chunks (B = 2000). Primary new test = runs-wave/newtest/test_primary.json.

## Old-test reproduction

- both, grid: 0.10375 vs June 0.1038 — reproduced
- both, all166: 0.09204 vs June 0.0920 — reproduced
- lean, grid: 0.09045 vs June 0.0904 — reproduced
- lean, all166: 0.08079 vs June 0.0808 — reproduced
- whisper, grid: 0.09330 vs June 0.0933 — reproduced
- whisper, all166: 0.08172 vs June 0.0817 — reproduced

## old (23 chunks)

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.1038 | 0.0904 | 0.0933 | 2.537 | 6.5% [3.0, 11.0] | 25.4% [21.3, 29.0] | 68.2% [63.3, 73.9] |
| all166 | 0.0920 | 0.0808 | 0.0817 | 2.092 | 7.8% [4.1, 12.8] | 24.2% [20.5, 27.4] | 68.0% [63.3, 73.5] |

Unique/shared as % of R²(Both) [95% CI].

## new primary (317 chunks)

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.0871 | 0.0735 | 0.0819 | 1.996 | 5.0% [4.0, 6.1] | 26.0% [24.4, 27.5] | 68.9% [65.6, 72.2] |
| all166 | 0.0755 | 0.0646 | 0.0703 | 1.580 | 5.7% [4.6, 6.9] | 25.3% [23.8, 26.8] | 68.9% [65.4, 72.4] |

Unique/shared as % of R²(Both) [95% CI].

## new all (493 chunks)

| Electrodes | r Both | r LEAN | r Whisper | R² Both (%) | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| grid | 0.0810 | 0.0690 | 0.0752 | 1.679 | 5.8% [4.9, 6.8] | 25.3% [23.8, 26.8] | 68.9% [65.1, 72.9] |
| all166 | 0.0711 | 0.0614 | 0.0657 | 1.342 | 6.6% [5.5, 7.7] | 24.5% [23.0, 25.9] | 69.0% [65.2, 73.1] |

Unique/shared as % of R²(Both) [95% CI].

