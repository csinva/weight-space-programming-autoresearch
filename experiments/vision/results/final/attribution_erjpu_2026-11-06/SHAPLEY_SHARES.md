# Conditional Shapley attribution: champion erjpu (LAION search data)

Shapley values computed with Stage 1 (Gabor energy) always present, 60 permutations, 5 subjects averaged.
Values ×1000 (encoding r); ± is the permutation SE. Caveat: LAION test images were used in search, so magnitudes may be inflated.

## Scores

| | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| Stage 1 alone | 0.443 | 0.543 | 0.337 | 0.565 | 0.546 | 0.530 | 0.502 | 0.254 | 0.401 |
| Full champion | 0.487 | 0.580 | 0.390 | 0.608 | 0.579 | 0.565 | 0.544 | 0.328 | 0.445 |
| **Gain (×1000)** | **44.3** | **36.7** | **52.3** | **43.7** | **32.4** | **34.9** | **41.3** | **73.6** | **44.3** |

## Each family's share of the gain (×1000 ± SE)

| Family | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| Surface geometry | 11.2 ± 0.7 | 12.9 ± 0.8 | 9.5 ± 0.7 | 14.2 ± 1.0 | 11.9 ± 0.7 | 13.5 ± 0.8 | 15.4 ± 0.8 | 12.5 ± 0.9 | 2.8 ± 0.5 |
| Global layout/symmetry | 8.0 ± 0.7 | 7.8 ± 0.8 | 8.1 ± 0.6 | 8.5 ± 0.9 | 6.8 ± 0.7 | 8.7 ± 0.8 | 8.9 ± 0.8 | 9.1 ± 0.7 | 8.4 ± 0.6 |
| Cross-octave | 5.2 ± 0.7 | 3.3 ± 0.7 | 7.1 ± 0.8 | 4.4 ± 1.0 | 2.8 ± 0.6 | 2.7 ± 0.6 | 2.6 ± 0.5 | 10.0 ± 1.3 | 8.0 ± 0.8 |
| Geometry relations | 4.8 ± 0.6 | 1.7 ± 0.5 | 8.1 ± 0.7 | 3.0 ± 0.7 | 1.2 ± 0.4 | 0.9 ± 0.3 | 3.3 ± 0.3 | 12.2 ± 1.2 | 8.6 ± 0.7 |
| Texture covariance | 4.6 ± 0.5 | 1.8 ± 0.4 | 7.6 ± 0.5 | 2.4 ± 0.6 | 1.5 ± 0.4 | 1.6 ± 0.3 | 4.3 ± 0.4 | 13.9 ± 1.0 | 4.6 ± 0.4 |
| Phase congruency | 3.0 ± 0.5 | 3.5 ± 0.6 | 2.5 ± 0.5 | 4.2 ± 0.7 | 3.2 ± 0.5 | 3.0 ± 0.6 | 2.4 ± 0.5 | 1.4 ± 0.5 | 3.0 ± 0.4 |
| Joint covariance | 2.9 ± 0.5 | 2.3 ± 0.5 | 3.6 ± 0.5 | 3.1 ± 0.7 | 2.1 ± 0.4 | 1.7 ± 0.3 | 1.6 ± 0.3 | 6.4 ± 1.0 | 3.4 ± 0.4 |
| Orientation field | 2.2 ± 0.3 | 1.3 ± 0.2 | 3.1 ± 0.4 | 1.2 ± 0.2 | 1.4 ± 0.2 | 1.2 ± 0.3 | 0.3 ± 0.2 | 5.5 ± 0.6 | 4.1 ± 0.4 |
| Second-order | 1.5 ± 0.3 | 1.4 ± 0.3 | 1.6 ± 0.2 | 1.4 ± 0.4 | 1.3 ± 0.3 | 1.7 ± 0.3 | 2.2 ± 0.3 | 2.1 ± 0.3 | 0.2 ± 0.1 |
| Colour layout | 0.8 ± 0.0 | 0.5 ± 0.0 | 1.1 ± 0.1 | 1.3 ± 0.1 | 0.2 ± 0.0 | -0.1 ± 0.0 | 0.3 ± 0.0 | 0.5 ± 0.0 | 1.1 ± 0.1 |

## Each family as % of that ROI's gain

| Family | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| Surface geometry | 25% | 35% | 18% | 32% | 37% | 39% | 37% | 17% | 6% |
| Global layout/symmetry | 18% | 21% | 15% | 19% | 21% | 25% | 21% | 12% | 19% |
| Cross-octave | 12% | 9% | 14% | 10% | 9% | 8% | 6% | 14% | 18% |
| Geometry relations | 11% | 5% | 15% | 7% | 4% | 3% | 8% | 17% | 19% |
| Texture covariance | 10% | 5% | 15% | 5% | 5% | 4% | 10% | 19% | 10% |
| Phase congruency | 7% | 10% | 5% | 10% | 10% | 9% | 6% | 2% | 7% |
| Joint covariance | 7% | 6% | 7% | 7% | 7% | 5% | 4% | 9% | 8% |
| Orientation field | 5% | 4% | 6% | 3% | 4% | 4% | 1% | 8% | 9% |
| Second-order | 3% | 4% | 3% | 3% | 4% | 5% | 5% | 3% | 1% |
| Colour layout | 2% | 1% | 2% | 3% | 1% | -0% | 1% | 1% | 3% |

## Grouped, as % of each ROI's gain

| Group | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| Correlation statistics (cross-octave + geometry relations + texture + joint covariance) | 40% | 25% | 50% | 30% | 23% | 20% | 29% | 58% | 55% |
| Surface geometry | 25% | 35% | 18% | 32% | 37% | 39% | 37% | 17% | 6% |

Source: `attribution_conditional_summary.json` (per-permutation data in `shapley_conditional_permutations.json`).
