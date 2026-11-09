# Computational-family importance: search champion erjpu (LAION search data)

Scope: LAION-fMRI, 5 subjects, pooled early+ventral encoding readout identical to search (per-voxel 5-fold-CV ridge).
Families = groups of the champion's 36 feature blocks (see attribution.py). NSD not used. Values x1000 (Pearson r).
Caveat: these LAION test images were optimised against by ~600 search candidates; magnitudes may be inflated.

## Scores

| Family | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| Stage 1 (V1-like energy) alone | 0.443 | 0.543 | 0.337 | 0.565 | 0.546 | 0.530 | 0.502 | 0.254 | 0.401 |
| Full champion | 0.487 | 0.580 | 0.390 | 0.608 | 0.579 | 0.565 | 0.544 | 0.328 | 0.445 |
| Gain over Stage 1 (x1000) | 44.3 | 36.7 | 52.3 | 43.7 | 32.4 | 34.9 | 41.3 | 73.6 | 44.3 |

## Shapley share of the gain over Stage 1 (Stage 1 always present; 60 permutations; ± SE)

| Family | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| surface_geometry | 11.2 ± 0.7 | 12.9 ± 0.8 | 9.5 ± 0.7 | 14.2 ± 1.0 | 11.9 ± 0.7 | 13.5 ± 0.8 | 15.4 ± 0.8 | 12.5 ± 0.9 | 2.8 ± 0.5 |
| global_layout_symmetry | 8.0 ± 0.7 | 7.8 ± 0.8 | 8.1 ± 0.6 | 8.5 ± 0.9 | 6.8 ± 0.7 | 8.7 ± 0.8 | 8.9 ± 0.8 | 9.1 ± 0.7 | 8.4 ± 0.6 |
| cross_octave | 5.2 ± 0.7 | 3.3 ± 0.7 | 7.1 ± 0.8 | 4.4 ± 1.0 | 2.8 ± 0.6 | 2.7 ± 0.6 | 2.6 ± 0.5 | 10.0 ± 1.3 | 8.0 ± 0.8 |
| geometry_relations | 4.8 ± 0.6 | 1.7 ± 0.5 | 8.1 ± 0.7 | 3.0 ± 0.7 | 1.2 ± 0.4 | 0.9 ± 0.3 | 3.3 ± 0.3 | 12.2 ± 1.2 | 8.6 ± 0.7 |
| texture_covariance | 4.6 ± 0.5 | 1.8 ± 0.4 | 7.6 ± 0.5 | 2.4 ± 0.6 | 1.5 ± 0.4 | 1.6 ± 0.3 | 4.3 ± 0.4 | 13.9 ± 1.0 | 4.6 ± 0.4 |
| phase_congruency | 3.0 ± 0.5 | 3.5 ± 0.6 | 2.5 ± 0.5 | 4.2 ± 0.7 | 3.2 ± 0.5 | 3.0 ± 0.6 | 2.4 ± 0.5 | 1.4 ± 0.5 | 3.0 ± 0.4 |
| joint_covariance | 2.9 ± 0.5 | 2.3 ± 0.5 | 3.6 ± 0.5 | 3.1 ± 0.7 | 2.1 ± 0.4 | 1.7 ± 0.3 | 1.6 ± 0.3 | 6.4 ± 1.0 | 3.4 ± 0.4 |
| orientation_field | 2.2 ± 0.3 | 1.3 ± 0.2 | 3.1 ± 0.4 | 1.2 ± 0.2 | 1.4 ± 0.2 | 1.2 ± 0.3 | 0.3 ± 0.2 | 5.5 ± 0.6 | 4.1 ± 0.4 |
| second_order | 1.5 ± 0.3 | 1.4 ± 0.3 | 1.6 ± 0.2 | 1.4 ± 0.4 | 1.3 ± 0.3 | 1.7 ± 0.3 | 2.2 ± 0.3 | 2.1 ± 0.3 | 0.2 ± 0.1 |
| color_layout | 0.8 ± 0.0 | 0.5 ± 0.0 | 1.1 ± 0.1 | 1.3 ± 0.1 | 0.2 ± 0.0 | -0.1 ± 0.0 | 0.3 ± 0.0 | 0.5 ± 0.0 | 1.1 ± 0.1 |

## Unique contribution: drop when the family is removed from the full model (x1000; * = 95% image-bootstrap CI excludes 0; (k/5) subjects positive)

| Family | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| surface_geometry | 6.6* (5/5) | 8.1* (5/5) | 5.0* (5/5) | 8.4* (5/5) | 7.6* (5/5) | 9.1* (5/5) | 10.1* (5/5) | 6.9* (5/5) | -0.1 (3/5) |
| global_layout_symmetry | 1.9* (5/5) | 1.4* (5/5) | 2.5* (5/5) | 1.1* (4/5) | 1.3* (5/5) | 1.8* (5/5) | 1.5* (5/5) | 3.0* (5/5) | 3.8* (5/5) |
| cross_octave | 0.5* (4/5) | -0.4* (0/5) | 1.5* (5/5) | -0.6* (0/5) | -0.4* (0/5) | -0.2 (1/5) | -0.0 (4/5) | 1.5 (5/5) | 2.3* (5/5) |
| geometry_relations | 1.8* (5/5) | -0.0 (2/5) | 3.7* (5/5) | 0.0 (3/5) | -0.2 (2/5) | 0.1 (3/5) | 1.8* (5/5) | 5.3* (5/5) | 4.2* (5/5) |
| texture_covariance | 1.8* (5/5) | -0.2 (2/5) | 3.8* (5/5) | -0.2 (2/5) | -0.3 (1/5) | -0.0 (2/5) | 1.9* (5/5) | 7.5* (5/5) | 2.2* (5/5) |
| phase_congruency | 0.0 (3/5) | 0.1 (4/5) | -0.1 (2/5) | 0.3* (4/5) | 0.1 (3/5) | -0.2 (3/5) | -0.4 (2/5) | -0.8 (1/5) | 0.4 (4/5) |
| joint_covariance | 0.2* (5/5) | -0.1* (1/5) | 0.5* (5/5) | -0.1* (1/5) | -0.1* (1/5) | -0.0 (1/5) | 0.1 (5/5) | 0.6* (4/5) | 0.7* (5/5) |
| orientation_field | 0.2 (3/5) | -0.2 (0/5) | 0.7* (5/5) | -0.2 (2/5) | -0.2 (0/5) | -0.4* (1/5) | -0.8* (0/5) | 1.6* (3/5) | 1.3* (4/5) |
| second_order | 0.1 (3/5) | -0.2* (1/5) | 0.3 (4/5) | -0.3* (0/5) | -0.2* (2/5) | 0.1 (3/5) | 0.3* (4/5) | 0.3 (3/5) | -0.2 (2/5) |
| color_layout | 0.5* (5/5) | 0.4* (4/5) | 0.7* (5/5) | 0.9* (5/5) | 0.2* (4/5) | -0.0 (2/5) | 0.2 (4/5) | 0.4 (4/5) | 0.6 (4/5) |
| stage1_V1_energy | 7.6* (5/5) | 14.7* (5/5) | 0.1 (2/5) | 16.1* (5/5) | 15.7* (5/5) | 12.1* (5/5) | 3.8* (3/5) | -2.1 (1/5) | 0.3 (3/5) |

## Grouped

| Family | Pooled | Early | Ventral | V1 | V2 | V3 | hV4 | FFA | PPA |
|---|---|---|---|---|---|---|---|---|---|
| Correlation statistics (cross-octave + geometry relations + texture cov + joint cov), % of gain | 40% | 25% | 50% | 30% | 23% | 20% | 29% | 58% | 55% |
| Surface geometry, % of gain | 25% | 35% | 18% | 32% | 37% | 39% | 37% | 17% | 6% |
