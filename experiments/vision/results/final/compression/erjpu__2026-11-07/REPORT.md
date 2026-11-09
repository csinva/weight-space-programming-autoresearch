# Natural-language compression (2026-11-07)

Claude cost: $3.21. Retention = (rebuild - gabor_norm) / (original - gabor_norm); CKA = linear CKA with the original's features on 1,000 LAION test images.

## laion__visual__encoding

| model | pooled | early | ventral | retention | CKA |
|---|---|---|---|---|---|
| gabor_norm | 0.430 | 0.543 | 0.310 | 0.00 | nan |
| original | 0.487 | 0.580 | 0.390 | 1.00 | 1.00 |
| rebuild_200w_0 | 0.447 | 0.549 | 0.339 | 0.29 | 0.71 |
| rebuild_200w_1 | 0.452 | 0.558 | 0.339 | 0.38 | 0.71 |
| rebuild_200w_2 | 0.454 | 0.557 | 0.345 | 0.42 | 0.76 |
| rebuild_200w_3 | 0.452 | 0.560 | 0.337 | 0.38 | 0.71 |

Rebuilds failing the contract check: none.

## description_200w

A fixed, parameter-free pipeline for square RGB images in 0–1. Apply a compressive gamma (offset 0.04, power 0.45), then convert to orthonormal luminance, red-green and blue-yellow channels. Also make a foveally magnified "cortical" copy, warping each axis by 0.58u plus 0.42u cubed.

Three octaves: 7×7 quadrature Gabors (8 orientations, 2 frequencies) plus two difference-of-Gaussians give energies and rectified DoG responses; chromatic channels use coarse Gabors. Each pathway is divisively normalized by local 3×3 plus half global mean energy, then log1p-compressed. Means pool on cortical grids (9, 6, 4 per side); standard deviations plus peaks on retinal grids.

Derived maps: orientation harmonics, curvature, surround pop-out; phase congruency, bar-versus-edge polarity; second-order modulation; off-diagonal channel correlations, global and per-quadrant; cross-octave persistence and differences; foveal-versus-peripheral contrast; mirror correlations.

Most heavily weighted: five-octave surface geometry. This includes Hessian shape-index bins weighted by normalized curvedness, structure-tensor eigenvalues and coherence, isophote curvature, and chromatic corner strength. It is pooled on shrinking grids with weights 1.5 down to 0.35, plus cross-octave correlations.

Extras: coarse color grids, anchor-distance path maps, local-extremum counts. Every block is separately L2-normalized and weighted; output totals roughly 10,000 dimensions.

