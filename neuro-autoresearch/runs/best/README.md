# Best models (`runs/best/`)

The single best `model.py` per `(dataset, subject-config)`, distilled from the gitignored
search runs by `runs/best/export_champions.py`. Each `model.py` is the champion's source
copied verbatim below a provenance header; regenerate with `python runs/best/export_champions.py`.

`test_corr` is comparable **only within the same (dataset, subjects) group** -- NSD/`nnn`
scores are noise-ceiling-normalized, `ieeg` is raw mean r, and different subject sets are
different rewards. See the top-level `CLAUDE.md` for the scoring details.

| Dataset | Subjects | test_corr | n_params | Model | Source run | Idea |
|---------|----------|-----------|----------|-------|------------|------|
| ieeg | 01,02,03 | 0.0630 | 198,528 | `plaxh` | ieeg__2026-07-07 | 0.0626 champion with informativeness-weighted context (content words 1.3x in context bag) |
| nnn | 1,2,3 | 0.2425 | 96,014 | `cosgm` | nnn__2026-07-15 | V4 curvature-at-angular-position shape code (Pasupathy-Connor conjunction) |
| nnn | 1,3,4 | 0.3057 | 558,674 | `nenlr` | nnn__2026-07-08 | retinal input transforms + compressive color-Gabor + up-weighted multi-scale HOG + foveal-zoom + color histograms + per-family gains |
| nsd | 1 | 0.4407 | 1,826,744 | `qq145` | nsd__2026-06-27 | abs (phase-invariant) nonlinearity at the c2 mid-level conv instead of ReLU |
| nsd | 1,2 | 0.4348 | 396,704 | `k7y09` | nsd__2026-06-29 | energy-augmented deep Gram: bind oriented Gabor energy with deep tower features in one covariance (cross-domain energy x deep terms) |
| nsd | 1,2,5,7 | 0.4192 | 90,000 | `svc3f` | nsd__2026-07-04 | Sweep partial-standardization exponent to std^-0.5 (beta=0.3 gave champion 0.4187) to find the peak |
