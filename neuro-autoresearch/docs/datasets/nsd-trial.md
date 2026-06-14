# NSD 2-Iteration Trial — Loop Mechanics Validation

> **Historical trial — superseded.** This early run predates the current NSD
> pipeline. Paths/details below are stale: the model is now repo-root `model.py`
> (not `evolve/model.py`), scoring lives in `harness/reward.py` (not
> `harness/eval.py`), and the NSD split is now the HRF-offset (`neural_offset=5.0`),
> repeat-collapsed, image-disjoint **shared-1000** hold-out — not the old seeded
> 200/50 whole-volume split. Read it for the loop-mechanics narrative only.

A short end-to-end trial of the AlphaEvolve-style autoresearch loop on the
Natural Scenes Dataset (NSD). The loop proposes a **fixed, never-trained** model
in `evolve/model.py`; the harness (`harness/eval.py`) fits **only** an L2 ridge
readout from the model's activations to recorded fMRI and reports the held-out
mean per-voxel Pearson r (`test_corr`) as the verifiable reward.

> **Note:** these are measured numbers from a completed 2-iteration trial.
> Nothing was re-run to produce this report.

## Dataset

| Property | Value |
| --- | --- |
| Study | `Allen2022MassiveSample` (NSD), fMRI |
| Stimuli → X | 250 × 768 (16×16×3 raw RGB pixels, `imsize=16`) |
| Neural → Y | 250 × 31448 voxels (whole-volume) |
| Split | 200 train / 50 test, `seed=0` |
| Segment | `start=0.0`, `duration=3.0s` |
| Local sample | 622 MB pre-downloaded (225 stimulus PNGs + 4 fMRI runs), `download=False` |

The NSD sample is read from a local path (`harness/config.py`), so no aws CLI or
NSD terms-of-service acceptance is needed.

## Timing: the prediction is sub-second; data loading dominates

The reported ~9s "eval" and ~13s wall time per iteration is **almost entirely
one-off data loading**, not the linear prediction. Measured with a single shared
data load:

| Stage | Time |
| --- | --- |
| `load_split` (data + segmentation + fMRI extraction) | **9.17s** — dominates |

The ridge readout itself — the actual evaluation — is sub-second:

| Model | forward | ridge.fit | ridge.predict | activation dim (A_dim) |
| --- | --- | --- | --- | --- |
| `identity` | 0.2 ms | 300.7 ms | 20.3 ms | 768 |
| `random_projection` | 103.0 ms | 83.3 ms | 13.4 ms | 64 |
| `spatial_pyramid` | 98.1 ms | 98.5 ms | 34.8 ms | 63 |

**Takeaways:**
- The fit + predict cost is **80–335 ms total** per model — trivial next to the
  9.17s load that happens once per iteration.
- `ridge.fit` scales with the activation dimension: 768 dims → ~300 ms, vs ~64
  dims → ~83 ms (the ridge solve cost grows with feature count).
- `identity` has a ~0 ms forward (it passes pixels through), while the other two
  pay ~100 ms to compute their fixed transform.

## Models

Each model is a fixed callable `X (N, D_in) → activations (N, D_out)`; weights
live in code and are never trained.

| Model | n_params | Transform |
| --- | --- | --- |
| `identity` | 0 | Raw 768 input pixels, no transform |
| `random_projection` | 49152 | ReLU of a fixed `seed=0` Gaussian projection to 64 dims |
| `spatial_pyramid` | 0 | Fixed mean-pool of the image at 4×4 + 2×2 + 1×1 grids per channel → 63 dims |

(`random_projection`'s 49152 = 768 × 64 projection matrix is fixed/seeded, not
learned.)

### Overfitting

`identity` exposes all 768 pixel dimensions to the ridge readout with only 200
training samples, and **overfits badly**: train_corr = 0.9202 but test_corr =
0.0417. The high-dimensional readout memorizes the train set and generalizes
poorly. The 64-d random bottleneck and the 63-d pyramid both have far lower train
correlation (0.45 and 0.38) and score near chance on the held-out set.

## Reward progression

| Iteration | Model | test_corr | train_corr | n_params | eval | wall |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `random_projection` | 0.0382 | 0.4467 | 49152 | 9.08s | 13.27s |
| 1 | `identity` | **0.0417** | 0.9202 | 0 | 9.19s | 13.39s |
| 2 | `spatial_pyramid` | 0.0202 | 0.3804 | 0 | 9.34s | 13.70s |

Iteration 1 evaluated two starting proposals. Iteration 2 made **one
modification** — the `spatial_pyramid` transform — intended to suit early-visual
fMRI signal with smooth low-variance features.

**The modification regressed.** `spatial_pyramid` (0.0202) scored *worse* than
both iteration-1 proposals. The framework's verifiable reward correctly flags the
change as a step backwards: in a continued loop the agent would **revert** this
modification and try a different direction rather than build on it. The current
best remains `identity` at **0.0417**.

## Caveats

All correlations are small, and that is expected for this trial:
- **Tiny sample:** only 250 NSD stimuli (200 train / 50 test).
- **Coarse stimuli:** 16×16 RGB pixels discard most visual detail.
- **Whole-volume targets:** 31448 voxels span the whole volume, most of which are
  not visually driven, so mean per-voxel r is diluted toward zero.

This trial therefore validates the **loop mechanics end-to-end** — propose a
fixed model, fit only the ridge readout, score held-out predictivity, and let the
verifiable reward accept or reject the modification — and **not** encoding
performance. The numbers confirm the pipeline runs and the reward behaves
correctly (rejecting a regression); they are not a claim about NSD encoding
quality.
