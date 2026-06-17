# Ventral-stream search — result of the hypothesis-driven phase (`test_corr = 0.3122`)

*Untrained models, NSD subject-1 ventral visual stream, ridge readout. This documents the
search that ran before the open-ended phase.*

## Goal & setup

- **Task.** Propose a **fixed, never-trained** model — a callable mapping a stimulus image
  `X` to activations — whose **ridge read-out** against recorded NSD brain activity has the
  highest held-out correlation. Only the ridge is fit; that fit *is* the evaluation.
- **Reward.** Mean per-voxel Pearson `test_corr` on the image-disjoint **shared-1000** test
  set, narrowed to the **ventral stream** (NSD `streams` atlas, **1820 voxels**, verified
  canonical and correctly masked).
- **The bar.** A frozen ImageNet-pretrained **AlexNet** (fc7), read out through the *same*
  ridge: **ventral 0.2838** (reproduced on the identical code path; himalaya cross-checked).
- **The target.** **1.5 × AlexNet = 0.4257**.
- **The ceiling.** Estimated **noise ceiling ≈ 0.50** (mean over ventral voxels; from
  all 10k images × 3 repeats, single-trial reliability `r₁≈0.127` → Spearman-Brown to the
  3-rep test target). So 0.4257 is **85% of the explainable variance** — extremely ambitious
  but, in principle, physically achievable.

## Result

| | ventral `test_corr` | vs AlexNet (0.2838) | vs ceiling (~0.50) |
|---|---|---|---|
| identity floor (raw pixels) | 0.1279 | — | 26% |
| **best untrained model** | **0.3122** | **+0.0284 (~+10%)** | 62% |
| trained AlexNet (bar) | 0.2838 | — | 57% |
| 1.5× target | 0.4257 | — | 85% |

**We beat trained AlexNet by ~10% with a never-trained model**, reaching **73% of the 1.5×
target**. The target was **not** reached in this phase.

## How it was achieved — the trajectory

The search ran as **dynamic-workflow evolutionary search**: each *generation* fans out K
candidate agents in parallel, each writing a complete `model.py`, evaluating it through the
fixed harness, and returning its score + per-region panel; between generations an **elite
archive** keeps the top models and seeds the next generation. ~25 models over 6 generations.

| stage | px | best ventral | what moved it |
|---|---|---|---|
| demo (2 gens) | 16 | 0.2467 | evolutionary search finds the texture-stats family |
| gen 1 | 16 | 0.2505 | depth + crossover |
| **gen 2** | **64** | **0.2817** | **input resolution 16→64px unlocked the deep families** |
| **gen 3** | **64** | **0.2966** | **ensembling complementary families (deep-conv ⊕ scattering)** |
| gen 4–4b | 128 | 0.3014 | canonical LN+divnorm cascade; resolution past 64 barely helps |
| gen 5 | 128 | 0.3071 | complementary-mechanism ensemble |
| **gen 6** | **128** | **0.3122** | **eccentricity-weighted readout (gain was early-cortex-driven)** |

**Two levers did almost all the work:**
1. **Input resolution (16 → 64px).** Four agents independently hit a "16×16 representational
   ceiling"; raising resolution (via a new operator knob `NEURO_IMSIZE`, which enriches the
   model input only — Y/readout/split/reward untouched) jumped the deep families from ~0.25
   to ~0.28. **Past 64px the gains nearly vanished** (128px gave only +0.003) — ventral
   became *compute-limited, not resolution-limited*.
2. **Ensembling complementary mechanisms.** Concatenating distinct untrained front-ends
   (deep random-conv + Morlet scattering + canonical cascade) beat any single one. This was
   the historical big lever — but by gen 6 it yielded only ~+0.006.

## The best model (`0.3122`)

An eccentricity-weighted read-out of a canonical untrained cascade: colour-opponent input →
LGN center-surround → V1 Gabor quadrature energy → divisive normalization → fixed
random-orthonormal compositional stages, read out through fixed analytic **foveal / mid /
peripheral Gaussian windows** plus foveal-magnification crops. **0 trained parameters**,
strictly per-sample (no test-set statistics — verified by grep + a clean re-run), train 0.422
/ test 0.312 (gap 0.11, no anomalous overfit). Promoted to `model.py` (needs
`NEURO_IMSIZE=128`); archived at `runs/evo-ventral/CHAMPION_ventral_0.3122.py`.

**Honest caveat:** although this was the top scorer, its **gain over the 0.3071 ensemble was
early-cortex-driven** (V1 0.47→0.50), *not* the high-level dissociation its hypothesis
predicted — see below.

## Per-region picture — where untrained wins and loses

| region | best untrained | trained AlexNet | |
|---|---|---|---|
| V1 | **0.50** | 0.34 | ✅ untrained *crushes* it |
| early / hV4 | **0.37 / 0.32** | 0.34 / 0.29 | ✅ |
| **ventral (reward)** | **0.312** | 0.284 | ✅ +0.028 |
| FFA / PPA / EBA | 0.29 / 0.32 / 0.32 | 0.40 / 0.40 / 0.44 | ❌ behind |
| **VWFA** | **0.22** | 0.32 | ❌ furthest behind |

The ventral-stream win is **driven by early/mid cortex**, where untrained oriented-energy +
divisive-normalization is demonstrably the right model. The **category-selective regions**
(FFA/PPA/EBA/VWFA) are where the trained network still leads — and where every untrained
manipulation stalled.

## The hypothesis-test ledger — what each idea taught

This phase was **hypothesis-driven**: each candidate implemented a *known* computational
theory of ventral cortex. The outcome, region-resolved:

| hypothesis | best ventral | verdict |
|---|---|---|
| oriented-energy / Gabor banks | ~0.30 | solves early/mid; plateaus |
| canonical LN + divisive-norm cascade | 0.304 | **right model for early/mid**; flat on high-level |
| deep random hierarchy (Gabor-anchored) | 0.301 | depth helps a little… |
| **depth to saturation (7 stages)** | 0.288 | **…SATURATES by ~3 stages — high-level NOT capacity-limited** |
| **eccentricity-bias (foveal/peripheral)** | **0.312** | **gain is early-cortex; FFA/PPA dissociation REFUTED** |
| texture / Gram statistics | 0.21–0.28 | position-invariant covariance **underperforms** |
| sparse / efficient coding | 0.300 | **sparsity monotonically HURTS** |
| V4 curvature / shape | 0.278 | no high-level lift |
| ensembling complementary mechanisms | 0.307 | gains from front-end diversity, diminishing |

## Scientific conclusions from this phase

1. **Untrained computation *solves* early/mid ventral.** V1 reaches 0.50 — far above trained
   AlexNet (0.34). The Carandini–Heeger canonical motif (oriented energy + divisive
   normalization) is demonstrably the right model there, no learning required.
2. **High-level category-selective cortex is a hard wall** for this class. FFA ~0.29,
   PPA ~0.32, EBA ~0.32, VWFA ~0.22 stayed flat across *every* manipulation — depth,
   eccentricity, texture, sparsity, ensembling.
3. **VWFA (visual word form) is hardest** (~0.22), consistent with it being the most
   explicitly *learned* representation in the ventral stream.

## The limitation that motivates the open-ended phase

Every candidate above was a **known neuroscience theory implemented top-down**. That confines
the search to the space of *already-understood* computations — and the "high-level needs
learning" conclusion is therefore only a statement about *known* untrained ideas, not a proof
of impossibility. The next phase removes that constraint: an **open-ended, score-guided
search** that may invent *any* untrained computation, judged only by held-out predictivity,
to test whether the ~0.31 plateau is real or merely the boundary of the ideas we thought to try.

*Invariants that remain (these make the score real, not restrictive): untrained (fixed/random
weights, the target Y is never seen) and per-sample (an image's activation depends only on
that image — no test-set leakage).*
