# Open-ended search (lab/ engine) — distilled results (`test_corr = 0.325`)

*Untrained models, NSD subject-1 ventral visual stream, ridge readout, `NEURO_IMSIZE=128`.
A 7-hour run of the decoupled propose/evaluate engine: Claude Code's dynamic Workflow
proposes models in parallel; the Copilot CLI evaluates the queue sequentially; `lab/results.csv`
is the shared memory.*

## Headline

- **240 models evaluated** in 16 iterations (**237 ok, 3 error** — 1.25%, all import/syntax
  failures caught by `record.py`).
- **Best untrained model: ventral `test_corr` = 0.3250** (`m_32507161a8`, iter 16) — beats a
  *trained* AlexNet (0.2838) by **+0.041 (~14%)** with **zero trained parameters**.
- vs the **1.5× target 0.4257** → **76%**; vs the **noise ceiling ~0.50** → **65%**. Target not reached.

## The champion (`m_32507161a8` ≈ `m_ddcf4cd4e0`, tied within noise)

A **five-stream fusion** of analytic, untrained computations on a **log-polar
(cortical-magnification) + steerable-pyramid** front-end, read out through a tuned log-polar grid:

1. **Richer Portilla–Simoncelli texture** statistics (steerable pyramid; cross-scale/orientation correlations).
2. **Mid-level shape** — curvature, end-stopping, co-circularity, proto-object **closure / boundary-completion**.
3. **Global-form eccentricity-ring profile** (log-polar geometry).
4. **Magnocellular center-surround** (DoG on/off).
5. **All three cross-stream conjunctions** under shared surround divisive normalization:
   **texture-on-figure**, **cross-orientation junctions** (energy × orthogonal energy → corners/T-/L-junctions),
   and **color-opponent** (luminance energy × chroma context).

The decisive mechanism is stream 5 — **multiplicative feature *interactions***. All weights are
analytic (steerable filters, DoG) or fixed geometry; normalization is strictly per-image; the
target `Y` is never read. Readout = spatial-pyramid `[1,2,3,5]` + a log-polar rings×sectors grid,
sized under the ridge SVD limit. Full source: `analyze/CHAMPION_ventral_0.3250.py` (= `model.py`).
Verified per-sample + untrained (no cross-batch statistics, no fit/backprop). train 0.417 / gap 0.093.

## Per-region (champion vs trained AlexNet)

| region | champion | AlexNet | |
|---|---|---|---|
| V1 | **0.506** | 0.342 | ✅ +0.16 |
| early | **0.373** | 0.342 | ✅ |
| hV4 | **0.334** | 0.289 | ✅ |
| **ventral (reward)** | **0.325** | 0.284 | ✅ **+0.041** |
| FFA | 0.285 | 0.396 | ❌ |
| PPA | 0.313 | 0.403 | ❌ |
| EBA | 0.312 | 0.436 | ❌ |
| VWFA | 0.215 | 0.321 | ❌ (worst) |

Untrained **solves early/mid cortex** and **loses every category-selective region** — VWFA hardest.

## Trajectory (best per iteration)

| iters | best | what drove it |
|---|---|---|
| 1 | 0.299 | strong seed: log-polar / PS-texture front-ends |
| 2–3 | 0.275→0.284 | exploration dip (new families tried & dropped) |
| 4–9 | 0.306→**0.321** | the real climb: **multi-stream fusion + cross-stream conjunctions** |
| 10–16 | 0.323→**0.325** | plateau: **readout-granularity tuning** (log-polar grid sweet spot); gains <0.005 = noise |

Three themes in order: **(a)** log-polar + steerable-pyramid texture front-end set the floor; **(b)**
gains to ~0.32 came from stacking **multiplicative conjunction terms**; **(c)** the last ~0.003 from
**readout spatial granularity** (the search found the sweet spot itself: moderate finer log-polar
helps; cartesian, over-dense, and per-region pooling all *worse* — the ridge overfits as D_out grows).
Saturated by iter ~10.

### Top 10 (all status=ok)

| id | iter | ventral |
|---|---|---|
| m_32507161a8 | 16 | 0.3250 |
| m_333598497c | 16 | 0.3248 |
| m_ddcf4cd4e0 | 15 | 0.3248 |
| m_fa39b88cd4 | 15 | 0.3248 |
| m_7fa462aa42 | 13 | 0.3247 |
| m_a97ae4769d | 14 | 0.3247 |
| m_8582eb0bef | 14 | 0.3247 |
| m_85d774779d | 14 | 0.3246 |
| m_2a9b6b835c | 12 | 0.3244 |
| m_19e5a53063 | 13 | 0.3244 |

## Verdict — the untrained ceiling, triangulated

| search method | best ventral | % of ceiling |
|---|---|---|
| hypothesis-driven (top-down theories) | 0.312 | 62% |
| open-ended (free invention) | 0.321 | 64% |
| **this efficient propose/eval engine** | **0.325** | **65%** |

Three independent methodologies converge on **~0.32** — and on the same *mechanism*: this run
**independently rediscovered** that the predictive untrained computation is **multiplicative feature
interactions / conjunctions on a texture-form backbone**, not deeper linear cascades. The convergence
makes **~0.32 a robust untrained-feedforward ceiling for `streams`-ventral**: early/mid cortex is fully
accounted for untrained (V1 0.50 ≫ AlexNet 0.34), while the **category-selective regions
(FFA/PPA/EBA/VWFA)** hold the line — their residual variance encodes **learned** structure (category,
identity, word form) that no untrained feedforward computation reached across three search regimes.
**1.5× AlexNet (0.426) almost certainly requires training; ~0.32 is the untrained frontier for this ROI.**

*Engine: `run.py` + `lab/{propose.js,evaluate.sh,record.py}`. The full per-model database (240 models
with inline source) lives in the gitignored `lab/results.csv`.*
