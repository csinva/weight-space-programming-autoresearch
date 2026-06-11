# program

Your `task` is to **propose a computational model of neural encoding** that
maximises the held-out predictivity reward (`test_corr`) **for the active reward
region** (chosen by `NEURO_ROI`, e.g. `FFA`). The readout is fit over all voxels and
every region's predictivity is logged, but your objective is the one region's score.

**The goal is to BEAT a trained AlexNet.** A frozen ImageNet-pretrained AlexNet
(fc7 features) read out with this exact ridge is the benchmark — see the per-region
"Trained-AlexNet reference baseline" in `prompts/datasets.md`. Your model is **never
trained** (weights written directly in code), yet the bar to clear is a network that
*was* trained on a million labelled images. Design a computational model whose
`test_corr` for your reward region matches or exceeds that trained-AlexNet number;
the identity/raw-pixel floor is merely the starting point, not the target.

## What you may edit

You may edit **only `model.py`** (the model) and **`experiment.py`** (the
run-config: `LOG_SLICES`). Everything in `harness/` (the dataset,
the data loader, and the evaluator) is FIXED — do not touch it.

## Where you work (isolated run folder)

You work inside a single **run folder** created by `python scripts/setup_run.py
<tag>` (which makes `runs/<tag>/`). `cd runs/<tag>` and stay there — no git branch,
no commits, everything is local to the folder:

- `harness/` and `prompts/` are **symlinks** to the FIXED repo code — read them,
  never edit through them.
- `model.py` and `experiment.py` are your **local, editable** copies. Edit these.
- `results/<study>.csv` (written here) is this run's append-only history.
- `models_lib/` is where you **snapshot every attempt's `model.py`** before
  re-running, e.g. `cp model.py models_lib/attempt-$(printf '%02d' N).py`, so the
  full trajectory of designs is recoverable alongside the score history.

(You can also run the single-tree way from the repo root with `python -m
harness.run`; the run-folder workflow is the same loop, just isolated.)

## The rules

1. A model is a plain callable: stimulus batch `X (N, D_in)` -> activations
   `(N, D_out)`. Write its weights **directly in code**.
2. The model is **never trained** — no gradient descent, no optimizer, no
   backprop. Only the ridge readout inside `harness/reward.py` is fit, and that is
   the evaluation, not part of your model.
3. Keep `model.py` self-contained and runnable: `python -m harness.run`.
4. **Favor simplicity.** Prefer the simplest model that predicts well: reach for a
   clear mechanistic idea before brute-force capacity. A smaller model that matches a
   bigger one's score is better (it is also far faster to evaluate). Avoid needlessly
   huge random feature banks.
5. **Report `n_params`, `idea`, and `summary` every iteration.** Set `model.n_params`
   to the actual count of hardcoded weights you wrote (0 for a parameter-free
   transform — be honest); `model.idea` to your forward-looking hypothesis for this
   iteration; and `model.summary` to what the model actually does. Keep `idea` and
   `summary` each to **at most two extremely dense, information-heavy sentences**. All
   are logged to the CSV.
6. **Evaluate every candidate through `python -m harness.run`** — one run = one model =
   one CSV row. Do **not** write side scripts that import the harness directly or batch
   many models off-ledger: the append-only CSV (and its `best_so_far`) is the record of
   record, and an attempt that is not run through the harness did not happen.
7. **Mind the GPU — it is SHARED across several parallel runs.** Keep activations modest
   (a few GB, roughly `N x D_out` float; the readout adds an `N x N`/`D x D` SVD): avoid
   huge random feature banks. A `CUDA out of memory` is logged as a `status="error"` row
   (NaN reward) and wasted; if you hit one, shrink `D_out`/the feature bank and retry.

## The loop

Each iteration, follow these steps in order:

1. **Read `prompts/datasets.md`** to understand the active dataset (chosen by
   `NEURO_DATASET`): its stimulus/neural modalities, the dims of `X` and `Y`, the
   train/test split, and the reward region.
2. **Read this run's history CSV** — `results/<study>.csv`, or for a per-region NSD
   loop `results/<study>__<reward_roi>.csv` (e.g. `results/NsdSubj01__ventral.csv`) —
   *before* proposing: the full append-only log of every past iteration. Find
   `best_so_far`, and read the `corr_<region>` columns to see **where** prior models
   did well or poorly.
3. **The reward region is FIXED for the run** (chosen by the operator via `NEURO_ROI`,
   default `ventral` for NSD) — do NOT change the objective. You do **not** declare
   `LOG_SLICES` for NSD: the harness automatically logs its own fixed region panel
   (`V1, hV4, FFA, PPA, EBA, VWFA, early, ventral`) as `corr_<region>` columns, frozen
   on the first row. The scalar reward is `test_corr` over the chosen region; the
   readout still fits all voxels, so every region's predictivity is logged for you.
4. **Set `model.idea`, `model.summary`, and `model.n_params`** each iteration —
   `idea` is your hypothesis of what you changed and why (logged verbatim as the
   `idea` column); `summary` is what the model actually does (logged as the `summary`
   column); `n_params` is the honest count of hardcoded weights (logged too). Keep
   `idea` and `summary` each to at most two dense, information-heavy sentences.
5. **Propose / edit the `Model`** in `model.py` aimed at **beating
   `best_so_far`** on the reward region, using the `corr_<region>` columns to decide
   *where* to push (e.g. more high-level visual structure if `FFA`/`PPA`/`ventral` lag,
   more spatial detail if `V1`/`hV4`/`early` lag). Treat a `test_corr` change **smaller
   than ~0.005 as a tie** (it is within reward noise, not a real gain); if `best_so_far`
   has not moved by more than that for several iterations, **switch to a different model
   family** rather than keep micro-tuning the current one. Before re-running, **snapshot
   the new `model.py` into `models_lib/`** (e.g. `cp model.py models_lib/attempt-NN.py`)
   so every design in the trajectory is preserved next to its score.
6. The fixed harness runs the stimulus through your model -> activations, linearly
   reads them out (ridge regression) to predict recorded neural activity, and appends
   one row (the CORE columns + the fixed `corr_<region>` panel) to the run's CSV. The
   held-out mean per-target Pearson `r` (`test_corr`) over the reward region is the reward.

So `model.py` is the artifact you evolve; `test_corr` is the verifiable
reward. Iterate to raise it, hill-climbing past `best_so_far`.
