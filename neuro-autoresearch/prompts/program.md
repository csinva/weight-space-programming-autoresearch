# program

Your `task` is to **propose a computational model of neural encoding** that
maximises the held-out predictivity reward (`test_corr`) **for the active reward
region** (chosen by `NEURO_ROI`, e.g. `FFA`). The readout is fit over all voxels and
every region's predictivity is logged, but your objective is the one region's score.

## What you may edit

You may edit **only `model.py`** (the model) and **`experiment.py`** (the
run-config: `ITERATIONS`, `LOG_SLICES`). Everything in `harness/` (the dataset,
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
5. **Report `n_params` and a concise `idea` every iteration.** Set `model.n_params`
   to the actual count of hardcoded weights you wrote (0 for a parameter-free
   transform — be honest), and `model.idea` to a **single concise sentence** naming
   what you changed this iteration. Both are logged to the CSV.

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
4. **Set `model.idea` (one concise sentence) and `model.n_params`** each iteration —
   `idea` is your one-line hypothesis of what you changed and why (logged verbatim as
   the `idea` column); `n_params` is the honest count of hardcoded weights (logged
   too). Keep `idea` to a single sentence.
5. **Propose / edit the `Model`** in `model.py` aimed at **beating
   `best_so_far`** on the reward region, using the `corr_<region>` columns to decide
   *where* to push (e.g. more high-level visual structure if `FFA`/`PPA`/`ventral` lag,
   more spatial detail if `V1`/`hV4`/`early` lag). Before re-running, **snapshot the new
   `model.py` into `models_lib/`** (e.g. `cp model.py models_lib/attempt-NN.py`) so
   every design in the trajectory is preserved next to its score.
6. The fixed harness runs the stimulus through your model -> activations, linearly
   reads them out (ridge regression) to predict recorded neural activity, and appends
   one row (the CORE columns + the fixed `corr_<region>` panel) to the run's CSV. The
   held-out mean per-target Pearson `r` (`test_corr`) over the reward region is the reward.

So `model.py` is the artifact you evolve; `test_corr` is the verifiable
reward. Iterate to raise it, hill-climbing past `best_so_far`.
