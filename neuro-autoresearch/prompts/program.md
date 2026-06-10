# program

Your `task` is to **propose a computational model of neural encoding** that
maximises the held-out predictivity reward (`test_corr`).

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

## The loop

Each iteration, follow these steps in order:

1. **Read `prompts/datasets.md`** to understand the active dataset (chosen by
   `NEURO_DATASET`): its stimulus/neural modalities, the dims of `X` and `Y`, the
   train/test split, and the sliceable metadata available.
2. **Read this dataset's history `results/<study>.csv`** (`<study>` = `dataset.study`,
   e.g. `results/NsdSubj01.csv`) *before* proposing — the full append-only log of every
   past iteration. Find `best_so_far`, and look at the `corr_<slice>` columns to see
   **where** prior models did well or poorly.
3. **Choose `LOG_SLICES` once, at the START of a run, and never change them.** Declare
   `LOG_SLICES = [...]` in `experiment.py`, picking names from the
   active dataset's slice menu in `prompts/datasets.md` (for `nsd`, ROIs like
   `["V1","V2","V3","hV4","FFA","PPA"]`; for `text`/`meg`, none yet → `[]`). The
   harness writes one `corr_<name>` column per slice and **freezes those columns on the
   first logged row** — changing `LOG_SLICES` mid-run does nothing, so they are your
   stable comparison axes across the whole run. Choose them deliberately.
4. **Set a one-line `idea` each iteration** — a free-text hypothesis of what you
   changed and why (set `model.idea`, or the model `description`). It is logged
   verbatim as the `idea` column.
5. **Propose / edit the `Model`** in `model.py` aimed at **beating
   `best_so_far`**, using the `corr_<slice>` columns to decide *where* to push (e.g.
   for `nsd`, more high-level visual structure if `FFA`/`PPA` lag, more spatial
   detail if early-visual `V1`–`hV4` lag). Before re-running, **snapshot the new
   `model.py` into `models_lib/`** (e.g. `cp model.py models_lib/attempt-NN.py`) so
   every design in the trajectory is preserved next to its score.
6. The fixed harness runs the stimulus through your model -> activations, linearly
   reads them out (ridge regression) to predict recorded neural activity, and appends
   one row (the CORE columns + your `corr_<slice>` columns) to `results/<study>.csv`. The
   held-out mean per-target Pearson `r` (`test_corr`) is the reward.

So `model.py` is the artifact you evolve; `test_corr` is the verifiable
reward. Iterate to raise it, hill-climbing past `best_so_far`.
