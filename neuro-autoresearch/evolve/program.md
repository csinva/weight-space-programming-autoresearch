# program

Your `task` is to **propose a computational model of neural encoding** that
maximises the held-out predictivity reward (`test_corr`).

## What you may edit

You may edit **only `evolve/model.py`**. Everything in `harness/` (the dataset,
the data loader, and the evaluator) is FIXED — do not touch it.

## The rules

1. A model is a plain callable: stimulus batch `X (N, D_in)` -> activations
   `(N, D_out)`. Write its weights **directly in code**.
2. The model is **never trained** — no gradient descent, no optimizer, no
   backprop. Only the ridge readout inside `harness/eval.py` is fit, and that is
   the evaluation, not part of your model.
3. Keep `model.py` self-contained and runnable: `python evolve/model.py`.

## The loop

1. You propose / edit a `Model` in `evolve/model.py`.
2. The fixed harness runs the stimulus through your model -> activations.
3. Activations are linearly read out (ridge regression) to predict recorded
   neural activity. The held-out mean per-target Pearson `r` (`test_corr`) is the
   reward, appended to `runs/overall_results.csv`.

So `evolve/model.py` is the artifact you evolve; `test_corr` is the verifiable
reward. Iterate to raise it.
