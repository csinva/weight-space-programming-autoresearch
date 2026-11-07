# autoresearch — interpretable transformers for fMRI language encoding

This is an experiment to have a coding agent autonomously research how to
hand-write the weights of a small transformer (ideally with human-understandable
structure) so that its embeddings predict **fMRI responses to language** as well
as possible. The reference point is a pretrained **GPT-2 XL** baseline.

The model is NEVER trained. The agent must write the weights directly into
`interpretable_transformer.py`.

## The encoding task

We use the Huth natural-language fMRI dataset. For every word a **10-gram** (the
word plus the preceding words) is embedded by the model; the final-token hidden
state is the feature for that word. Features are Lanczos-downsampled to the fMRI
TR timeline, z-scored, FIR-delayed, and a **ridge** model is fit to predict voxel
responses. The metric is the **mean test-set correlation** (`test_corr`) between
predicted and held-out actual responses, averaged over all voxels.

The fixed pipeline lives in `src/` (**do not edit**, you don't need to read these).

## Setup

The work happens inside a fresh **run folder**. Each run is an isolated copy
under `<repo_root>/runs-neuro/<tag>/`.

1. **Pick a unique run tag** based on today's date and a counter, e.g. `fmri-may27-run1`.
2. **Create the run folder**: `uv run setup_run.py <tag>`. This creates `runs-neuro/<tag>/`:
   - `src/` — **symlink** back to `evolve-neuro/src` (do not modify via the symlink)
   - `interpretable_transformer.py` — **local copy**, this is the file you edit
   - `results/` — stores your runs; seeded with the GPT-2 XL baseline row + plot
   - `interpretable_transformers_lib/` — empty, for snapshots

   `setup_run.py` runs the GPT-2 XL baseline through the identical pipeline when
   it is missing (the default on a fresh folder), so `results/overall_results.csv`
   starts with one `GPT2XL-baseline` row to beat. (`--skip-baseline` /
   `--force-baseline` override this.)
3. **`cd` into the run folder** and stay there: `cd runs-neuro/<tag>`. Only read your own run folder.
4. **Read the in-scope files**: `interpretable_transformer.py`, `src/eval.py`, `src/features.py`, and `results/overall_results.csv`.

## Experimentation

Each experiment runs on a single GPU. Launch it from inside the run folder:
`uv run interpretable_transformer.py --subject UTS03 --num-train 5`.

**What you CAN do:**

- Edit `interpretable_transformer.py` in the run folder (except the evaluation
  harness in the **main** block at the bottom):
  - The `SimpleTransformer`, `Block`, `CausalSelfAttention`, `MLP` architectures.
  - The `InterpretableEmbedder` tokenization / pooling and the `VOCAB`.
  - The `write_weights(model)` function — set any parameter tensors directly:
    hardcoded constants, NumPy arrays, hand-built attention circuits, lookup
    tables — anything, as long as you do not train.
  - The architecture hyperparameters (depth, width, heads, ff size, seq len).
  - The `model_shorthand_name` and `model_description` strings.
- Save snapshots of `interpretable_transformer.py` under `interpretable_transformers_lib/`.

**What you CANNOT do:**

- Run any training, gradient descent, optimizer step, fitting loop, or backprop.
  All parameters must be set in closed form.
- Load or read weights from any pretrained model (that's what the baseline is for).
- Use external tools/libraries to compute the embedding — it must come from the
  transformer forward pass.
- Modify anything reached through a symlink: anything in `src/` or `program.md`.

**The goal is simple: maximize `test_corr`** — the mean held-out voxel
correlation. The GPT-2 XL baseline row is the target to approach/beat.

**Interpretability criterion**: all else being equal, more interpretable is
better. A small improvement that adds ugly complexity is not worth it. Parameter
count is a rough proxy, but the real criterion is human-understandable structure.

## Output format

When the script finishes it prints a summary and updates
`results/overall_results.csv` with columns:

```
subject,test_corr,train_corr,frac_test_voxels_above_0.2,encoding_seconds,status,model_shorthand_name,n_params,description,roi_Broca,roi_AC,roi_sPMv,roi_EBA,roi_FFA,roi_PPA,roi_RSC,roi_IPS
```

1. subject (e.g. `UTS03`)
2. `test_corr` — **primary metric**, mean held-out voxel correlation (empty for crashes). The other metrics are secondary and just give helper information, this is the metric to optimize.
3. `train_corr` — mean in-sample (training) voxel correlation (gauges overfitting)
4. `frac_test_voxels_above_0.2` — fraction of test voxels with correlation > 0.2
5. `encoding_seconds` — wall-clock time of the encoding pipeline (features + ridge fit)
6. status: `success` or `crash`
7. shorthand unique name of the model attempt
8. total number of model parameters (`sum(p.numel() for p in model.parameters())`)
9. brief description of what this attempt tried
10. `roi_*` — mean test correlation within popular language/semantic ROIs (Broca's
    area, auditory cortex, sPMv, EBA, FFA, PPA, RSC, IPS); blank if unavailable for
    the subject

It also refreshes `results/corr_over_iterations.png`.

## The experiment loop

You are always inside the run folder (`runs-neuro/<tag>/`).

LOOP FOREVER:

1. Edit `interpretable_transformer.py` with one experimental idea. Update
   `model_shorthand_name` (unique within this run) and `model_description`.
2. Run the experiment: `uv run interpretable_transformer.py > run.log 2>&1`
3. Read results: `tail -n 15 run.log` and `grep <model_shorthand_name> results/overall_results.csv`
4. If it crashed, check `tail -n 50 run.log` for the stack trace and fix it.
5. Update the row in `results/overall_results.csv` with the appropriate status.
6. Save a snapshot as `interpretable_transformers_lib/<model_shorthand_name>.py`.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human whether to
continue. Run until manually stopped. Even if you match the baseline, keep going
to find a simpler / more interpretable solution. Do not stop before 30 iterations.

**Example ideas to try** (not exhaustive — be creative):

- Try implementing primitives from the neurosci/cogsci literature
- Try inducing hierarchies of features in different ways
- You might want to emphasize the final token / tokens near the end since the fMRI signal is more likely to reflect recent words.
- Use the MLP layers as lookup tables mapping character patterns to semantic axes that brain language regions are known to track (e.g. word length, concreteness).
- Make `token_emb` encode something meaningful per character (e.g. orthographic features) so the final-token state reflects the recent letters/word.
- Use attention to average / select context across the ngram (a "bag of chars" or "last-word" circuit) rather than just reading the last character.

BE CREATIVE.
Try out-of-the-box, diverse ideas.
