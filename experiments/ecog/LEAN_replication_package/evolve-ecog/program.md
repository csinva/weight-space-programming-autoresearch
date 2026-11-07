# autoresearch — interpretable audio encoder for patient1 ECoG high-gamma

This is an experiment to have a coding agent autonomously research how to
hand-write the weights of a small transformer (ideally with human-understandable
structure) so that its per-frame embeddings of the audio **mel-spectrogram**
predict **ECoG high-gamma cortical activity** (patient patient1) as well as
possible. The reference point is a **linear ridge regression from melspec to
ECoG**.

The model is NEVER trained. The agent must write the weights directly into
`interpretable_transformer.py`.

## The encoding task

Every feature is sampled at **30 Hz** and time-aligned. For each chunk the
embedder maps the mel-spectrogram `(T, 50)` to a per-frame embedding `(T, H)`.
Those become features (FIR-delayed by a few 30 Hz taps, the neural **100 ms lag**
applied, the first **30 frames** trimmed, then z-scored) and a **bootstrapped
ridge** model is fit to predict the **166** high-gamma electrodes.

The metric is the project's **per-sample per-electrode correlation** (`test_corr`):
for each held-out chunk, the Pearson r between predicted and actual high-gamma is
computed per electrode and averaged over electrodes; the chunk values are then
averaged. This is the same metric the main patient1 codebase reports, so results are
comparable to the trained models there (audio-only trained FiLM ≈ 0.068; the full
audio+video model ≈ 0.148 — those use video and training, this is audio-only and
untrained).

The fixed pipeline lives in `src/` (**do not edit**, you don't need to read most
of it).

## Setup

The work happens inside a fresh **run folder**. Each run is an isolated copy
under `<repo_root>/runs-ecog/<tag>/`.

1. **Pick a unique run tag** based on today's date and a counter, e.g. `ecog-jun01-run1`.
2. **Create the run folder**: `python setup_run.py <tag>`. This creates `runs-ecog/<tag>/`:
   - `src/` — **symlink** back to `evolve-ecog/src` (do not modify via the symlink)
   - `interpretable_transformer.py` — **local copy**, this is the file you edit
   - `program.md` — local copy of these instructions
   - `results/` — stores your runs; seeded with the ridge baseline row + plot
   - `interpretable_transformers_lib/` — empty, for snapshots

   `setup_run.py` runs the linear ridge baseline through the identical pipeline
   when it is missing, so `results/overall_results.csv` starts with one
   `Ridge-melspec-baseline` row to beat. (`--skip-baseline` / `--force-baseline`
   override this.)
3. **`cd` into the run folder** and stay there: `cd runs-ecog/<tag>`. Only read your own run folder.
4. **Read the in-scope files**: `interpretable_transformer.py`, `src/eval.py`, `src/features.py`, and `results/overall_results.csv`.

## Experimentation

Each experiment runs best on a single GPU (a CPU run works for small
`--num-train`). Launch it from inside the run folder:
`python interpretable_transformer.py --num-train 60 --num-test 20`.

On this HPC the login node has no GPU — for full-size runs submit the provided
SLURM script (`slurm_run.sh`) or `srun` onto a GPU node. Small CPU runs are fine
for quick iteration.

**What you CAN do:**

- Edit `interpretable_transformer.py` in the run folder (except the evaluation
  harness in the **main** block at the bottom):
  - The `SimpleTransformer`, `Block`, `CausalSelfAttention`, `MLP` architectures.
  - The `input_proj` (continuous melspec → model), positional encoding, and the
    `MelspecEmbedder` blocking / pooling.
  - The `write_weights(model)` function — set any parameter tensors directly:
    hardcoded constants, NumPy arrays, hand-built filter banks / attention
    circuits, lookup tables — anything, as long as you do not train.
  - The architecture hyperparameters (depth, width, heads, ff size, block size).
  - The `model_shorthand_name` and `model_description` strings.
- Save snapshots of `interpretable_transformer.py` under `interpretable_transformers_lib/`.

**What you CANNOT do:**

- Run any training, gradient descent, optimizer step, fitting loop, or backprop.
  All parameters must be set in closed form.
- Load or read weights from any pretrained model.
- Use external tools/libraries to compute the embedding — it must come from the
  transformer forward pass.
- Modify anything reached through a symlink: anything in `src/`.

**The goal is simple: maximize `test_corr`** — the mean held-out per-chunk
per-electrode correlation. The `Ridge-melspec-baseline` row is the target to beat.

**Interpretability criterion**: all else being equal, more interpretable is
better. A small improvement that adds ugly complexity is not worth it. Parameter
count is a rough proxy, but the real criterion is human-understandable structure.

## Output format

When the script finishes it updates `results/overall_results.csv` with columns:

```
subject,test_corr,train_corr,frac_test_elec_above_0.1,encoding_seconds,status,model_shorthand_name,n_params,description
```

1. `subject` — always `patient1`
2. `test_corr` — **primary metric**, mean held-out per-chunk per-electrode correlation
3. `train_corr` — same metric on the training chunks (gauges overfitting)
4. `frac_test_elec_above_0.1` — fraction of electrodes with concatenated-test correlation > 0.1
5. `encoding_seconds` — wall-clock time of the encoding pipeline (features + ridge fit)
6. `status` — `success` or `crash`
7. `model_shorthand_name` — unique name of the model attempt
8. `n_params` — total model parameters (`sum(p.numel() for p in model.parameters())`)
9. `description` — brief description of what this attempt tried

It also refreshes `results/corr_over_iterations.pdf`.

## The experiment loop

You are always inside the run folder (`runs-ecog/<tag>/`).

LOOP FOREVER:

1. Edit `interpretable_transformer.py` with one experimental idea. Update
   `model_shorthand_name` (unique within this run) and `model_description`.
2. Run the experiment: `python interpretable_transformer.py > run.log 2>&1`
3. Read results: `tail -n 15 run.log` and `grep <model_shorthand_name> results/overall_results.csv`
4. If it crashed, check `tail -n 50 run.log` for the stack trace and fix it.
5. Save a snapshot as `interpretable_transformers_lib/<model_shorthand_name>.py`.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human whether to
continue. Run until manually stopped. Even if you beat the baseline, keep going to
find a simpler / more interpretable solution. Do not stop before 30 iterations.

**Example ideas to try** (not exhaustive — be creative):

- Make `input_proj` an interpretable **mel filter bank**: rows that pool adjacent
  mel bins into broadband / formant / spectral-contrast channels that auditory
  cortex is known to track.
- Build attention that **integrates a short window of recent audio** (the relevant
  timescale for high-gamma is ~tens to a few hundred ms), e.g. a causal box / decay
  average, or onset / offset detectors via difference circuits.
- Use the MLP layers as **lookup tables** mapping spectral patterns (e.g. spectral
  flux, energy, voicing) to cortical feature axes.
- Encode **spectro-temporal modulation** (STRF-like) filters directly in the
  weights — sweeps up/down in frequency, amplitude modulation rates.
- Try emphasizing **acoustic edges / onsets** (envelope derivatives), which drive
  auditory high-gamma strongly.
- Vary the number of FIR delays vs. in-model temporal integration to see where the
  temporal context is best captured.

BE CREATIVE.
Try out-of-the-box, diverse ideas.
