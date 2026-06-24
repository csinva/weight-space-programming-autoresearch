# neuro-autoresearch

**Can an AI discover better models of the brain — without ever training one?**

`neuro-autoresearch` is a self-improving research loop for **neural encoding**: the
task of predicting recorded brain activity from the stimulus a person (or animal) saw
or heard. An agent writes a small, **fixed** computational model in code, the harness
scores how well that model's activations predict real neural recordings, and the agent
iterates to push the score up. No gradient descent, no optimizer — the model is
*designed*, not trained.

The unit of progress is a single file (`model.py`) and the unit of truth is a single
number (`test_corr`).

## Why this is interesting

In computational neuroscience the standard way to test a theory of how the brain
represents a stimulus is **encoding**: extract features from the stimulus, fit a linear
readout to neural activity, and measure how well it predicts held-out data. The
features *are* the hypothesis. Here an agent authors those features directly — as plain
Python mapping a stimulus to activations — and the held-out predictivity is an honest,
automatically-computed reward. That turns "propose a better representation of the brain"
into a closed optimization loop the agent can run on its own.

Crucially, **the proposed model is never trained.** Its weights are written straight
into the code (analytic or random-but-fixed). The only thing that gets fit is the
ridge-regression readout, which *is* the evaluation — so the score can't be gamed by
memorizing the data, and the model never sees the neural target `Y`.

## The loop (3 steps)

1. **Propose.** A model in `model.py` — a callable mapping a batch of stimuli
   `X (N, D_in)` to activations `(N, D_out)`. Weights are hard-coded; nothing is
   trained; a sample's activations depend only on that sample.
2. **Activate.** The fixed harness runs the stimulus through the model.
3. **Read out & score.** A himalaya `RidgeCV` readout (fit on GPU) linearly predicts the
   recorded neural activity from those activations. The reward is **`test_corr`** — the
   mean per-target Pearson correlation on a held-out split. Only the readout is fit; the
   model is frozen.

A high `train_corr` with a low `test_corr` means the readout overfit, not that the model
is good.

## The model contract

A self-contained `model.py` defining `class Model` with attributes `name` / `idea` /
`summary` / `n_params` and `__call__(self, X)` mapping `(N, D_in) -> (N, D_out)`, ending
with `MODEL = Model()`. For image datasets `X` is flattened RGB pixels of a square image
(`side = round(sqrt(D_in/3))`). Two hard rules keep the reward real:

1. **Untrained** — every weight is analytic or random-but-fixed (fixed seeds); never fit
   to data, never read the neural target `Y`.
2. **Per-sample** — a sample's activations depend only on that sample (no statistic
   across the batch); keep the output width `D_out < 10000`.

The repo-root `model.py` is the **random-projection baseline** (a fixed random linear
projection of the raw pixels) — the floor every proposed model should beat, and the seed each
search proposer starts from.

## What you edit vs. what's fixed

```
model.py             EVOLVED: the one model (seed: random-projection baseline)
lab.py               the ONE API + CLI over a run's search.db (DB / GPU / dataset hidden)
search/              the open-ended search loop (launcher + proposer pool + evaluator)
  proposer.md          the task description handed to each proposer
harness/             FIXED evaluation harness — must not be edited
  datasets.py          the Dataset registry (pick one via NEURO_DATASET)
  loaders/             one reader per dataset family (the only file/network I/O)
  reward.py            the predictivity reward (himalaya ridge → test_corr)
  run.py               the manual single-eval driver — `python -m harness.run`
scripts/setup_run.py scaffolds an isolated run folder + proposer sandboxes
runs/<dataset>__<date>/  one search: search.db + models/ + proposers/ + eval/ (gitignored)
```

`model.py` is the only artifact that evolves; the fixed harness is what makes the reward
trustworthy.

## Install

```bash
pip install .
```

## Run (offline, no downloads)

### Manual single eval (from the repo root)

```bash
python -m harness.run                    # score the current model.py on the active dataset
NEURO_DATASET=meg python -m harness.run  # pick a preset (text | nsd | meg | ieeg | tvsd | nnn)
```

Loads the data split once, evaluates `model.py` exactly once, prints `train_corr` /
`test_corr` / timing, and appends one row to `results/<study>.csv` (gitignored).

### Open-ended search (unattended)

```bash
search/search.sh nsd --hours 12              # NSD, 12 h, 10 proposers + 1 evaluator
search/search.sh tvsd --hours 6 --fleet 6    # another dataset, 6 proposers
```

`search.sh` scaffolds a dated run and supervises two always-on, **decentralized** loops
against it — there is no conductor and no synchronization barrier:

- **An N-proposer pool** (`propose.sh` → N headless Copilot CLIs via `proposer.sh`). Each
  proposer is confined to its own sandbox, queries shared history, edits its own
  `model.py`, proposes it, blocks on *its own* score, and repeats — self-paced.
- **A single GPU evaluator** (`evaluate.sh` / `evaluator.py`) that claims queued models
  one at a time, scores each with the fixed harness in an isolated subprocess, and writes
  the result back. `repair.sh` auto-fixes small mechanical model faults (≤2 attempts).

Everything shared lives in one SQLite file, `runs/<dataset>__<date>/search.db` — the
queue, the full history, and the scores. Re-launching the same date **resumes** the run
losslessly; a new date starts fresh.

### The `lab` API/CLI

Agents never touch the DB, GPU, or dataset directly — they go through `lab.py`:

```bash
python lab.py propose model.py --idea "..."   # snapshot + queue this model -> prints its id
python lab.py wait <id>                        # block until the evaluator scores it
python lab.py champions -n 5                   # ranked best models, with source
python lab.py sample -n 5                       # random prior models (diversity)
python lab.py get <id> | query "SELECT ..."    # one model / arbitrary read-only SQL
python lab.py summary                           # one-line DB tally + current champion
```

(`next` / `record` / `requeue` are the evaluator-side verbs.)

## Datasets

Presets are registered in `harness/datasets.py` and selected by the `NEURO_DATASET`
environment variable — no source edit, so parallel runs never collide. All are
pre-staged locally and loaded with `download=False`, so they run **fully offline**.

| `NEURO_DATASET` | study | neural | stimulus | reward |
| --- | --- | --- | --- | --- |
| `text` (default) | Le Petit Prince (LPP) | fMRI | audio | whole-study |
| `nsd` | Natural Scenes Dataset | fMRI | image | **ventral** stream |
| `meg` | LPP-Listen | MEG | audio | whole-study |
| `ieeg` | LPP iEEG | iEEG | audio | whole-study |
| `tvsd` / `nnn` | macaque IT | spiking | image | whole-study |

- **`nsd`** uses the prepared single-trial **GLMdenoise betas**
  (`betas_fithrf_GLMdenoise_RR`, Algonauts-2023 style: z-scored per session, 3-rep
  averaged). The reward is the fine-grained **ventral visual stream** (~8.6k voxels). The
  canonical **shared-1000** images are held out as an image-disjoint test set. All 8
  subjects are available via `NEURO_SUBJECT=1..8`.
- The reward region is a fixed **per-dataset** attribute, not an operator knob — there is
  no per-region ROI menu and no way to point the readout at the target.

### How the split avoids leakage

By default the train/test split is a **contiguous temporal hold-out** so time-adjacent
samples that share a neural window can't leak across the split. For NSD (where stimuli
repeat) the split is instead **image-disjoint** (the shared-1000 set). Constant targets
(e.g. non-brain voxels) are dropped. The split is cached on disk, keyed by the full
dataset identity, so reruns load in seconds.

### Adding a dataset

Append a `Dataset(...)` entry to `PRESETS` in `harness/datasets.py` (and a loader under
`harness/loaders/` if it's a new family). Dataset specifics live only there; the harness
and models need no changes. Never point an unattended run at a gated `download=True`
study.

---

*Data loading and alignment are provided by NeuralSet / NeuralFetch.*
