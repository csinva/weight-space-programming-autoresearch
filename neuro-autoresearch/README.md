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
3. **Read out & score.** A ridge readout (fit on GPU) linearly predicts the recorded
   neural activity from those activations. The reward is **`test_corr`** — the mean
   per-target Pearson correlation on a held-out split. Only the readout is fit; the model
   is frozen. The readout defaults to a fast eigendecomposition + leave-one-out ridge
   (`NEURO_READOUT=fast`, ~20× faster on GPU); set `NEURO_READOUT=himalaya` for the
   reference himalaya `RidgeCV` (same per-target alpha selection, agrees to ~1e-4).

A high `train_corr` with a low `test_corr` means the readout overfit, not that the model
is good.

## The model contract

A self-contained `model.py` defining `class Model` with attributes `name` / `idea` /
`description` / `insight` / `n_params` and `__call__(self, X)` mapping `(N, D_in) -> (N, D_out)`,
ending with `MODEL = Model()`. For image datasets `X` is flattened RGB pixels of a square image
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
hub.py               the ONE API + CLI over a run's search.db (DB / GPU / dataset hidden)
search/              the open-ended search loop (launcher + proposer pool + evaluator)
  proposer.md          the task description handed to each proposer
harness/             FIXED evaluation harness — must not be edited
  datasets.py          the Dataset registry (pick one via NEURO_DATASET)
  loaders/             the only file/network I/O — timeline.py + studies.py (Segmenter timeline
                       studies) vs vision/ (pre-aligned response tables); load_split dispatch
  reward.py            the predictivity reward: forward → readout → test_corr (orchestration)
  ridge.py             the ridge readout — fast eigh+LOO (default) | himalaya, via NEURO_READOUT
  roi.py               reward-region target selection + advisory per-ROI breakdown
  run.py               the manual single-eval driver — `python -m harness.run`
scripts/setup_run.py scaffolds an isolated run folder + proposer sandboxes
scripts/benchmark/   profile the readout pipeline phase-by-phase (3 concurrent) + HTML dashboard
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
NEURO_DATASET=ieeg python -m harness.run # pick a preset (nsd | tvsd | nnn | ieeg)
NEURO_READOUT=himalaya python -m harness.run  # reference readout (default: the ~20× faster fast ridge)
```

Loads the data split once, evaluates `model.py` exactly once, and prints the scored row
(`train_corr` / `test_corr` / timing). It does not persist anything — result recording lives
only in the unattended search loop (see `hub.py`).

### Open-ended search (unattended)

```bash
search/search.sh nsd --hours 12              # NSD, 12 h, 10 proposers + 1 evaluator + conductor (steers every 60m)
search/search.sh nsd --hours 12 --effort high   # ONE knob: thinking effort for ALL proposers + conductor
search/search.sh nsd --hours 12 --epoch-mins 0  # conductor OFF — pure decentralized pool
search/search.sh tvsd --hours 6 --fleet 6    # another dataset, 6 proposers
search/search.sh tvsd --hours 6 --gpu 1      # pin to GPU 1 — run a dataset per card in parallel on a multi-GPU box
```

On a **multi-GPU** box, `--gpu N` (or just `CUDA_VISIBLE_DEVICES=N`) pins the whole run to one card,
so you can launch several datasets side-by-side — `nsd` on GPU 0, `tvsd` on GPU 1, … each fully
isolated. Default (no flag) uses the current default GPU, unchanged.

Thinking **effort** is a single knob — `--effort` (or the `NEURO_EFFORT` env var; one of
`low|medium|high|xhigh|max`, default `xhigh`) — applied uniformly to **every proposer and
the conductor**. The model *mix* is separate: by default the fleet is spread round-robin across a
roster of five distinct models — Opus 4.8 / 4.7 / 4.6, GPT‑5.5, and Gemini 3.1 Pro (pinned to
`high` effort) — so a pool of 10 runs 2 proposers per model. Override the roster via
`NEURO_MODEL_ROSTER` (space-separated `model` or `model:effort` entries).

`search.sh` scaffolds a dated run and supervises up to three always-on, **decentralized**
loops against it — the search stays self-paced with no synchronization barrier:

- **An N-proposer pool** (`propose.sh` → N headless Copilot CLIs via `proposer.sh`). Each
  proposer is confined to its own sandbox, queries shared history, edits its own
  `model.py`, proposes it, blocks on *its own* score, and repeats — self-paced.
- **A single GPU evaluator** (`evaluate.sh` / `evaluator.py`) that claims queued models
  one at a time, scores each with the fixed harness in an isolated subprocess, and writes
  the result back. `repair.sh` auto-fixes small mechanical model faults (≤2 attempts).
- **An optional conductor** (`conduct.sh` / `conductor.md`) — a steering overlay. Every
  `--epoch-mins` (default 60; `0` disables it) it runs one headless pass that reads the
  board and writes ONE broad, **advisory** `direction.txt` per proposer to keep the pool
  diverse and unstuck. It never gates the queue or scoring — proposers just read their
  `direction.txt` to bias what they explore.

Everything shared lives in one SQLite file, `runs/<dataset>__<date>/search.db` — the
queue, the full history, and the scores. Re-launching the same date **resumes** the run
losslessly; a new date starts fresh.

### The `hub` API/CLI

Agents never touch the DB, GPU, or dataset directly — they go through `hub.py`:

```bash
python hub.py propose-model model.py --idea "..."  # snapshot + queue this model -> prints its id
python hub.py wait-for-score <id>              # block until the evaluator scores it
python hub.py random-models -n 5               # random prior models: scores + ideas (no source)
python hub.py my-best [--no-code]              # YOUR island's best model (+source) to build on
python hub.py peek-island                      # best of one RANDOM other island: idea + notes only (no source, no score)
python hub.py query-db "SELECT ..."            # arbitrary read-only SQL
python hub.py summary                          # one-line DB tally + current champion
```

Each proposer is its own **island**: it can read source only from `my-best` (its own lineage);
`peek-island` shares one random other island's idea and notes (never source, never score), so
backbones diverge instead of all grafting onto one recipe. (`top-models` is a conductor-only
scoreboard; `requeue` / `migrate` / `reset-island` are operator/evaluator-side verbs.)

## Datasets

Presets are registered in `harness/datasets.py` and selected by the `NEURO_DATASET`
environment variable — no source edit, so parallel runs never collide. All are
pre-staged locally and loaded with `download=False`, so they run **fully offline**.

Two families — **vision** (image → neural response tables) and **language** (spoken words → intracranial recordings):

| `NEURO_DATASET` | study | neural | stimulus | reward |
| --- | --- | --- | --- | --- |
| `nsd` (default) | Natural Scenes Dataset | fMRI | image | **early+ventral** stream, noise-normalized |
| `tvsd` | THINGS ventral-stream (macaque) | spiking | image | all electrodes (V1→IT) |
| `nnn` | Triple-N (macaque) | spiking | image | all reliable units (EVC+IT), noise-normalized |
| `ieeg` | 'Podcast' ECoG (human) | iEEG | text (chars) | all electrodes, 5-fold temporal CV |

- **`nsd`** uses the prepared single-trial **GLMdenoise betas**
  (`betas_fithrf_GLMdenoise_RR`, Algonauts-2023 style: z-scored per session, 3-rep
  averaged). The reward is the joint **early+ventral visual stream** (~14.5k voxels),
  noise-ceiling-normalized (mean sign-preserving r²/NC_r2 over the reliable voxels). The
  canonical **shared-1000** images are held out as an image-disjoint test set. Subjects
  1, 2, 5, 7 are staged in-repo via `NEURO_SUBJECT` (1+2 the search reward, 5+7 held out).
- **`ieeg`** is the language dataset: human intracranial **high-gamma** (70–200 Hz) power to a
  ~30-min spoken podcast. The stimulus is **`chars`** — each word encoded as the char-id grid of
  its N-gram, so `model.py` must build linguistic structure from characters up (no pretrained
  embeddings). The reward is a rotating **5-fold temporal CV** (a single fixed hold-out is trivially
  overfit), meaned across patients via `NEURO_SUBJECT`.
- The reward region is a fixed **per-dataset** attribute, not an operator knob — there is
  no per-region ROI menu and no way to point the readout at the target.

### How the split avoids leakage

By default the train/test split is a **contiguous temporal hold-out** so time-adjacent
samples that share a neural window can't leak across the split. For NSD (where stimuli
repeat) the split is instead **image-disjoint** (the shared-1000 set). Constant targets
(e.g. non-brain voxels) are dropped. The split is cached on disk, keyed by the full
dataset identity, so reruns load in seconds.

### Adding a dataset

Append a `Dataset(...)` entry to `PRESETS` in `harness/datasets.py` (and a loader: extend
`harness/loaders/timeline.py` if it's a NeuralSet timeline Study, or add a reader under
`harness/loaders/vision/` if it's a pre-aligned table read directly). Dataset specifics live only there; the harness and models need no changes. Never point an unattended run at a gated `download=True`
study.

---

*Timeline segmentation and alignment are provided by NeuralSet.*
