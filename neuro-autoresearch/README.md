# neuro-autoresearch

**Can an AI discover better models of the brain — without ever training one?**

`neuro-autoresearch` is a self-improving research loop for **neural encoding**:
the task of predicting recorded brain activity from the stimulus a person saw or
heard. An agent writes a small, **fixed** computational model in code, the harness
scores how well that model's activations predict real neural recordings, and the
agent iterates to push the score up. No gradient descent, no optimizer — the model
is *designed*, not trained.

The result is a tight, verifiable loop where the unit of progress is a single file
(`model.py`) and the unit of truth is a single number (`test_corr`).

## Why this is interesting

In computational neuroscience, the standard way to test a theory of how the brain
represents a stimulus is **encoding**: extract features from the stimulus, fit a
linear readout to neural activity, and measure how well it predicts held-out data.
The features *are* the hypothesis. Here we let an agent author those features
directly — as plain Python that maps a stimulus to activations — and use the
held-out predictivity as an honest, automatically-computed reward. That turns
"propose a better representation of the brain" into a closed optimization loop the
agent can run on its own.

Crucially, **the proposed model is never trained.** Its weights are written
straight into the code. The only thing that gets fit is a ridge-regression readout,
which *is* the evaluation — so the score can't be gamed by memorizing the data.

## The loop (3 steps)

1. **Propose.** The agent writes a model in `model.py` — a callable mapping
   a batch of stimuli `X (N, D_in)` to activations `(N, D_out)`. Weights are
   hard-coded; nothing is trained.
2. **Activate.** The fixed harness runs the stimulus through the model to get
   activations.
3. **Read out & score.** A ridge regression linearly predicts the recorded neural
   activity from those activations. The reward is **`test_corr`** — the mean
   per-target Pearson correlation on a held-out split. Only the ridge readout is
   fit; the model is frozen.

`model.py` is the evolved artifact; `test_corr` is the verifiable reward. A high
`train_corr` with a low `test_corr` means the readout overfit, not that the model
is good.

## One data interface for every dataset

Every dataset, neural modality, and stimulus type flows through a single unified
interface (**NeuralSet / NeuralFetch**). There is **no raw file or network I/O**
anywhere in `harness/`, `evolve/`, or `scripts/` — adding a new dataset is pure
configuration, never bespoke parsing. This is what lets the same loop run, untouched,
across fMRI and MEG, and across image, text, and audio stimuli.

The loop is **modality-agnostic**: neither the model (`stimulus X -> activations`)
nor the reward ever assumes a modality. Each dataset declares a `stimulus` type and
the harness dispatches the matching offline feature extractor:

| `stimulus` | Stimulus → X | extractor |
| --- | --- | --- |
| `image` | flattened `imsize×imsize×3` RGB pixels | `RawPixels` |
| `text` | the trigger word's `[length, zipf-frequency]` | `WordLength` + `WordFrequency` |
| `audio` | time-averaged Mel bands | `MelSpectrum` |

The neural side (`fmri` / `meg` / `eeg`) is dispatched the same way.

## What you edit vs. what's fixed

```
model.py               EDITABLE: the one evolving model (seed: identity)
experiment.py          EDITABLE: run-config — ITERATIONS + LOG_SLICES
prompts/program.md     the task description handed to the agent
prompts/datasets.md    the active-dataset context the agent reads each run
harness/datasets.py    FIXED: the Dataset registry (pick one via NEURO_DATASET)
harness/studies/       FIXED: per-study NeuralSet subclasses (NSD subj1, LPP subj)
harness/data.py        FIXED: stimulus extractors + aligned, split (X, Y) tensors
harness/reward.py      FIXED: the predictivity reward, evaluate()
harness/results.py     FIXED: append-only result ledger, save_results()
harness/run.py         FIXED: the driver — `python -m harness.run`
scripts/setup_run.py   FIXED: `python scripts/setup_run.py <tag>` — make an isolated run folder
results/               <study>.csv, one appended row per evaluation (gitignored)
runs/<tag>/            isolated per-run folder (symlinked harness + local model.py; gitignored)
```

`model.py` (and `experiment.py`) are the **only** files the agent edits. `harness/`
is the fixed evaluation harness and must not be touched — that fixedness is what
makes the reward trustworthy.

## Install

```bash
pip install .   # core deps + copilot-keepgoing (the nonstop agent wrapper)
```

## Run (offline, no downloads)

There are two ways to run, and both work.

### Single tree (quick, from the repo root)

```bash
python -m harness.run                    # evaluate the current model on the active dataset
NEURO_DATASET=meg python -m harness.run  # pick a preset (text | nsd | meg)
NEURO_DATASET=nsd NEURO_ROI=PPA python -m harness.run  # NSD: reward = the PPA region
NEURO_DATASET=nsd NEURO_ROI=FFA NEURO_RUN=run1 python -m harness.run  # unique CSV per run
```

Run from the repo root so results land in `results/`. A run loads the data split
once and evaluates the model in `model.py`, printing `train_corr`, `test_corr`, and
`frac>0.1` with timing. Each evaluation **appends** one row to
`results/<study>.csv` — or `results/<study>__<roi>.csv` for a per-region NSD reward,
or `results/<study>__<roi>__<run>.csv` when `NEURO_RUN` is set so **each run gets its
own unique CSV**. The history is append-only (keyed by a monotonic `iteration`) and
`results/` is **gitignored** — runs never write to git, only to their CSV.

### Isolated run folder (recommended for autoresearch)

For an unattended hill-climb, give each run its own **isolated folder** so its
`model.py`, attempt snapshots, and results never collide with another run — no git
branch, no commits, everything local to the folder:

```bash
python scripts/setup_run.py demo-run1    # create runs/demo-run1/
cd runs/demo-run1
NEURO_DATASET=nsd python -m harness.run  # one evaluation pass, writes results/<study>.csv here
```

`scripts/setup_run.py <tag>` creates `runs/<tag>/` containing:

```
harness/      -> relative symlink to the repo's harness/ (FIXED; do not edit through it)
prompts/      -> relative symlink to the repo's prompts/ (FIXED)
model.py        local copy — EDIT this one (the evolving model)
experiment.py   local copy — EDIT this one (ITERATIONS + LOG_SLICES)
results/        empty — the run-local <study>.csv lands here
models_lib/     empty — drop a snapshot of model.py per attempt here
```

The loop inside a run folder: edit the local `model.py`, snapshot each attempt
into `models_lib/` (e.g. `cp model.py models_lib/attempt-NN.py`), run
`python -m harness.run`, and read the appended row from the run-local
`results/<study>.csv`. Because `harness` is symlinked, `import harness.run` and the
local `model.py`/`experiment.py` all resolve, and `harness.results.save_results`
writes to the cwd-relative `results/` — i.e. into `runs/<tag>/results/`. The whole
`runs/` tree is gitignored.


## Datasets

Three real studies are registered in `harness/datasets.py` and selected by the
`NEURO_DATASET` environment variable (default `text`) — no source edit, so parallel
runs never collide:

| `NEURO_DATASET` | Study | neural | stimulus | window (`start` / `duration`) |
| --- | --- | --- | --- | --- |
| `text` (default) | `Li2022Petit` (Le Petit Prince) | fMRI | audio | neural 2.0 / 4.0 s · stim 0.0 / 0.5 s |
| `nsd` | `NsdSubj01` (Natural Scenes Dataset, subj 1) | fMRI | image | neural HRF offset 5.0 / 4.0 s, TR-mean |
| `meg` | `Bel2026PetitListen` (LPP-Listen) | MEG | audio | 0.0 / 0.5 s |

- **`text`** — the **Le Petit Prince** narrative-listening study. Each spoken word
  triggers a segment; X is the Mel spectrum of the audio at the word and the target
  is the fMRI volume. The neural window is offset to capture the delayed (HRF) BOLD
  response while the audio is read from a short window at the word, so the stimulus
  and neural windows are decoupled (`stim_start` / `stim_duration`).
- **`nsd`** — the **Natural Scenes Dataset**, subject 1. Each shown image triggers a
  segment; X is the resized image. The target is the **HRF-peak** fMRI response: the
  window is shifted forward (`neural_offset=5.0 s`, since BOLD peaks ~5 s after onset)
  and the peak TRs in a 4 s window are averaged into one value per voxel — without
  this offset even a trained AlexNet scores ~0.006; with it, ~0.32. Subject 1's 10,000
  images are each shown ~3×, so the loader collapses repeats to one sample per unique
  image (averaging the neural target) and holds out the canonical **shared-1000**
  images as an image-disjoint test set (fit on the other ~9000).
  - **Per-region reward (`NEURO_ROI`).** The scalar `test_corr` is computed over ONE
    brain region, chosen per run (default `ventral`); menu: `V1`, `hV4`, `FFA`, `PPA`,
    `EBA`, `VWFA` + visual streams `early`, `ventral`. The ridge readout still fits
    **all** nsdgeneral voxels and every region's predictivity is logged as a
    `corr_<name>` column — only the headline reward narrows. Each region writes its own
    `results/<study>__<roi>.csv`, so parallel per-region hill-climbs never collide.
- **`meg`** — **Le Petit Prince MEG** (LPP-Listen), the loop's electrophysiological
  modality. Each spoken word triggers a 0.5 s MEG epoch; X is the Mel spectrum of the
  audio in that same window (MEG is near-instant, so stimulus and neural windows
  coincide).

All presets are pre-staged locally and loaded with `download=False`, so they run
**fully offline**.

### How the split avoids leakage

By default the train/test split is a **contiguous temporal hold-out** (the latest
`test_frac` of triggers by onset), so time-adjacent samples that share a neural
window can't leak across the split. For NSD, where stimuli repeat, the split is
instead **image-disjoint** (the shared-1000 set), asserting zero train/test image
overlap. Targets that are constant across the training set (e.g. non-brain voxels)
are dropped, since they carry no signal and only dilute the mean correlation.

The resulting split is **cached on disk**, keyed by the full dataset identity, so
every rerun loads in seconds instead of repaying the multi-minute build. The cache
lives outside the repo at `~/.cache/neuro-autoresearch/splits` (override with
`NEURO_SPLIT_CACHE`, bypass with `NEURO_NO_CACHE`).

### Adding a dataset

Name a study from `neuralset.Study.catalog()` and append a `Dataset(...)` entry to
`PRESETS` in `harness/datasets.py`. Dataset specifics live only in `datasets.py`; the
harness and models need no changes. Keep unattended runs pointed at a staged local
path — **never** point an unattended run at a gated `download=True` study.

## Drive the loop

Launch an autoresearch loop with `scripts/autoresearch.sh`. It **always** runs the
Copilot CLI with the **Opus** model; the **only configurable knob is the reasoning
effort** (default `medium`). The experiment is chosen by the `NEURO_*` env vars:

```bash
# NSD, reward = FFA region, its own results CSV, default (medium) effort:
NEURO_DATASET=nsd NEURO_ROI=FFA NEURO_RUN=$(date +%s) scripts/autoresearch.sh

# higher reasoning effort (none|low|medium|high|xhigh|max):
NEURO_DATASET=nsd NEURO_ROI=PPA scripts/autoresearch.sh high
```

The launcher runs `copilot -p "Follow prompts/program.md" --model claude-opus-4.8
--effort <effort> --autopilot --allow-all`, which keeps proposing and evaluating
models in one turn until it stops or is killed. Bound a run with `timeout`:

```bash
NEURO_DATASET=nsd NEURO_ROI=FFA NEURO_RUN=$(date +%s) timeout 15m scripts/autoresearch.sh
```

---

*Data loading and alignment are provided by NeuralSet / NeuralFetch.*
