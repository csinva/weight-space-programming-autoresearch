# neuro-autoresearch

A bare-minimum, AlphaEvolve-style **autoresearch** proof-of-concept for
neural encoding, built on **NeuralSet + NeuralFetch**. Structure mirrors
[`csinva/circuit-building-autoresearch`](https://github.com/csinva/circuit-building-autoresearch)
(`evolve-neuro/`), generalised to any NeuralSet dataset.

## The loop (3 steps)

1. An agent **proposes a computational model** (`model.py`) — weights written
   directly in code, **never trained** (no gradient descent, no optimizer).
2. The fixed harness passes the stimulus through the model → **activations**.
3. Activations are **linearly read out with ridge regression** to predict the
   recorded neural activity. The held-out **`test_corr`** (mean per-target
   Pearson r) is the **reward**. Only the ridge readout is fit.

So `model.py` is the evolved artifact; `test_corr` is the verifiable reward.

## File map

```
evolve/model.py     EDITABLE: the model(s) + main that evaluates and saves
evolve/program.md   the task description / instructions to the agent
harness/config.py   FIXED: the one active Dataset (edit one line to switch datasets)
harness/data.py     FIXED: per-modality stimulus extractors + aligned, split (X, Y) tensors
harness/eval.py     FIXED: predictivity reward + evaluate() + save_results()
runs/               overall_results.csv + test_corr.png (written by save_results)
```

`evolve/` is the only zone the agent edits; `harness/` is the fixed evaluation
harness the agent must not touch.

## Stimulus modalities

The loop is **modality-agnostic**: the model (`stimulus batch X -> activations`)
and the reward never assume a modality. Each `Dataset` declares a `stimulus`
field and `harness/data.py` dispatches the matching offline NeuralSet extractor
(no model-weight downloads):

| `stimulus` | Stimulus → X | extractor |
| --- | --- | --- |
| `image` | flattened `imsize×imsize×3` RGB pixels | `RawPixels` |
| `text` | the trigger word's `[length, zipf-frequency]` | `WordLength` + `WordFrequency` |
| `audio` | time-averaged Mel bands | `MelSpectrum` |

The neural side (`neural`: `meg` / `eeg` / `fmri`) is dispatched the same way.
Targets that are constant across the training set (e.g. non-brain fMRI voxels)
are dropped — they carry no signal and only dilute the mean-r. The train/test
split is a **contiguous temporal hold-out** (the latest `test_frac` of triggers
by onset), so time-adjacent samples that share a neural window can't leak across
the split.

## Install

```bash
pip install .   # core deps + copilot-keepgoing (the nonstop agent wrapper)
```

## Run (offline, no downloads)

```bash
/usr/bin/python3.12 evolve/model.py                  # two iterations on the active dataset
/usr/bin/python3.12 evolve/model.py --model identity # a single named model
```

Run from the repo root so results land in `runs/`. The default run loads the
split once and evaluates two proposals (an initial model + one modification),
printing `train_corr`, `test_corr`, and `frac>0.1` per iteration with timing.
Each evaluated model **upserts** a row into `runs/overall_results.csv` (keyed by
`(dataset, model_name)`), refreshing the `runs/test_corr.png` bar plot.

## Active dataset: Le Petit Prince story (fMRI + word stimulus)

The active dataset is the **Le Petit Prince** narrative-listening study —
NeuralSet's `Li2022PetitSample` (fMRI), `stimulus=text`. Each spoken **Word**
(~1517 of them, one subject) triggers a segment; the stimulus is that word's
`[length, zipf-frequency]` and the neural target is the fMRI volume. The segment
window is offset (`start=2.0`, `duration=4.0s`) to capture the delayed (HRF)
BOLD response; because the stimulus uses the *trigger* word, that offset only
shifts the neural window, not X. With ~1208 train words and a 2-D stimulus,
`n_train ≫ feature_dim`, so the ridge readout is not overfit (`train_corr` and
`test_corr` are both reported to confirm).

The **sample subset is pre-downloaded locally** and pointed to via the `path=`
field with `download=False` in `harness/config.py`, so it **runs offline**.

> The previous active dataset, the **Natural Scenes Dataset (NSD)** image study
> (`Allen2022MassiveSample`, fMRI), is kept one line away — see the commented
> `Dataset(...)` in `harness/config.py`. A full download (`download=True`) would
> need the `aws` CLI and NSD Terms & Conditions acceptance; the local sample does
> not.

## Switch datasets

Edit the single `DATASET = Dataset(...)` line in `harness/config.py` (set
`study=`, `neural=`, `stimulus=`, `trigger=`, and either `download=True` or a
local `path=`). Nothing else changes — dataset specifics live only in
`config.py`. The image/fMRI NSD sample is included as a ready-to-paste
alternative.

## Drive the autoresearch loop nonstop

The Copilot CLI sometimes stops mid-run. Wrap it with `copilot-keepgoing`
(installed via `pip install .`) to auto-nudge it whenever it goes idle:

```bash
copilot-keepgoing -- copilot --autopilot --yolo   # then: "Follow evolve/program.md"
```

Credits: data + alignment via NeuralSet/NeuralFetch; loop structure after csinva's
circuit-building-autoresearch.
