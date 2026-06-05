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

> **Unified data interface:** every dataset, modality, and stimulus flows
> through **NeuralSet / NeuralFetch** — there is no raw file or network I/O
> anywhere in `harness/`, `evolve/`, or `scripts/`. Adding a dataset is pure
> config, never bespoke parsing. See
> [`.github/copilot-instructions.md`](.github/copilot-instructions.md) for the
> agent operating contract.

## File map

```
evolve/model.py     EDITABLE: the model(s) + main that evaluates and saves
evolve/program.md   the task description / instructions to the agent
harness/config.py   FIXED: the Dataset PRESETS registry (pick one with NEURO_DATASET)
harness/nsd_subj01.py  FIXED: NSD study restricted to subject 1 (NsdSubj01)
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

The stimulus and neural windows can be **decoupled**: a `Dataset` may set
`stim_start` / `stim_duration` (default `None` → reuse `start` / `duration`). When
they differ, the stimulus and neural are segmented separately and re-aligned by
trigger — needed for audio-on-fMRI, where the BOLD target is HRF-delayed
(`start=2.0`) but the audio stimulus is read from a short window **at the word**
(`stim_start=0.0`). When the windows coincide (nsd, meg), a single segmenter
extracts both.

Loading is parallel (16 DataLoader workers, override `NEURO_LOAD_WORKERS`) and the
resulting split is **cached on disk**, keyed by the full dataset identity, so every
proposal rerun loads it in seconds instead of repaying the multi-minute build. The
cache lives outside the repo at `~/.cache/neuro-autoresearch/splits` (override
`NEURO_SPLIT_CACHE`, bypass `NEURO_NO_CACHE`).

## Install

```bash
pip install .   # core deps + copilot-keepgoing (the nonstop agent wrapper)
```

## Run (offline, no downloads)

```bash
/usr/bin/python3.12 evolve/model.py                  # two iterations on the active dataset
/usr/bin/python3.12 evolve/model.py --model identity # a single named model
NEURO_DATASET=meg /usr/bin/python3.12 evolve/model.py  # pick a preset (text|nsd|meg)
```

Run from the repo root so results land in `runs/`. The default run loads the
split once and evaluates two proposals (an initial model + one modification),
printing `train_corr`, `test_corr`, and `frac>0.1` per iteration with timing.
Each evaluated model **upserts** a row into `runs/overall_results.csv` (keyed by
`(dataset, model_name)`), refreshing the `runs/test_corr.png` bar plot.

## Datasets: three real presets (pick one with `NEURO_DATASET`)

`harness/config.py` registers three real datasets in `PRESETS` and selects the
active one from the `NEURO_DATASET` environment variable (default `text`) — no
source edit, so parallel runs never collide:

| `NEURO_DATASET` | Study | neural | stimulus | window (`start`/`duration`) |
| --- | --- | --- | --- | --- |
| `text` (default) | `Li2022Petit` (Le Petit Prince, full) | fMRI | audio | neural 2.0 / 4.0 s · stim 0.0 / 0.5 s |
| `nsd` | `NsdSubj01` (NSD, subject 1) | fMRI | image | 0.0 / 3.0 s |
| `meg` | `Bel2026PetitListenSample` (LPP-Listen) | MEG | audio | 0.0 / 0.5 s |

- **`text`** — the **Le Petit Prince** narrative-listening study. Each spoken
  **Word** triggers a segment; the stimulus is the **Mel spectrum of the audio at
  the word** and the neural target is the fMRI volume. The neural window is offset
  to capture the delayed (HRF) BOLD response, but the audio is near-instant, so the
  stimulus is read from a short window **at the word** (`stim_start`/`stim_duration`
  decoupled from the HRF-delayed neural window).
- **`nsd`** — the **Natural Scenes Dataset**, restricted to subject 1 via the
  `NsdSubj01` subclass in `harness/nsd_subj01.py`. Each shown **Image** triggers a
  segment; the stimulus is the resized image and the target is the fMRI volume.
- **`meg`** — **Le Petit Prince MEG** (LPP-Listen, OpenNeuro `ds007523`), the
  loop's first electrophysiological modality. Each spoken **Word** triggers a 0.5 s
  MEG epoch; the stimulus is the Mel spectrum of the audio in that same window (MEG
  is near-instant, so stimulus and neural windows coincide). The raw `.fif` is
  Elekta MaxShield, handled in `data.py` via `allow_maxshield=True`.

All three are pre-downloaded locally and pointed to via `path=` with
`download=False`, so they **run offline**.

## Switch datasets

Set `NEURO_DATASET` to one of `text` / `nsd` / `meg` (an unknown value raises a
clear error listing the valid keys). To add another, name a study from
`neuralset.Study.catalog()` and append a `Dataset(...)` entry to `PRESETS` in
`harness/config.py` — dataset specifics live only in `config.py`.

## Datasets staged for offline use

Four studies are pre-downloaded locally (under the gitignored
`~/Research/neuralset/data/`, nested as `<path>/<StudyName>/`) so a switch is a
one-line, **offline** `download=False` edit. Together they span two neural
modalities (fMRI + MEG) and three stimulus types:

| Study | neural | stimulus | local path | status |
| --- | --- | --- | --- | --- |
| `Li2022PetitSample` (LPP story) | fmri | text | `…/li2022petit_sample` | **active** |
| `Allen2022MassiveSample` (NSD) | fmri | image | `…/nsd_subj01`, `…/nsd_sample` | staged |
| `Bel2026PetitListenSample` (LPP MEG, listen) | meg | text/audio | `…/bel2026petitlisten_sample` | staged |
| `Bel2026PetitReadSample` (LPP MEG, read) | meg | text | `…/bel2026petitread_sample` | staged |

Per-dataset download notes and ready-to-paste `Dataset(...)` config blocks live
in `runs/` (`NSD_DOWNLOAD.md`, `THIRD_DATASET.md`). MEG `.fif` raws are MaxShield
(need `allow_maxshield=True`); the NSD full pull (`nsd_subj01`) must be extracted
before its `path` is used. **Never point an unattended run at a gated
`download=True` study** — keep everything offline against a staged path.

## Status & open ideas

The loop runs end-to-end offline and the reward is genuinely held-out. Before a
hands-off, multi-iteration overnight run, `runs/OVERNIGHT_READINESS.md` documents
the remaining improvements: append-only result history (today's `save_results`
upserts by name and loses the trajectory), an error safety net
(`status="error"` instead of a bare crash), fail-fast on a missing data path
(instead of falling through to a gated download), and a more stable reward
(multi-seed / k-fold, a tie band, or signal-bearing target selection). See
[`.github/copilot-instructions.md`](.github/copilot-instructions.md) for the full
agent contract and these open items.

## Drive the autoresearch loop nonstop

The Copilot CLI sometimes stops mid-run. Wrap it with `copilot-keepgoing`
(installed via `pip install .`) to auto-nudge it whenever it goes idle:

```bash
copilot-keepgoing -- copilot --autopilot --yolo   # then: "Follow evolve/program.md"
```

Credits: data + alignment via NeuralSet/NeuralFetch; loop structure after csinva's
circuit-building-autoresearch.
