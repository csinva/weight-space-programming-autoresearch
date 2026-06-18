# Copilot instructions — `neuro-autoresearch`

AlphaEvolve-style autoresearch for **neural encoding**, on the **NeuralSet /
NeuralFetch** unified data interface. An agent proposes a fixed, never-trained
model; the harness reads it out with ridge regression against recorded brain
activity; held-out `test_corr` is the reward.

## Rules

- **Edit only `model.py`** (the one evolving model) and **`experiment.py`** (the
  run-config: `LOG_SLICES` is for non-ROI studies — NSD auto-logs its
  own fixed region panel). `harness/` is fixed — the only exceptions are two env
  vars: `NEURO_DATASET` (which preset) and `NEURO_ROI` (which reward region).
- **Models are never trained:** weights written directly in code, no optimizer/
  backprop. Only the ridge readout in `harness/reward.py` is fit — that *is* the eval.
- A model is a plain callable `X (N, D_in) -> activations (N, D_out)`. Prefer
  modality-agnostic models (image-only ones that reshape X to an RGB image need
  `stimulus=image`).
- **All data goes through NeuralSet** — no raw file/network I/O (no nibabel,
  np.load, requests, boto3) in `harness`/`scripts` or `model.py`. Keep this invariant.

## Reward

`test_corr` = mean per-target Pearson r of the ridge readout on the held-out test
split (NSD: image-disjoint **shared-1000**, fit on the other ~9000 unique images;
text/meg: contiguous temporal tail by onset). For NSD the scalar reward narrows to
ONE region chosen by `NEURO_ROI` (default `ventral`; menu: V1/V2/V3/hV4/FFA/PPA/EBA/
VWFA + early/ventral) — the readout still fits ALL nsdgeneral voxels and every region's
predictivity is logged as `corr_<name>`. `train_corr` is reported too (high train +
low test = overfit readout). Raise `test_corr`; small diffs (<~0.005) are noise.

**The goal is to BEAT a trained AlexNet.** A frozen ImageNet AlexNet (fc7) read out
with this same ridge is the per-region bar (logged in `prompts/datasets.md`: e.g.
V1 0.342 · V2 0.299 · V3 0.321 · FFA 0.396 · PPA 0.403 · ventral 0.284). Untrained
hand-designed models already match/beat it in early/mid visual cortex (V1/V2/V3); the
open frontier — and the point of this project — is to beat it in the high-level regions
(ventral/FFA/PPA/EBA/VWFA) too. Treat that as an unsolved, achievable target, not a
ceiling: the right mechanistic, untrained computational model is expected to clear the
AlexNet bar there. Do not assume any region is out of reach.

NSD neural target: the HRF-peak response (FmriExtractor `offset=5.0s`, peak TRs in a
4s window averaged to one value per voxel) — the offset is what makes the signal real
(trained AlexNet ~0.006 at offset 0 → ~0.32 here).

Each `python -m harness.run` evaluates the current model **once** and appends a single
row to the per-study CSV; the append-only `iteration` column counts the models tried
across runs (one row = one model = one iteration). Core columns:
`iteration, timestamp, study, model_name, idea, summary, n_params, n_train, n_test,
n_targets, test_corr, train_corr, gap, best_so_far, seconds, status` + `corr_<region>`.
Set `model.idea` (hypothesis) and `model.summary` (what it does), each ≤2 dense
sentences. A model that raises or returns a bad shape is caught and logged as a
`status="error"` row (NaN reward) instead of crashing the loop.

## Run

```bash
python -m harness.run                          # evaluate the current model
NEURO_DATASET=nsd NEURO_ROI=PPA python -m harness.run   # NSD, reward = PPA region
NEURO_DATASET=meg python -m harness.run
```

Run from repo root (results land in `results/`). Split loads once (cached in seconds;
first NSD build re-reads the timeseries, ~2 min).

**Open-ended search** (the unattended hill-climb) lives in `search/` and runs on Copilot:
a persistent conductor (`search/conductor.md`) drives a per-round Node driver
(`search/round.mjs`) that fans out ~10 parallel proposer agents, evaluates the queue on the
GPU one model at a time, and records every model into the SQLite DB
`search/state/results.db`. Launch it with:

```bash
search/search.sh 8     # 8-hour budget; conductor + per-round driver, all state on disk
```

Full design in `docs/copilot-search-migration.md`.

## Datasets (staged offline, switch = `NEURO_DATASET`; never `download=True` unattended)

`NsdSubj01` fmri/image (`nsd`) · `Li2022PetitSubj` fmri/audio (`text`) ·
`Bel2026PetitListenSample` meg/audio (`meg`) · `Zada2025Podcast` iEEG/audio (`ieeg`) ·
`TvsdMonkeyF`/`TvsdMonkeyN` macaque spiking/image on THINGS (`tvsd`) · `NnnV1` macaque
spiking/image on the NSD-1000 — Triple-N (`nnn`). (`tvsd`/`nnn` ship pre-averaged tables, so
they use `loader="table"` in `harness/data.py`, not a NeuralSet study.) `NsdSubj02`/`NsdSubj03`
subclasses exist for the cross-subject generalization study. Under gitignored
`~/Research/neuralset/data/`. Per-dataset notes in `docs/datasets/`; agent-read context
(incl. baselines) in `prompts/datasets.md`; human field-guides in `docs/learn/`.

## Layout

`model.py` + `experiment.py` (editable) · `harness/{datasets,reward,results,run}.py` +
`harness/loaders/` (the data loaders, one module per family: `core`/`neuralset`/`tvsd`/`nnn`;
`harness/data.py` is a thin back-compat facade) + `harness/studies/` (fixed) · `search/` the
open-ended search loop (`conductor.md`,
`search.sh`, `round.mjs`, `lib/`, `db.py`, `evaluate.sh`, `record.py`; machine-written state
+ the SQLite DB under gitignored `search/state/`) · `prompts/{program,datasets}.md` (agent
reads) · `scripts/{setup_run.py,generalization.py,…}` (standalone tools) · `docs/` (humans:
`codebase-overview.html`, dataset notes, `learn/` field-guides) · `results/<study>.csv` (or
`<study>__<roi>__<run>.csv` per reward region/run; machine writes, gitignored).

## Generalization study (`scripts/generalization.py`)

Takes each region's best subject-1 model and re-evaluates it UNCHANGED on subjects
1/2/3 (ridge re-fit per subject; the shared-1000 test images are common across NSD
subjects), one row per (region, subject) → `results/generalization.csv`. Reuses the
fixed harness wholesale. Needs subj02/03 data staged (`NSD_SUBJ02_PATH`/`NSD_SUBJ03_PATH`).

## Open items (see `docs/overnight-readiness.md`)

Append-only single-row history, the `idea`/`summary` columns, the `status` error net
(`evaluate_on_split_safe`), the trained-AlexNet baseline + beat-AlexNet goal, the structured
Copilot open-ended search (`search/`, SQLite DB), and V2/V3 in the ROI menu have all landed;
the main remaining item is a more stable reward (multi-seed/k-fold or a tie band).
