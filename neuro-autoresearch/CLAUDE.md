# Copilot instructions — `neuro-autoresearch`

AlphaEvolve-style autoresearch for **neural encoding**, on the **NeuralSet /
NeuralFetch** unified data interface. An agent proposes a fixed, never-trained
model; the harness reads it out with ridge regression against recorded brain
activity; held-out `test_corr` is the reward.

## Rules

- **Edit only `model.py`** (the one evolving model — a callable input→representation map).
  `harness/` is fixed; the only knob is `NEURO_DATASET` (which preset). The reward is the
  dataset's **fixed region** (NSD: the ventral stream) — there is no operator ROI knob.
  (`experiment.py` is a vestigial stub.)
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
text/meg: contiguous temporal tail by onset). For NSD the reward covers the **ventral
visual stream** — a fixed per-dataset region, NOT an operator knob. `train_corr` is
reported too (high train + low test = overfit readout). Raise `test_corr`; small diffs
(<~0.005) are noise.

**The goal is to BEAT a trained AlexNet** read out with this same ridge. A frozen ImageNet
AlexNet (fc7) scores ~0.284 on ventral; the noise ceiling is ~0.50. **The search target is
`test_corr` ≥ 0.40 on the ventral stream.** Untrained hand-designed
models already match/beat AlexNet in early/mid visual cortex; the open frontier — and the point
of this project — is the high-level **ventral stream**. Treat that as an unsolved, achievable
target: the right mechanistic, untrained computational model is expected to clear the bar.

NSD neural target: the HRF-peak response (FmriExtractor `offset=5.0s`, peak TRs in a
4s window averaged to one value per voxel) — the offset is what makes the signal real
(trained AlexNet ~0.006 at offset 0 → ~0.32 here).

A manual `python -m harness.run` evaluates the current `model.py` once and appends a row to
`results/<study>.csv`. The **open-ended search** instead records each model as one row in the
run's `search.db` (one row per model, doubling as the queue). Set `model.idea` (hypothesis) and
`model.summary` (what it does), each ≤2 dense sentences. A model that raises or returns a bad
shape is caught and logged as a `status="error"` row instead of crashing the loop.

## Run

```bash
python -m harness.run                          # evaluate the current model.py once (manual)
NEURO_DATASET=meg python -m harness.run        # switch dataset
```

Run from repo root (results land in `results/`). Split loads once (cached in seconds;
first NSD build re-reads the timeseries, ~2 min).

**Open-ended search** (the unattended hill-climb) lives in `search/` + the importable `lab` API,
and runs on Copilot as two always-on, decentralized loops against one **run**: an N-proposer pool
(`search/propose.sh`, 10 headless Copilot CLIs — one per sandbox) invents untrained models and
queues them; the single evaluator (`search/evaluate.sh`) scores them one at a time on the GPU.
There is no conductor and **no wave barrier** — each proposer is **self-paced** (it waits only for
its OWN model's score via `lab.wait_for_result`, then proposes again). The run's SQLite DB
`runs/<dataset>__<date>/search.db` (one `models` row per model, doubling as the queue via a
`queued → running → ok|error` status) is the only shared memory, reached only through `lab`
(`lab.propose/wait_for_result/champions/sample/get_model/query` + `get_next_model/record`).
`search/search.sh` scaffolds the dated run and supervises both loops until a wall-clock budget:

```bash
search/search.sh <dataset> [imsize] [--hours H] [--fleet N]   # e.g. search/search.sh nsd 128 --hours 12
```

Each run is isolated in `runs/<dataset>__<date>/` (its own `search.db` + shared `models/` archive
+ per-proposer `proposers/pNN/` sandboxes + one `eval/`); same dataset re-launched the same day
resumes losslessly, a new date starts fresh. Killing the launcher leaves the DB consistent
(orphaned `running` rows reset to `queued`).

## Datasets (staged offline, switch = `NEURO_DATASET`; never `download=True` unattended)

`NsdSubj01` fmri/image (`nsd`) · `Li2022PetitSubj` fmri/audio (`text`) ·
`Bel2026PetitListenSample` meg/audio (`meg`) · `Zada2025Podcast` iEEG/audio (`ieeg`) ·
`TvsdMonkeyF`/`TvsdMonkeyN` macaque spiking/image on THINGS (`tvsd`) · `NnnV1` macaque
spiking/image on the NSD-1000 — Triple-N (`nnn`). (`tvsd`/`nnn` ship pre-averaged tables, so
they use `loader="table"` in `harness/data.py`, not a NeuralSet study.) `NsdSubj02`/`NsdSubj03`
subclasses exist for the cross-subject generalization study. Under gitignored
`~/Research/neuralset/data/`. Per-dataset notes in `docs/datasets/`; the AlexNet bar + per-dataset
framing live in the proposer prompt (`search/proposer.sh`); human field-guides in `docs/learn/`.

## Layout

`model.py` (editable) · `harness/{datasets,reward,results,run}.py` +
`harness/loaders/` (the data loaders, one module per family: `core`/`neuralset`/`tvsd`/`nnn`;
`harness/data.py` is a thin back-compat facade) + `harness/studies/` (fixed) · `lab.py` (the
ONE importable run API over `search.db` — propose/wait/champions/sample/get/query +
get_next_model/record/score) · `search/` the open-ended search (launcher `search.sh` scaffolding
a dated run and supervising two always-on loops — the proposer pool `propose.sh` running N
`proposer.sh` Copilot CLIs + the evaluator `evaluate.sh` wrapping `evaluator.py`; `proposer.md`
the prompt; `repair.sh`) · `scripts/{setup_run.py,generalization.py,…}` (`setup_run.py` scaffolds
a run + its proposer sandboxes, each seeded from the IDENTITY model) · machine-written run state
under gitignored `runs/<dataset>__<date>/` (`search.db` + `models/` + `proposers/pNN/` + `eval/`) ·
`docs/` (humans: `codebase-overview.html`, dataset notes, `learn/` field-guides) ·
`results/<study>.csv` (manual `harness.run` output; machine writes, gitignored).

## Generalization study (`scripts/generalization.py`)

Takes each region's best subject-1 model and re-evaluates it UNCHANGED on subjects
1/2/3 (ridge re-fit per subject; the shared-1000 test images are common across NSD
subjects), one row per (region, subject) → `results/generalization.csv`. Reuses the
fixed harness wholesale. Needs subj02/03 data staged (`NSD_SUBJ02_PATH`/`NSD_SUBJ03_PATH`).

## Open items

Append-only single-row history, the `idea`/`summary` columns, the `status` error net
(`evaluate_on_split_safe`), the trained-AlexNet baseline + beat-AlexNet goal, and the
reorganized open-ended search (`runs/<dataset>__<date>/`, the `lab` API over `search.db`,
self-paced proposers, no ROI/barrier) have all landed; the main remaining item is a more
stable reward (multi-seed/k-fold or a tie band).
