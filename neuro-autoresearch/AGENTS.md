# Copilot instructions — `neuro-autoresearch`

AlphaEvolve-style autoresearch for **neural encoding**, on the **NeuralSet /
NeuralFetch** unified data interface. An agent proposes a fixed, never-trained
model; the harness reads it out with ridge regression against recorded brain
activity; held-out `test_corr` is the reward.

## Rules

- **Edit only `model.py`** (the one evolving model) and **`experiment.py`** (the
  run-config: `ITERATIONS`; `LOG_SLICES` is for non-ROI studies — NSD auto-logs its
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
ONE region chosen by `NEURO_ROI` (default `ventral`; menu: V1/hV4/FFA/PPA/EBA/VWFA +
early/ventral) — the readout still fits ALL nsdgeneral voxels and every region's
predictivity is logged as `corr_<name>`. `train_corr` is reported too (high train +
low test = overfit readout). Raise `test_corr`; small diffs are noisy.

NSD neural target: the HRF-peak response (FmriExtractor `offset=5.0s`, peak TRs in a
4s window averaged to one value per voxel) — the offset is what makes the signal real
(trained AlexNet ~0.006 at offset 0 → ~0.32 here).

## Run

```bash
python -m harness.run                          # evaluate the current model
NEURO_DATASET=nsd NEURO_ROI=PPA python -m harness.run   # NSD, reward = PPA region
NEURO_DATASET=meg python -m harness.run
```

Run from repo root (results land in `results/`). Split loads once (cached in seconds;
first NSD build re-reads the timeseries, ~2 min), reused across the run's iterations.

## Datasets (staged offline, switch = `NEURO_DATASET`; never `download=True` unattended)

`Li2022PetitSubj` fmri/audio (active, `text`) · `NsdSubj01` fmri/image (`nsd`) ·
`Bel2026PetitListenSample` meg/audio (`meg`).
Under gitignored `~/Research/neuralset/data/`. Notes live in `docs/datasets/`.

## Layout

`model.py` + `experiment.py` (editable) · `harness/{datasets,data,reward,results,run}.py`
+ `harness/studies/` (fixed) · `prompts/{program,datasets}.md` (agent reads) ·
`docs/` (humans) · `results/<study>.csv` (or `<study>__<roi>.csv` per reward region;
machine writes, gitignored).

## Open items (see `docs/overnight-readiness.md`)

Append-only history and `status` field now landed; remaining: a more stable reward
(multi-seed/k-fold or a tie band) and an explicit error safety net around `model()`.
