# Copilot instructions — `neuro-autoresearch`

AlphaEvolve-style autoresearch for **neural encoding**, on the **NeuralSet /
NeuralFetch** unified data interface. An agent proposes a fixed, never-trained
model; the harness reads it out with ridge regression against recorded brain
activity; held-out `test_corr` is the reward.

## Rules

- **Edit only `evolve/model.py`.** `harness/` is fixed — the sole exception is the
  one-line `DATASET = Dataset(...)` in `harness/config.py` when switching datasets.
- **Models are never trained:** weights written directly in code, no optimizer/
  backprop. Only the ridge readout in `harness/eval.py` is fit — that *is* the eval.
- A model is a plain callable `X (N, D_in) -> activations (N, D_out)`. Prefer
  modality-agnostic models (image-only ones like `spatial_pyramid` need `stimulus=image`).
- **All data goes through NeuralSet** — no raw file/network I/O (no nibabel,
  np.load, requests, boto3) in `harness`/`evolve`/`scripts`. Keep this invariant.

## Reward

`test_corr` = mean per-target Pearson r of the ridge readout on a contiguous
**temporal** hold-out (latest `test_frac` by onset). `train_corr` is reported too
(high train + low test = overfit readout). Raise `test_corr`; small diffs are noisy.

## Run

```bash
python evolve/model.py            # two iterations on the active dataset
python evolve/model.py --model identity
```

Run from repo root (results land in `runs/`). Split loads once (~14s), reused
across proposals.

## Datasets (staged offline, switch = one config line; never `download=True` unattended)

`Li2022PetitSample` fmri/text (active) · `Allen2022MassiveSample` fmri/image ·
`Bel2026PetitListenSample` meg/text-audio · `Bel2026PetitReadSample` meg/text.
Under gitignored `~/Research/neuralset/data/`. Notes + config blocks in `runs/`.

## Open items (see `runs/OVERNIGHT_READINESS.md`)

Append-only result history (current `save_results` upserts by name), an error
safety net (`status="error"` vs crash), fail-fast on missing path, and a more
stable reward (multi-seed/k-fold or a tie band).
