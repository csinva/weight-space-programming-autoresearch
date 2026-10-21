# neuro-autoresearch

Search for fixed, interpretable image representations using **RSA feedback**, then
measure transfer to a different neural dataset using **encoding**. Codex with
Astra and Antigravity (`agy`) with Argon 4 propose and revise `model.py`; a fixed
harness computes their reward.

| Track | Search feedback | Final evaluation |
| --- | --- | --- |
| fMRI | LAION-fMRI, all five subjects, RSA | NSD, all eight subjects, encoding |
| Spiking | Triple-N, all five monkeys, RSA | TVSD, both monkeys, encoding |

The loaders and metric kernels are copied from `looped-networks` revision
`3ee575baeceb1ad784102162a051b7444182fdaa`. This repository has no runtime dependency
on that checkout. See [source provenance](harness/vendor/looped/PROVENANCE.md) and
the [implementation plan](docs/migration-plan.md).

## Scientific contract

The model receives only flattened HWC RGB images in `[0, 1]`, resized to 128×128
by default. It returns a finite `(images, features)` matrix with fewer than 10,000
features. Parameters must be fixed, analytic or seeded random, with at most
100,000 materialized parameters. Model outputs must be independent of batch
composition. Only `model.py` evolves; the harness, datasets and protocol stay fixed.

**RSA search:** construct a Pearson correlation dissimilarity matrix for model
features and for each neural region, then compare their upper triangles using
average-rank Spearman correlation. No readout is fitted during search. Invalid,
constant or nonfinite results fail the attempt instead of receiving a reward.

LAION uses the shared test images with the 240 NSD-source stimuli excluded:
**1,252 images** in the current prepared release, across subjects sub-01, sub-03,
sub-05, sub-06 and sub-07. Its subject-private training images are not part of
this fixed search reward. Triple-N uses **all 1,000 images**, across M1–M5; its
source folds are combined because search fits no encoding readout. These are
search stimuli, not independent held-out tests after iterative optimization.

The primary anatomical regions match the source evaluator's default: early and
ventral visual streams. Average regions within each subject, then average
subjects equally. Triple-N M1/M4 have early and ventral coverage; M2/M3/M5 have
ventral coverage. Every subject must succeed, and its actual region scores are
retained. Source filters are fixed: ncsnr >0.2 for LAION/NSD, reliability >0.4 for
Triple-N, oracle >0.4 for TVSD. These are **raw RSA and raw Pearson encoding**;
old noise-normalized encoding scores are not comparable.

**Final encoding:** freeze the model selected solely by search RSA, then use the
paired dataset's official train/test image split. Center features with training
statistics and normalize their overall training scale; z-score neural targets
with training statistics. Each target chooses its ridge penalty using five-fold
training CV over the source's 20-alpha grid. Report held-out Pearson r, aggregated
by region then subject. NSD and TVSD never supply a search reward.

This is cross-dataset transfer, with known NSD-source LAION images excluded. It
does not claim a comprehensive pixel-level duplicate audit across releases.

## Setup on this machine

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The working checkout already has a `.venv` with runtime and test dependencies.
Search shell scripts prefer `.venv/bin/python`; override with `NEURO_PYTHON`.
The installed CLIs resolve `codex:astra` to `gpt-6-astra` and `agy:argon4` to
`gemini-4-argon-high`. `search/config.env` holds the roster and role defaults;
`NEURO_CODEX_MODEL` and `NEURO_AGY_MODEL` override exact IDs. Provider authentication
must already be configured locally. Preflight checks availability without
submitting model requests.

Prepared data is reused in place; no dataset download or multi-gigabyte copy is
needed. Defaults are the existing shared files:

| Dataset | Response directory environment variable | Default |
| --- | --- | --- |
| LAION-fMRI | `LAION_DATA_DIR` | `/data/shared/datasets/laion_fmri/processed` |
| Triple-N | `TRIPLEN_DATA_DIR` | `/data/shared/datasets/triplen/processed` |
| NSD | `NSD_DATA_DIR` | `/data/shared/datasets/nsd_processed` |
| TVSD | `TVSD_DATA_DIR` | `/data/shared/datasets/papale2025.tvsd/processed` |

Stimulus paths follow the source loaders: `LAION_FMRI_DIR`, `NSD_ROOT`,
`NSD_STIMULI_HDF5_PATH`, and `TVSD_DATA_DIR`. NSD's ncsnr volumes are also needed
for its final reliability filter. Paths can be set in the shell or local `.env`.
The old `scripts/prepare_data` baked formats are retired for these vision tasks.

## Run

```bash
# Validate Python, data paths, fixed cohorts, and both headless providers.
search/run.sh laion --preflight
search/run.sh triplen --preflight

# Manually evaluate the seed model on the complete search cohort.
.venv/bin/python -m harness.run --dataset laion --stage search
.venv/bin/python -m harness.run --dataset triplen --stage search

# Launch autonomous search (these commands make actual model requests).
search/run.sh laion --hours 12 --fleet 4 --gpu 0
search/run.sh triplen --hours 12 --fleet 4 --gpu 0

# After search: freeze its RSA champion and evaluate the paired OOD dataset.
.venv/bin/python -m experiments.evaluation.final --run runs/laion__2026-10-21
.venv/bin/python -m experiments.evaluation.final --run runs/triplen__2026-10-21
```

`laion-fmri`/`laion_fmri` and `nnn`/`triple-n` are accepted aliases. The only
search datasets are `laion` and `triplen`; `search/run.sh nsd` and `tvsd` fail.
The full search cohort is mandatory. Use `--epoch-mins 0` to disable the conductor.
On this one-GPU machine, launch the two searches sequentially unless deliberately
sharing the GPU. `NEURO_EVALUATORS` controls queue workers.

`python experiments/evaluation/run.py` is an alias for the final evaluation CLI.
For an explicit, unsearched model, use `--model model.py --search-dataset laion`;
the artifact records that no RSA champion selection occurred. Full final
artifacts go to `results/final/<protocol>/<model-hash>/`, containing frozen source,
subject scores, region scores and exact data provenance. An existing artifact is
not overwritten. Final results never update `search.db`.

## Runs and reproducibility

A run lives in `runs/<dataset>__<date>/` and contains `search.db`, immutable model
snapshots, proposer workspaces, evaluator logs and `env.json`. The manifest
records the protocol, image policy, subjects, data-file identities, source
revision, harness hash, numerical package versions, input resolution, parameter cap
and provider settings.
Resuming with different science or modified data/code is rejected; use a new
`--date`. Historical encoding runs cannot be resumed as RSA searches.

`test_corr` remains the queue's compatibility column name, but each current row
is explicitly tagged `metric=rsa`, `stage=search`, and the protocol version.
Per-subject feedback is retained. Final evaluation verifies the selected model's
SHA256 against the scored snapshot before it opens final data. File identities
use path, size and modification time; frozen model and harness identities use
SHA256. Provider logs retain their native usage events; unknown monetary cost is
stored as null rather than estimated from obsolete Copilot credits.

The existing source/network guards and agent instructions support reward
integrity; they are not a hostile-code security boundary. Provider workspace
permissions restrict writes, but final-dataset read isolation is enforced by the
protocol and scoring interfaces, not a separate OS account or data-access service.

## Layout and validation

- `harness/protocols.py`: dataset pairs, cohorts and versioned contracts.
- `harness/vendor/looped/`: copied source loaders and metric kernels.
- `harness/loaders/vision/looped.py`: source-to-model tensor adapter and provenance.
- `harness/metrics.py`, `harness/reward.py`: validated scoring and aggregation.
- `search/providers.py`: shared Codex/Antigravity execution adapter.
- `hub.py`, `search/`: queue, proposers, evaluators and optional conductor.
- `experiments/evaluation/final.py`: frozen-model OOD encoding.

Run `.venv/bin/python -m pytest -q`. Tests cover source metric agreement, tied
ranks, training-only encoding selection, alignment, invalid rewards, cohort
completeness, resume identity, provider commands and final/search separation.
See [migration validation](docs/migration-validation.md) for checks and actual-data
smoke results. Historical champions, schema readers and plotting utilities are retained under
`runs/best/` and the `legacy_*` evaluation modules. Older HTML concept documents
and language experiments describe the pre-migration system; they are not current
vision launch instructions. Language loaders are optional (`pip install -e
'.[language]'`) and are outside the two new search protocols.
