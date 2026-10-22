# neuro-autoresearch

Search broadly for fixed, untrained image representations using **RSA feedback**, then
freeze the selected model and measure cross-dataset transfer using **encoding**.
Only `model.py` evolves during search; the harness and scientific protocol stay fixed.
Prediction guides search. Mechanistic interpretation and hypothesis testing are separate
later work, not prerequisites for proposing a model.

| Track | Search feedback | Final evaluation |
| --- | --- | --- |
| fMRI | LAION-fMRI, all five subjects, RSA | NSD, all eight subjects, encoding |
| Spiking | Triple-N, all five monkeys, RSA | TVSD, both monkeys, encoding |

NSD and TVSD are final-only: their scores must never guide search. Full cohorts,
fixed source filters and equal-subject aggregation are mandatory. Read the
[scientific contract](docs/protocol.md) and [access/feedback contract](docs/access-feedback.md) before changing a model or harness.

## Start here

Run from the repository root. The existing checkout has runtime and test
dependencies in `.venv`; for a fresh checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

This version changes the access and feedback contract. Start a fresh run date;
older runs remain readable but cannot resume under changed harness code.

Prepared datasets are reused in place. See [data paths](docs/data.md) if they
are not available at the shared defaults.

```bash
# Inspect the fixed roster, then validate data and providers (no model requests).
search/run_fixed.sh --show-config
search/run_fixed.sh laion --preflight

# Start search: this sends real model requests.
search/run_fixed.sh laion --hours 12 --gpu 0

# After search, freeze its RSA champion and evaluate on NSD.
# Replace the example date with the run directory printed by the launcher.
.venv/bin/python -m experiments.evaluation.final --run runs/laion__2026-10-21
```

For the spiking track, replace `laion` with `triplen`; final evaluation selects
TVSD automatically. On this one-GPU machine, run the tracks sequentially.
Relaunching the same run date resumes it; use `--date YYYY-MM-DD` for another run.
Stop with Ctrl+C. `--epoch-mins 0` disables the optional conductor.

The recommended launcher fixes Astra, Opus and Argon proposer slots at medium
effort, with an Astra xhigh conductor. It uses existing machine authentication
and skips permission prompts. See the [launcher runbook](docs/search-launcher.md)
for exact models, authentication, logs, failure recovery and configurable launches.

## Evaluate or simplify a model

```bash
# Score model.py on a complete RSA search cohort without launching agents.
.venv/bin/python -m harness.run --dataset laion --stage search

# Refactor a run's champion with activation-equivalence checks (sends model requests).
scripts/simplify/run.sh runs/laion__2026-10-21 --mode exact

# Run the test suite without live provider calls.
.venv/bin/python -m pytest -q
```

[Simplification](docs/simplification.md) supports exact and RSA-tolerant modes.
[Final evaluation](experiments/evaluation/README.md) documents frozen artifacts and
explicit-model evaluation. Final results never update the search database.

## Where to look

| Area | Entry point |
| --- | --- |
| Dataset pairs, cohorts and scientific contracts | `harness/protocols.py`, [protocol](docs/protocol.md) |
| Source loaders and metric kernels | `harness/vendor/looped/`, [provenance](harness/vendor/looped/PROVENANCE.md) |
| Model input and data provenance | `harness/loaders/vision/looped.py` |
| Scoring and aggregation | `harness/metrics.py`, `harness/reward.py` |
| Queue and autonomous search | `hub.py`, `search/`, [runbook](docs/search-launcher.md) |
| Frozen-model final encoding | `experiments/evaluation/final.py` |

The copied `looped-networks` source has no runtime dependency on its original
checkout. Tests cover numerical reference agreement, alignment, invalid rewards,
complete cohorts, run/model identity, queue recovery, provider execution and
search/final separation; provider tests use fake CLIs.

[Documentation index](docs/README.md) separates current operator guidance from
historical migration records and concept documents. Historical encoding scores
are not comparable to the current raw RSA and raw Pearson encoding scores.
