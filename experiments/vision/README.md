# neuro-autoresearch

Open-ended agentic search for computational models of the visual cortex. Agents
change `model.py` to discover image representations, score them against neural
responses, then freeze the winner for final evaluation. The scientific code stays
fixed during search. A high score is predictive evidence, not a mechanistic explanation.

Held-out subjects, held-out images, TVSD responses and final artifacts must never
guide search, repair, coordination or model selection. LAION and NSD hold out both
subjects and images; Triple-N search is followed by TVSD final evaluation. Readouts
for LAION and NSD are fit on each subject's unique images only. See the
[scientific protocol](docs/protocol.md) for cohorts, splits and scoring rules.

## Setup and search

Use Python 3.12 or newer. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m search.run --show-config
python -m search.run --preflight
python -m search.run --gpu 0
```

Preflight checks data, provider authentication and Bubblewrap isolation without
model requests. Search sends real model requests and defaults to 12 hours of
LAION visual encoding. `--dataset nsd|triplen`, `--region early|ventral|visual`,
`--metric rsa|encoding` and `--hours` select alternatives; Triple-N defaults to
ventral. Run one configuration at a time on a single GPU. Ctrl+C stops a run;
relaunching the same configuration and date resumes it only when code, data and
scientific settings still match.

See [operations](docs/operations.md) for provider setup, Claude authentication,
data paths, configuration, resuming, costs, isolation and the dashboard.

## Final evaluation

Use the run directory printed by the launcher. Selection and freezing happen
before held-out responses are loaded; final evaluation always uses encoding.

```bash
python final_eval.py --run search_runs/laion__visual__encoding__2026-11-05
# Or evaluate an explicitly supplied frozen model:
python final_eval.py --model model.py --search-dataset laion --region visual
```

Final artifacts go to `final_results/` and never enter search feedback. Existing
artifacts are not overwritten. [Protocol details](docs/protocol.md) explain the
headline scores and diagnostic subject × image comparisons.

## Baseline comparisons and compression

Fixed baselines and frozen models use the same scientific evaluator. Compression
tests how well independent models rebuilt from a short description retain the
original model's score and representation.

```bash
pip install -e '.[baselines]'
python -m baselines.run --dataset laion --region visual
python -m baselines.run --dataset laion --region visual --final
python compress.py --model path/to/model.py --words 200 --builders 4
```

Results stay in `final_results/baselines/` and `final_results/compression/`.
See [comparisons](docs/comparisons.md) for available baselines, frozen-model
comparisons, word budgets and retention scores.

## Repository map

| Path | Responsibility |
| --- | --- |
| `model.py` | Starting candidate exporting `MODEL` |
| `search/` | Search launcher, agents, queue, scoring, configuration and costs |
| `final_eval.py` | Select, freeze and evaluate the winner |
| `baselines/`, `compress.py` | Reference comparisons and description-based rebuilding |
| `neuro_eval/` | Fixed scientific contracts, data loading, model isolation and evaluation |
| `neuro_eval/comparison.py` | Shared scoring for baseline and compression comparisons |
| `dashboard/` | Read-only search monitoring |
| `tests/` | Scientific and operational checks |
| `search_runs/`, `final_results/` | Generated search state and final artifacts; untracked |

The [full map](docs/repository.md) identifies individual modules. Run checks with
`python -m pytest -q`; tests use fake providers and synthetic data, without live
model requests. Keep run databases, data, caches, credentials and machine-local
`.env` settings out of Git.
