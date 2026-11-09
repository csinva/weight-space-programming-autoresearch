# Repository map

[README](../README.md) · [Operations](operations.md) · [Scientific protocol](protocol.md) · [Comparisons](comparisons.md)

| Path | Responsibility |
| --- | --- |
| `model.py` | Starting candidate, exporting `MODEL` |
| `src/search/run.py` | Configure, validate, start and stop autonomous search |
| `src/search/agents.py`, `providers.py` | Agent loops, workspaces and provider execution |
| `src/search/costs.py`, `pricing.json` | Token accounting and fixed API price catalog |
| `src/search/queue.py`, `api.py` | Candidate storage and restricted search feedback |
| `src/search/worker.py`, `score.py` | Process queued models and score the configured objective |
| `src/search/prompts/` | Proposer and coordinator instructions |
| `src/baselines/` | Fixed reference models and the baseline comparison CLI |
| `src/compress.py` | Describe, rebuild and compare models |
| `src/neuro_eval/comparison.py` | Shared trusted-model comparison scoring |
| `src/neuro_eval/environment.py` | Scoped search and final evaluation environment settings |
| `src/final_eval.py` | Select, freeze and evaluate a winner; write final artifacts |
| `src/neuro_eval/rsa.py`, `encoding.py` | Scientific RSA and encoding evaluation |
| `src/neuro_eval/_evaluation.py` | Shared input checks, cohort execution and aggregation |
| `src/neuro_eval/datasets.py` | Data loading, alignment, caching and provenance |
| `src/neuro_eval/readers.py` | Dataset-file readers, reliability filtering and lazy image access |
| `src/neuro_eval/benchmarks.py` | Dataset pairs, full cohorts and fixed scientific rules |
| `src/neuro_eval/model.py` | Model validation, isolated execution, code identity and trusted standalone imports |
| `dashboard/` | Search monitoring backend and bundled UI |
| `tests/` | Scientific and operational contract checks |
| `docs/` | Operations, protocol, comparisons and technical-writing guidance |
| `.env` | Optional machine-local configuration; stays at the repository root and out of Git |
| `artifacts/search/` | Generated search databases, candidate snapshots, notes and logs |
| `artifacts/final/` | Frozen selected models, selection records and final scores |
| `artifacts/logs/` | Top-level operational logs |
| `artifacts/archive/runs/`, `artifacts/archive/results/`, `artifacts/archive/local/` | Archived legacy `runs/`, `results/` and `.local/` contents |

Research code lives under `src/`, with the monitoring application in root
`dashboard/`. Editable installation keeps import names and module commands unchanged (`search`, `neuro_eval`, `baselines`, `dashboard`,
`final_eval` and `compress`). The starting candidate stays at root `model.py`.
Generated artifacts are untracked.

The supported layout is `src/` for research code, root `dashboard/` for
monitoring, and `artifacts/` for generated outputs.

Baseline and compression comparisons import trusted models in-process. Search candidates and the frozen winner use isolated execution.

The dashboard uses a prebuilt, locally adapted
[CORAL](https://github.com/Human-Agent-Society/CORAL) frontend.
