# Repository map

[README](../README.md) · [Operations](operations.md) · [Scientific protocol](protocol.md) · [Comparisons](comparisons.md)

| Path | Responsibility |
| --- | --- |
| `model.py` | Starting candidate, exporting `MODEL` |
| `search/run.py` | Configure, validate, start and stop autonomous search |
| `search/agents.py`, `providers.py` | Agent loops, workspaces and provider execution |
| `search/costs.py`, `pricing.json` | Token accounting and fixed API price catalog |
| `search/queue.py`, `api.py` | Candidate storage and restricted search feedback |
| `search/worker.py`, `score.py` | Process queued models and score the configured objective |
| `search/prompts/` | Proposer and coordinator instructions |
| `baselines/` | Fixed reference models and the baseline comparison CLI |
| `compress.py` | Describe, rebuild and compare models |
| `neuro_eval/comparison.py` | Shared trusted-model comparison scoring |
| `neuro_eval/environment.py` | Scoped search and final evaluation environment settings |
| `final_eval.py` | Select, freeze and evaluate a winner; write final artifacts |
| `neuro_eval/rsa.py`, `encoding.py` | Scientific RSA and encoding evaluation |
| `neuro_eval/_evaluation.py` | Shared input checks, cohort execution and aggregation |
| `neuro_eval/datasets.py` | Data loading, alignment, caching and provenance |
| `neuro_eval/readers.py` | Dataset-file readers, reliability filtering and lazy image access |
| `neuro_eval/benchmarks.py` | Dataset pairs, full cohorts and fixed scientific rules |
| `neuro_eval/model.py` | Model validation, isolated execution, code identity and trusted standalone imports |
| `dashboard/` | Search monitoring backend and bundled UI |
| `tests/` | Scientific and operational contract checks |
| `search_runs/` | Generated search databases, candidate snapshots, notes and logs |
| `final_results/` | Frozen selected models, selection records and final scores |

Baseline and compression comparisons import trusted models in-process. Search candidates and the frozen winner use isolated execution.

The dashboard uses a prebuilt, locally adapted
[CORAL](https://github.com/Human-Agent-Society/CORAL) frontend.
