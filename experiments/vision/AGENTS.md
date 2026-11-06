# neuro-autoresearch

Carefully follow the instructions in `SKILL-TECHNICAL-WRITING.md`

Open-ended agentic search for computational models of the visual cortex, aiming
to discover and understand its computational principles. Read `README.md` for
the current search workflow, evaluation protocols and commands.

## Codebase

- `model.py`: starting candidate, exporting callable `MODEL` that returns image features.
- `search/run.py`: launcher and preflight; `config.env`: defaults; `agents.py` and `providers.py`: agent supervision, workspaces and provider execution.
- `search/queue.py` and `api.py`: candidate storage and restricted feedback; `worker.py` and `score.py`: queued and single-model objective evaluation.
- `search/notes.py`: resumable working-note templates; `search/prompts/`: proposer/coordinator instructions; proposer instructions are copied into workspace `AGENTS.md`, `GEMINI.md` and `CLAUDE.md` files.
- `search/costs.py` and `pricing.json`: usage accounting and fixed price snapshot.
- `neuro_eval/`: scientific contracts (`benchmarks.py`), data loading (`datasets.py`, `readers.py`), model validation/isolation (`model.py`), metrics (`rsa.py`, `encoding.py`) and shared evaluation (`_evaluation.py`).
- `final_eval.py`: select and freeze the search winner, then evaluate on the paired final dataset.
- `dashboard/`: read-only monitoring UI/API; `tests/`: scientific and operational checks.
- `search_runs/` and `final_results/`: generated run state and final artifacts; untracked.

## Working agreements

- When creating Git commits or otherwise acting on GitHub for the user, use the user's name, Yash Mehta. Before committing, verify that the Git author name is `Yash Mehta`.
- During autonomous model search, only `model.py` and the proposer's notes/scratch are editable. Infrastructure maintenance is a separate task.
- Search defaults to LAION, `--region visual` (every early and ventral unit pooled) and `--metric encoding`; `--region early|ventral` and `--metric rsa` are supported, and Triple-N defaults to ventral (IT, all five monkeys); Triple-N early/visual use M1 and M4, the only monkeys with early units. Region/metric/dataset are fixed per run. Reward is the pooled score over the selected units, averaged over subjects; per-stream and subregion values are diagnostics only.
- LAION-fMRI and Triple-N supply search feedback. LAION includes all image sources, including NSD-source stimuli with LAION responses. NSD and TVSD neural responses, scores and final artifacts are final-only; never use them to guide search, repair, coordination or model selection. Final NSD evaluation is not image-held-out from LAION search.
- Keep all subjects, fixed image policies, source filters, source metric kernels and explicit protocol provenance intact when maintaining `neuro_eval/`.
- Do not commit run databases, data files, caches, credentials or machine-local `.env` settings.
