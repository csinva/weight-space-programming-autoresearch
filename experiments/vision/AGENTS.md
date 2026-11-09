# neuro-autoresearch

Carefully follow the instructions in `docs/technical-writing.md`

Open-ended agentic search for computational models of the visual cortex, aiming
to discover and understand its computational principles. Read `README.md` for
the current search workflow, evaluation protocols and commands.

## Codebase

- `model.py`: starting candidate, exporting callable `MODEL` that returns image features.
- `src/search/run.py`: launcher and preflight; `config.env`: defaults; `agents.py` and `providers.py`: agent supervision, workspaces and provider execution.
- `src/search/queue.py` and `api.py`: candidate storage and restricted feedback; `worker.py` and `score.py`: queued and single-model objective evaluation.
- `src/search/notes.py`: resumable working-note templates; `src/search/prompts/`: proposer/coordinator instructions; proposer instructions are copied into workspace `AGENTS.md`, `GEMINI.md` and `CLAUDE.md` files.
- `src/search/costs.py` and `pricing.json`: usage accounting and fixed price snapshot.
- `src/neuro_eval/`: scientific contracts (`benchmarks.py`), data loading (`datasets.py`, `readers.py`), model validation/isolation (`model.py`), metrics (`rsa.py`, `encoding.py`) and shared evaluation (`_evaluation.py`).
- `src/final_eval.py`: select and freeze the search winner, then evaluate it on held-out subjects and images (TVSD for Triple-N).
- `src/baselines/` and `src/compress.py`: fixed reference comparisons and natural-language reconstruction; shared scoring lives in `src/neuro_eval/comparison.py`, with scoped evaluation settings in `src/neuro_eval/environment.py`.
- `src/analysis/families.py`: per-voxel Shapley attribution of the LAION champion erjpu over six computational families (edges, color, surface shape, corners/contours, texture statistics, global layout).
- `dashboard/`: read-only monitoring UI/API; `tests/`: scientific and operational checks.
- `docs/`: operations, protocol, comparisons and technical-writing guidance.
- `artifacts/search/` and `artifacts/final/`: generated run state and final artifacts; `artifacts/logs/`: operational logs; `artifacts/archive/{runs,results,local}/`: legacy contents. All are untracked.
- `.env`: optional machine-local configuration, retained at the repository root and untracked.

Install with `pip install -e .` before running module commands. Package imports
stay unchanged; use `python -m search.run`, `python -m final_eval` and
`python -m compress`. Research code lives in `src/`, monitoring in root
`dashboard/`, and generated outputs in `artifacts/`.

## Working agreements

- When creating Git commits or otherwise acting on GitHub for the user, use the user's name, Yash Mehta. Before committing, verify that the Git author name is `Yash Mehta`.
- During autonomous model search, only `model.py` and the proposer's notes/scratch are editable. Infrastructure maintenance is a separate task.
- Search defaults to LAION, `--region visual` (every early and ventral unit pooled) and `--metric encoding`; `--region early|ventral` and `--metric rsa` are supported, and Triple-N defaults to ventral (IT, all five monkeys); Triple-N early/visual use M1 and M4, the only monkeys with early units. Region/metric/dataset are fixed per run. Reward is the pooled score over the selected units, averaged over subjects; per-stream and subregion values are diagnostics only.
- LAION-fMRI, NSD and Triple-N supply search feedback. LAION and NSD hold out subjects and images within the dataset: search uses LAION sub-01, 03, 05 or NSD subjects 1–4 on one half of the shared images (split within each image source); final evaluation uses LAION sub-06, 07 or NSD subjects 5–8 on the other half. Readouts are fit on each subject's unique images only. Held-out subjects, held-out images, TVSD responses, scores and final artifacts are final-only; never use them to guide search, repair, coordination or model selection.
- Keep all subjects, fixed image policies, source filters, source metric kernels and explicit protocol provenance intact when maintaining `src/neuro_eval/`.
- Do not commit run databases, data files, caches, credentials or machine-local `.env` settings.
