# Copilot instructions — `neuro-autoresearch`

Open-ended autoresearch for **neural encoding**: an agent proposes a fixed, **never-trained**
model `X (N, D_in) -> activations (N, D_out)`; the harness reads it out with ridge regression
against recorded brain activity; held-out `test_corr` is the reward. The detailed task handed to
each search proposer lives in **`search/proposer.md`** (+ per-dataset framing in
`search/proposer.sh`). This file is the high-level codebase map.

## Invariants (what keeps the reward honest)
- **Only `model.py` is the evolving artifact**; everything under `harness/` is fixed (the one knob
  is `NEURO_DATASET`). If the harness could be edited, the reward would be meaningless.
- **Never trained:** weights are analytic / random-but-fixed in code; only the ridge readout
  (`harness/reward.py`, himalaya `RidgeCV` on GPU) is fit — that *is* the eval. The model only ever
  receives `X`, never the neural target `Y`.
- **Per-sample** activations (no statistic across the batch); representation `D_out < 10000`.
- Raw file/network I/O is confined to `harness/loaders/`; never in `model.py`.

## Architecture
- **`harness/`** (fixed eval): `datasets.py` (presets; pick via `NEURO_DATASET`) · `loaders/`
  (`load_split` dispatch in `__init__`; one reader per dataset family + `common.py` shared helpers
  & `studies.py` NeuralSet subclasses) · `reward.py` (himalaya ridge → the region `test_corr`) ·
  `results.py` / `run.py` (the manual single-eval path).
- **`lab.py`** — the ONE importable API + CLI over a run's `search.db`
  (`propose` / `wait_for_result` / `champions` / `sample` / `get_model` / `query` +
  `get_next_model` / `record` / `score`). Hides the DB, GPU, and dataset from agents.
- **`search/`** — the open-ended loop: `search.sh` scaffolds a dated run and supervises two
  always-on, decentralized loops — the N-proposer pool (`propose.sh` → N headless Copilot CLIs via
  `proposer.sh`, each confined to its own sandbox) + the single GPU evaluator
  (`evaluate.sh` / `evaluator.py`). Self-paced (each proposer waits only for its OWN score); no
  conductor, no barrier. `proposer.md` is the task; `repair.sh` auto-fixes small model bugs;
  `lib.sh` holds the supervisor primitives shared by `search.sh` + `propose.sh`.
- **`runs/<dataset>__<date>/`** (gitignored) — one search: `search.db` (the only shared memory +
  queue) + `models/<id>.py` (immutable archive) + `proposers/pNN/` sandboxes (seeded from the
  random-projection baseline `model.py`) + `eval/`. Same date resumes losslessly; a new date starts fresh.
- **`model.py`** — the root random-projection baseline (the manual-path model + the proposer seed). **`docs/`** — human docs.

## Run
```bash
python -m harness.run                   # evaluate the current model.py once (manual)
search/search.sh nsd --hours 12         # launch the unattended search (10 proposers + 1 evaluator)
```

## Datasets (staged offline under `~/Research/neuralset/data/`; switch via `NEURO_DATASET`)
`nsd` — NSD prepared GLMdenoise betas, reward = **ventral stream**, all 8 subjects via
`NEURO_SUBJECT` · `text` / `meg` / `ieeg` — NeuralSet timeline studies · `tvsd` / `nnn` — macaque
spiking tables. Never `download=True` unattended.
