# Copilot instructions — `neuro-autoresearch`

Open-ended autoresearch for **neural encoding**: an agent proposes a fixed, **never-trained**
model `X (N, D_in) -> activations (N, D_out)`; the harness reads it out with ridge regression
against recorded brain activity; held-out `test_corr` is the reward. The detailed task handed to
each search proposer lives in **`search/proposer.md`** (+ per-dataset framing in
`search/proposer.sh`). This file is the high-level codebase map.

## Invariants (what keeps the reward honest)
- **Only `model.py` is the evolving artifact**; everything under `harness/` is fixed (operator knobs:
  `NEURO_DATASET` picks the dataset, `NEURO_READOUT` picks the ridge implementation). If the harness
  could be edited, the reward would be meaningless.
- **Never trained:** weights are analytic / random-but-fixed in code; only the ridge readout
  (`harness/reward.py`) is fit — that *is* the eval. The model only ever receives `X`, never the
  neural target `Y`. The readout (`harness/ridge.py`, both implementations + the switch) is
  `NEURO_READOUT=fast` (default; one eigendecomposition + closed-form leave-one-out alpha selection,
  ~20× faster on GPU) or `NEURO_READOUT=himalaya` (the reference himalaya `RidgeCV`, 5-fold SVD).
  Both fit per-target alpha;
  their `test_corr` agree to ~1e-4. `fast` avoids himalaya's per-fold cuSOLVER SVD (~12 s each), which
  was the entire readout cost and serialized ~3× across concurrent evals.
- **Per-sample** activations (no statistic across the batch); representation `D_out < 10000`.
- Raw file/network I/O is confined to `harness/loaders/`; never in `model.py`.

## Architecture
- **`harness/`** (fixed eval): `datasets.py` (presets; pick via `NEURO_DATASET`) · `loaders/`
  (`load_split` dispatch in `__init__`; `shared.py` (RawPixels, cache dir, study-path, constant-col
  drop); `neuralset/` (Segmenter loader + `studies.py` Study subclasses + `roi.py` the NSD ROI
  resolver) for text/meg/ieeg; `standalone/` (`nsd.py`, `tvsd.py`, `nnn.py`) for the pre-aligned
  table datasets) · `reward.py` (himalaya ridge → the region `test_corr`) · `results.py` / `run.py`
  (the manual single-eval path).
- **`hub.py`** — the ONE importable API + CLI over a run's `search.db`
  (`propose` / `wait_for_result` / `champions` / `sample` / `get_best_model` /
  `get_best_model_from_other_island` / `query` + `get_next_model` / `record` / `score`). Hides the
  DB, GPU, and dataset from agents. Each proposer is an ISLAND: it reads source only from its own
  best (`get_best_model`) or one random other island (`get_best_model_from_other_island`), never the
  global champion, so backbones stay diverse.
- **`search/`** — the open-ended loop: `search.sh` scaffolds a dated run and supervises up to three
  always-on loops — the N-proposer pool (`propose.sh` → N headless Copilot CLIs via `proposer.sh`,
  each confined to its own sandbox) + the single GPU evaluator (`evaluate.sh` / `evaluator.py`) + an
  OPTIONAL **conductor** (`conduct.sh` / `conductor.md`, runs from the repo root, not sandboxed).
  The search stays self-paced and decentralized — each proposer waits only for its OWN score, no
  barrier, and the conductor never gates the queue or scoring. Every `--epoch-mins` (default 30; `0`
  disables it) the conductor runs one headless pass that reads the board and writes ONE broad,
  ADVISORY `direction.txt` per proposer slot — keeping the pool diverse and unstuck; proposers read
  their `direction.txt` to bias what they explore. `proposer.md` / `conductor.md` are the agent
  prompts; `repair.sh` auto-fixes small model bugs; `lib.sh` holds the supervisor primitives shared
  by `search.sh` + `propose.sh`.
- **`runs/<dataset>__<date>/`** (gitignored) — one search: `search.db` (the only shared memory +
  queue) + `models/<id>.py` (immutable archive) + `proposers/pNN/` sandboxes (seeded from the
  random-projection baseline `model.py`, each with a `direction.txt`) + `eval/` + `conductor.log`.
  Same date resumes losslessly; a new date starts fresh.
- **`model.py`** — the root random-projection baseline (the manual-path model + the proposer seed). **`docs/`** — human docs.

## Run
```bash
python -m harness.run                   # evaluate the current model.py once (manual)
search/search.sh nsd --hours 12         # unattended search: 10 proposers + 1 evaluator + a conductor steering every 30m
search/search.sh nsd --hours 12 --effort high    # ONE effort knob for ALL proposers + conductor (low|medium|high|xhigh|max; default xhigh)
search/search.sh nsd --hours 12 --epoch-mins 0   # conductor OFF (pure decentralized pool)
```

## Datasets (staged offline under `~/Research/neuralset/data/`; switch via `NEURO_DATASET`)
`nsd` — NSD prepared GLMdenoise betas, reward = **ventral stream**, all 8 subjects via
`NEURO_SUBJECT` · `text` / `meg` / `ieeg` — NeuralSet timeline studies · `tvsd` / `nnn` — macaque
spiking tables. Never `download=True` unattended.
