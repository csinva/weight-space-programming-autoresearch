# Copilot instructions — `neuro-autoresearch`

Open-ended autoresearch for **neural encoding**: an agent proposes a fixed, **never-trained**
model `X (N, D_in) -> activations (N, D_out)`; the harness reads it out with ridge regression
against recorded brain activity; held-out `test_corr` is the reward. On NSD `test_corr` is **noise-
ceiling-normalized** — mean sign-preserving `r²/NC_r2` (Algonauts fraction-of-explainable-variance)
over the reliable voxels only, NOT raw Pearson r; datasets without a repeat-based ceiling keep raw
mean r. So NSD scores are NOT comparable to pre-normalization runs or to raw-r datasets. The detailed
task handed to each search proposer lives in **`search/proposer.md`** (+ per-dataset framing in
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
- Raw/network *staging* lives in `scripts/process_dataset/`; runtime file I/O is confined to
  `harness/loaders/` (reading the in-repo `data/<name>/` tree); never in `model.py`.

## Architecture
- **`harness/`** (fixed eval): `datasets.py` (presets + the `Dataset`/`Timeline` config; pick via
  `NEURO_DATASET`) · `loaders/` (`load_split` dispatch in `__init__`; `shared.py` (RawPixels, cache
  dir, constant-col drop); `timeline.py` (the NeuralSet `Segmenter` loader) + `studies.py` (the
  `Zada2025Podcast` Study) for the timeline dataset `ieeg`; `vision/` (`nsd.py`, `tvsd.py`, `nnn.py`)
  for the pre-aligned image-table datasets) · `ridge.py` (the readout: fast eigh+LOO default / himalaya,
  via `NEURO_READOUT`) · `reward.py` (orchestration: forward → readout → the region `test_corr`) ·
  `roi.py` (one `reward_region_mask` selector over the loader's per-ROI masks + the advisory per-ROI
  breakdown) · `run.py` (the manual single-eval CLI; prints the
  row — the search loop records into `search.db` via `hub.py`) · `scripts/benchmark/` (profile the
  readout pipeline phase-by-phase under GPU contention + HTML dashboard).
- **`hub.py`** — the ONE importable API + CLI over a run's `search.db`
  (`propose` / `wait_for_result` / `champions` / `sample` / `get_best_model` /
  `get_best_model_from_other_island` / `query` + `get_next_model` / `record` / `score`). Hides the
  DB, GPU, and dataset from agents. Each proposer is an ISLAND: it reads source only from its own
  best (`get_best_model`) or one random other island (`get_best_model_from_other_island`), never the
  global champion, so backbones stay diverse. Scores (not code) ARE shared: `sample` (`random-models`)
  surfaces other islands' `test_corr`/`roi_corr`, anonymized (no id/proposer). The `VANILLA_AUTORESEARCH=1`
  env (default `0` = the full system) closes every cross-island channel (`sample` +
  `get_best_model_from_other_island`) and forces the conductor off — each proposer sees ONLY its own
  lineage, i.e. the N-independent-loops baseline.
- **`search/`** — the open-ended loop: `search.sh` scaffolds a dated run and supervises up to three
  always-on loops — the N-proposer pool (`propose.sh` → N headless Copilot CLIs via `proposer.sh`,
  each confined to its own sandbox) + the single GPU evaluator (`evaluate.sh` / `evaluator.py`) + an
  OPTIONAL **conductor** (`conduct.sh` / `conductor.md`, runs from the repo root, not sandboxed).
  The search stays self-paced and decentralized — each proposer waits only for its OWN score, no
  barrier, and the conductor never gates the queue or scoring. Every `--epoch-mins` (default 60; `0`
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
NEURO_READOUT=himalaya python -m harness.run     # reference readout (default: the ~20× faster fast ridge)
python scripts/benchmark/benchmark.py --dataset nsd   # profile the readout (3 concurrent) -> results/ + HTML
search/search.sh nsd --hours 12         # unattended search: 10 proposers + 1 evaluator + a conductor steering every 1h
search/search.sh nsd --hours 12 --effort high    # ONE effort knob for ALL proposers + conductor (low|medium|high|xhigh|max; default xhigh)
search/search.sh nsd --hours 12 --epoch-mins 0   # conductor OFF (pure decentralized pool)
```

## Datasets (runtime reads the in-repo `data/<name>/` tree; switch via `NEURO_DATASET`)
**Single source of truth:** every loader reads ONLY the in-repo `data/<name>/` store (relocate via
`NEURO_DATA_DIR`); if a dataset isn't staged there the loader raises and names its build script. The
external `~/Research/neuralset/data/` is build-time-only scratch — the raw source for the offline
`scripts/process_dataset/<name>.py` step, never touched at runtime, so no preset sets `path`. Two
families: **vision** (`nsd`/`tvsd`/`nnn`, standalone image-table readers) + **language** (`ieeg`,
NeuralSet `Segmenter`; a Huth-lab fMRI-language timeline is the next language dataset — the timeline
engine keeps its generic fMRI/audio paths for it). `ieeg` ships a slim re-segmentable tree
(per-patient high-gamma `.fif` + word onsets); the vision datasets ship pre-aligned tables — all read
from `data/<name>/`.

`nsd` — NSD prepared GLMdenoise betas, reward = **early+ventral** stream (noise-normalized; see top),
subjects **`{1,2,5,7}` staged in-repo** via `NEURO_SUBJECT` (1+2 the search reward, 5+7 held-out; other
ids raise — repack with `scripts.process_dataset.nsd`, then extend `_NSD_SUBJECTS`).
**`NEURO_SUBJECT` is the ONE subject knob**: a single id (`NEURO_SUBJECT=1`) scores that subject; a
comma list (`NEURO_SUBJECT=1,2`) makes the reward the MEAN over several subjects — scored per subject,
averaged, recorded as ONE `test_corr` + averaged `roi_corr` (per-subject values are never stored;
proposers stay blind); unset = the dataset's default subject.
**`search/search.sh nsd` defaults this to `1,2`** (search subjects); override with the `NEURO_SUBJECT`
env. Works for datasets that switch data by subject (`nsd` / `tvsd` / `ieeg`); `nnn` pools sessions so
its animals aren't separable yet. Hold out the rest for generalization
(`scripts/generalization/generalize.py` — re-evals a run's DB champion on the held-out units; the
per-dataset search/held-out split lives in `scripts/generalization/splits.py`). `NSD_NCSNR_MIN` (default `0.2`, the field-standard `NC_r`
reliability cut; `0` keeps all *reliable* voxels — an absolute `NC_r2` floor still drops near-zero-
ceiling voxels so the `r²/NC_r2` reward can't explode) shortlists the scored voxels. A blob without
provisioned `ncsnr` raises — (re)run `python -m scripts.process_dataset.nsd` to fetch+align it.
`ieeg` — the 'Podcast' ECoG timeline study (human intracranial high-gamma to spoken language). Stimulus
is **`chars`** (each word → char-id grid of its N-gram, so `model.py` builds linguistic structure
itself — no pretrained embeddings); reward = **5-fold rotating temporal CV** over all electrodes (raw
mean r, no ceiling; the CV blocks a single fixed hold-out from being grep'd), multi-subject mean via
`NEURO_SUBJECT` (patients 01-03 the search reward, 04-05 held out). Stage a patient with
`scripts.process_dataset.audio_neural --subject NN`. · `tvsd` / `nnn` — macaque spiking tables. `nnn`
is **noise-ceiling-normalized** (r²/NC_r2, NC_r = the release's per-unit split-half `reliability_best`)
like NSD — so `nnn` scores are NOT comparable to pre-normalization raw-r runs. Reward = reliable units
(`NNN_MIN_RELIABILITY` 0.4) pooled; `NNN_UNIT_TYPE` (unset=all, default; `=1` → single units only,
a robustness check) subsets by BombCell class (1=single unit, 2=MUA, 3=non-somatic). Never
`download=True` unattended.
