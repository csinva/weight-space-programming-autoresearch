# neuro-autoresearch

Find fixed, untrained image representations with autonomous **RSA search**, then
freeze the winner and measure transfer with **encoding evaluation**.
Only the candidate `model.py` evolves during search; the scientific code stays fixed.

| Track | Phase 1: search feedback | Phase 2: final evaluation |
| --- | --- | --- |
| fMRI | LAION-fMRI, all five subjects, RSA | NSD, all eight subjects, encoding |
| Spiking | Triple-N, all five monkeys, RSA | TVSD, both monkeys, encoding |

NSD and TVSD data and scores must never guide search, coordination, repair or model
selection. A high search score is predictive evidence, not a mechanistic explanation.

## Run

Use Python 3.12 or newer, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Print settings, then check data, providers and isolation without model requests.
python -m search.run --show-config
python -m search.run laion --preflight

# Phase 1: autonomous search (sends real model requests).
python -m search.run laion --hours 12 --gpu 0

# Phase 2: freeze the RSA winner and evaluate it on NSD.
# Use the run directory printed by the search launcher.
python final_eval.py --run search_runs/laion__2026-10-22
```

For the spiking track, replace `laion` with `triplen`; final evaluation chooses
TVSD automatically. Run tracks sequentially on a single GPU. Use `--date YYYY-MM-DD`
to name a run; relaunching the same date resumes it if code, data and scientific
settings still match. Ctrl+C stops the search. `--epoch-mins 0` disables the
coordinator. Start a new run after changing the evaluation code.

The default proposer roster is Astra, Opus and Argon at medium effort, with an
Astra xhigh coordinator and Astra medium repair. Defaults live in
[`search/config.env`](search/config.env). Provider CLIs reuse existing machine
logins, with permission prompts disabled inside the outer isolation boundary.
The launcher selects agy's ADC authentication and clears conflicting Claude
process-local authentication overrides; it does not change global login settings.
Missing providers, data or usable Bubblewrap isolation fail preflight.

Other entry points:

```bash
# Score one model on a complete RSA search cohort.
python -m search.score --dataset laion --model model.py

# Evaluate an explicitly supplied frozen model, recording that selection method.
python final_eval.py --model model.py --search-dataset laion

# Optional read-only dashboard over search_runs/.
pip install -e '.[dashboard]'
python -m dashboard

# Tests use fake providers and synthetic data; no live model requests.
python -m pytest -q
```

## Repository map

| Path | Responsibility |
| --- | --- |
| `model.py` | Starting candidate, exporting `MODEL` |
| `search/run.py` | Configure, validate, start and stop autonomous search |
| `search/agents.py`, `providers.py` | Agent loops, workspaces and provider execution |
| `search/queue.py`, `api.py` | Candidate storage and restricted search feedback |
| `search/worker.py`, `score.py` | Process queued models and request RSA scores |
| `search/prompts/` | Proposer and coordinator instructions |
| `final_eval.py` | Select, freeze and evaluate a winner; write final artifacts |
| `neuro_eval/rsa.py`, `encoding.py` | Scientific RSA and encoding evaluation |
| `neuro_eval/_evaluation.py` | Shared input checks, cohort execution and aggregation |
| `neuro_eval/datasets.py` | Data loading, alignment, caching and provenance |
| `neuro_eval/benchmarks.py` | Dataset pairs, full cohorts and fixed scientific rules |
| `neuro_eval/model.py` | Model validation, isolated execution and code identity |
| `neuro_eval/reference/` | Original readers and metric kernels copied from looped-networks |
| `dashboard/` | Search monitoring backend and bundled UI |
| `tests/` | Scientific and operational contract checks |
| `search_runs/` | Generated search databases, candidate snapshots, notes and logs |
| `final_results/` | Frozen selected models, selection records and final scores |

The copied source has no runtime dependency on its original checkout. Its exact
revision and adaptations are recorded in
[`neuro_eval/reference/PROVENANCE.md`](neuro_eval/reference/PROVENANCE.md).
The dashboard uses a prebuilt, locally adapted
[CORAL](https://github.com/Human-Agent-Society/CORAL) frontend.

## Scientific rules

Models receive flattened HWC RGB images in `[0, 1]`, resized to 128×128 by default,
and return a finite `(images, features)` matrix with fewer than 10,000 features.
Parameters must be fixed, analytic or seeded random, with at most 100,000
materialized parameters. Outputs must be independent of batch composition.

**RSA:** compare the upper triangles of model and neural Pearson-correlation
dissimilarity matrices using average-rank Spearman correlation. No readout is
fitted. Constant or nonfinite primary results fail the candidate.

- LAION uses shared test images excluding all NSD-source stimuli: 1,252 images
  in the current prepared release, for sub-01, sub-03, sub-05, sub-06 and sub-07.
  Subject-private training images are excluded.
- Triple-N combines its source folds and uses all 1,000 images for M1–M5.
- Use the source's early and ventral streams. Triple-N M1/M4 have both;
  M2/M3/M5 have ventral coverage. Average actual prescribed regions within each
  subject, then average all required subjects equally. A missing subject fails
  evaluation.
- Fixed source filters are ncsnr >0.2 for LAION/NSD, reliability >0.4 for Triple-N,
  and oracle >0.4 for TVSD.

Search feedback includes primary RSA and subject/region coverage. LAION also
reports source-group RSA for `laion`, `thingsplus` and `ood`, using only allowed
search images. A subgroup mean requires all subjects; unavailable diagnostics do
not alter the primary score. Repeatedly optimized search scores are not independent
held-out confirmation.

**Encoding:** freeze the selected model before loading final neural responses.
Use the paired dataset's source train/test split. Center features and normalize
their overall scale using training data; z-score targets with training statistics.
Choose a ridge penalty per neural target using five-fold training CV over the
source's 20-alpha grid. Report held-out Pearson r, averaging regions then subjects.
Both RSA and encoding are raw scores, without noise-ceiling normalization.
Known NSD-source images are excluded from LAION search; this is not a comprehensive
pixel-level duplicate audit across releases.

## Data, isolation and outputs

Prepared data is reused in place; no download or multi-gigabyte copy is needed.
Paths can be overridden in the shell or an untracked local `.env`.

| Variable | Default response file |
| --- | --- |
| `LAION_DATA_DIR` | `/data/shared/datasets/laion_fmri/processed/laion_data.pkl` |
| `TRIPLEN_DATA_DIR` | `/data/shared/datasets/triplen/processed/triplen_data.pkl` |
| `NSD_DATA_DIR` | `/data/shared/datasets/nsd_processed/nsd_data.pkl` |
| `TVSD_DATA_DIR` | `/data/shared/datasets/papale2025.tvsd/processed/tvsd_data.pkl` |

Overrides name directories; filenames stay fixed except for `NSD_DATA_FILE`.
Stimulus paths follow the source readers: `LAION_FMRI_DIR`, `NSD_ROOT`,
`NSD_STIMULI_HDF5_PATH` and `TVSD_DATA_DIR`. NSD also requires the source ncsnr
volumes for reliability filtering.

Search agents receive a private workspace and role-bound API. They can submit
candidate source and inspect bounded search feedback, but cannot read neural
responses, the queue database, other workspaces or final results directly.
The coordinator may set directions and reset proposer lineages. Repairs create
new child candidates; archived source is never overwritten. Independent-loop
mode restricts feedback to each proposer's own island.

Provider processes have network access for model requests. Candidate execution
has separate filesystem, process and network isolation and receives only source,
image inputs, installed runtime and necessary GPU access. Missing isolation support
fails closed. Numeric outputs are loaded without pickle; sampled singleton/batch
checks enforce the per-image contract within numerical tolerance. Static checks
and parameter counts supplement isolation; they cannot prove arbitrary code is
untrained or explainable.

Each search run records its benchmark, source/code identity, runtime versions and
data-file identities in `env.json`. Candidate SHA256 hashes identify the bytes
that were scored. Final evaluation verifies the winner and writes `selection.json`,
a frozen `model.py` and `evaluation.json` under `final_results/`. It never writes
scores to the search database, and existing final artifacts are not overwritten.
Run databases, data, outputs, credentials and caches must stay out of Git.
