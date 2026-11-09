# neuro-autoresearch

**For the paper figure:** use [figure/README.md](figure/README.md). It provides a
one-command render from included results, the frozen five-subject LAION model,
and scripts to recompute the numerical panels. No agent or coordinator is needed.
The search instructions below describe the newer subject/image holdout protocol;
the paper figure records its historical protocol explicitly.

Open-ended agentic search for computational models of the visual cortex: discover
image-to-representation functions and the computational principles they express.
Search a selected brain region with **encoding** (default) or **RSA**, then freeze
the winner and measure how it generalizes to new subjects and new images.
Only the candidate `model.py` evolves during search; the scientific code stays fixed.

| Track | Phase 1: search feedback | Phase 2: final evaluation |
| --- | --- | --- |
| fMRI | LAION-fMRI sub-01, 03, 05 on the search image half, RSA or encoding | LAION-fMRI sub-06, 07 on the held-out image half, encoding |
| fMRI | NSD subjects 1–4 on the search image half, RSA or encoding | NSD subjects 5–8 on the held-out image half, encoding |
| Spiking | Triple-N, all five monkeys (ventral) or M1 and M4 (early/visual), RSA or encoding | TVSD, both monkeys, encoding |

LAION and NSD each hold out subjects and images within the dataset. Their shared
images (seen by every subject) are split in half within each image source, by the
SHA-256 hash of the stimulus ID: LAION's 1,492 into 745 search and 747 held-out
images (LAION, THINGSplus, NSD and OOD sources), and NSD's 907 into 453 and 454.
NSD has 907 rather than 1,000 shared images because subjects 3, 4, 6 and 8 did not
complete all sessions; each cohort pairs two subjects who did with two who did not.
Encoding readouts are fit on each subject's own unique images only. Held-out
subjects, held-out images, TVSD responses and final artifacts must never guide
search, coordination, repair or model selection.
A high search score is predictive evidence, not a mechanistic explanation.

## Run

Dependencies live in the `pyproject.toml` at the root of this repo. Run these commands from this folder:

```bash
uv sync --extra dev --extra dashboard
source "$(git rev-parse --show-toplevel)/.venv/bin/activate"

# Print settings, then check data, providers and isolation without model requests.
python -m search.run --show-config
python -m search.run --preflight

# Phase 1: autonomous search, 12 hours by default (sends real model requests).
# Default: LAION, pooled early + ventral streams (--region visual), encoding.
python -m search.run --gpu 0
python -m search.run --dataset nsd --gpu 0

# Phase 2: freeze the search winner and evaluate it on held-out subjects and images.
# Use the run directory printed by the search launcher.
python final_eval.py --run search_runs/laion__visual__encoding__2026-11-05
```

`--dataset laion`, `--region visual` and `--metric encoding` are defaults.
`visual` pools every retained early and ventral unit into one target set, so its
score is the mean over all union units (streams are not reweighted); the two
streams are disjoint and roughly equal in size for LAION. `--region early` or
`--region ventral` selects one stream; `--metric rsa` compares representational
geometry instead of a ridge readout. For the spiking track use `--dataset triplen`,
which defaults to `--region ventral`; final evaluation chooses TVSD automatically.
Triple-N ventral (IT) search uses all five monkeys. Only M1 and M4 have early
(V1, V2, V4) responses, so `--region early|visual` on Triple-N uses exactly that
two-monkey cohort, with V1, V2 and V4 as diagnostics.
Run configurations sequentially on a single GPU. Each run is stored as
`<dataset>__<region>__<metric>__<date>`; `--date YYYY-MM-DD` selects the date.
Relaunching the same configuration/date resumes only if code, data and scientific
settings still match. Search defaults to 12 hours; `--hours` overrides it. Ctrl+C
stops the search. The coordinator starts at startup and every 120 minutes thereafter
(on a fixed schedule); an occupied slot is skipped to prevent overlapping reviews.
`--epoch-mins` changes this interval and `--epoch-mins 0` disables coordination.
Each review starts a fresh provider conversation without the preceding transcript.
Its short `scratch/notes.md`, current directions and search results persist and
remain available through the workspace/API. Start a new run after changing the
evaluation code.

The default proposer roster is one Claude Code Opus 5.5 and two agy Argon agents at medium effort, with an
Astra xhigh coordinator. Proposers correct their own failed candidates; automatic
repair is off by default (the worker's `--max-repairs` flag enables it). Defaults live in
[`search/config.env`](search/config.env). Codex and agy reuse existing machine
logins, with permission prompts disabled inside the outer isolation boundary.
Claude automation requires a token from `claude setup-token`;
it never copies the interactive Claude login or its refresh credentials.
The launcher selects agy's ADC authentication and clears conflicting Claude
API-key/provider overrides while preserving this automation token. Global login
settings remain unchanged.
Missing providers, data or usable Bubblewrap isolation fail preflight.

For a Claude proposer, run `claude setup-token` once and complete its browser
authorization. Then save the generated token using the hidden terminal prompt:

```bash
python -m search.auth
NEURO_MODEL_ROSTER="claude:opus5.5:medium" python -m search.run --preflight
NEURO_MODEL_ROSTER="claude:opus5.5:medium" python -m search.run --gpu 0
```

The token uses your Claude subscription. Keep it out of Git and run artifacts;
the saved file is `~/.config/neuro-autoresearch/claude-oauth-token` (mode 600,
inside a mode-700 directory). Claude launches load it automatically across SSH
sessions, tmux panes and reboots. An explicit `CLAUDE_CODE_OAUTH_TOKEN` environment
variable takes precedence. Re-run `python -m search.auth` to replace the saved token.
It enters only the Claude sandbox through a private temporary file that is removed
before Claude starts, and is not included in command-line arguments. Missing tokens
fail preflight; preflight does not make model requests or verify server-side token
validity. Expired or revoked tokens require a replacement from `claude setup-token`.
Claude authentication failures disable the affected slot instead of restarting it.
Other proposer failures retry with exponential delays, with a bounded number of
consecutive failures; healthy proposers and evaluators continue running.

Token-cost estimates use the public standard API rates verified on October 22,
2026 in [`search/pricing.json`](search/pricing.json), including cache usage and
reasoning tokens. Each run preserves this fixed snapshot; monitoring does not
fetch prices online. Estimates are not invoices and exclude subscription costs,
special service tiers and non-token charges. Gemini Argon's introductory rates
(advertised through November 30, 2026) stay frozen until the catalog is explicitly
updated; its notes record the announced later rates.

Each run saves `usage.json` with totals by provider, role, proposer and model, backed by
individual invocation records in `usage/` and its own `pricing.json`. This includes
the coordinator and optional repair agents. `known_usd` is the cost of captured,
priceable usage; `usd` is the complete estimate, or `null` if telemetry or pricing
is missing. Interrupted sessions retain captured usage and mark the total
incomplete. Resuming adds new invocations without double counting earlier ones.
The supervisor refreshes totals during the run and on shutdown; after a hard kill,
`python -m search.costs RUN_DIRECTORY` rebuilds totals from the saved records.

Other entry points:

```bash
# Score one model on a complete search cohort.
python -m search.score --model model.py
python -m search.score --region ventral --metric rsa --model model.py

# Evaluate an explicitly supplied frozen model, recording that selection method.
python final_eval.py --model model.py --search-dataset laion --region visual

# Optional read-only dashboard over search_runs/.
python -m dashboard

# Tests use fake providers and synthetic data; no live model requests.
python -m pytest -q
```

## Baselines

`baselines/` holds fixed reference models, each a `MODELS` dict that follows the
candidate output contract: pixels/low-level statistics, Gabor wavelet pyramid
(Kay et al., 2008) with and without divisive normalization, wavelet scattering
(Mallat), Portilla–Simoncelli texture statistics on a 2×2 grid (simplifying
Freeman & Simoncelli, 2011), banana-filter curvature (a proxy for Pasupathy &
Connor), GIST, HMAX, and ImageNet-trained and seeded-random AlexNet layers.
They run in-process as trusted code, with the same evaluator, and write to
`final_results/baselines/`, never to a search run. Their settings are fixed a
priori or chosen on search data only.

```bash
uv sync --extra baselines
python -m baselines.run --dataset laion --region visual            # search protocol
python -m baselines.run --dataset laion --region visual --final    # held-out LAION subjects and images
python -m baselines.run --dataset nsd --final --model champion=final_results/.../model.py  # any champion on NSD
python -m baselines.run --dataset triplen --only gabor,hmax
```

## Repository map

| Path | Responsibility |
| --- | --- |
| `model.py` | Starting candidate, exporting `MODEL` |
| `search/run.py` | Configure, validate, start and stop autonomous search |
| `search/agents.py`, `providers.py` | Agent loops, workspaces and provider execution |
| `search/costs.py`, `pricing.json` | Token accounting and fixed API price catalog |
| `search/queue.py`, `api.py` | Candidate storage and restricted search feedback |
| `search/worker.py`, `score.py` | Process queued models and score the configured objective |
| `search/prompts/` | Proposer and coordinator instructions |
| `final_eval.py` | Select, freeze and evaluate a winner; write final artifacts |
| `neuro_eval/rsa.py`, `encoding.py` | Scientific RSA and encoding evaluation |
| `neuro_eval/_evaluation.py` | Shared input checks, cohort execution and aggregation |
| `neuro_eval/datasets.py` | Data loading, alignment, caching and provenance |
| `neuro_eval/readers.py` | Dataset-file readers, reliability filtering and lazy image access |
| `neuro_eval/benchmarks.py` | Dataset pairs, full cohorts and fixed scientific rules |
| `neuro_eval/model.py` | Model validation, isolated execution and code identity |
| `dashboard/` | Search monitoring backend and bundled UI |
| `tests/` | Scientific and operational contract checks |
| `search_runs/` | Generated search databases, candidate snapshots, notes and logs |
| `final_results/` | Frozen selected models, selection records and final scores |

RSA and encoding kernels are implemented directly in `neuro_eval/rsa.py` and
`neuro_eval/encoding.py`; source-file loading is implemented in `neuro_eval/readers.py`.
These implementations originate from `looped-networks` revision
`3ee575baeceb1ad784102162a051b7444182fdaa` (`looped/metrics/{rsa,encoding}.py`
and `looped/data/{nsd,laion,triplen,tvsd,stimuli}.py`), imported October 21, 2026.
Active numerical algorithms, source filters and image decoding are preserved;
unused source helpers are omitted. Local adaptations resolve repository `.env`
and shared-storage paths, cache prepared files by identity, and identify NSD
reliability volumes from response metadata. There is no dependency on the original
checkout or a second copy of the implementations. `SOURCE_REVISION` in
`neuro_eval/benchmarks.py` and the evaluation-code hash identify their provenance
in every run and final artifact.
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

RSA passes the complete stimulus set to the model in one call. Within each
candidate evaluation, subjects with identical ordered image contents reuse the
validated features and model RDM ranks, including source-group ranks. Fixed neural
ranks are cached across candidates with response identity, mutation and subset
checks. Every subject and prescribed region is still scored separately. Models
must bound intermediate memory (internal chunking is allowed); encoding
retains its batched forwards.

- For RSA, LAION and NSD use the search half of the shared images for their search
  subjects. Subject-private training images are excluded.
- For RSA, Triple-N combines its source folds and uses all 1,000 images for M1–M5.
- The full selected stream mask determines each subject's reward; average these
  pooled-stream scores equally across all required subjects. Never average the
  subregion scores to obtain the reward. Missing pooled-stream coverage or a
  missing subject fails evaluation. Search diagnostics include only subregions
  within the selected stream; for `visual`, the early and ventral streams are
  diagnostics as well.
- Fixed source filters are ncsnr >0.2 for LAION/NSD, reliability >0.4 for Triple-N,
  and oracle >0.4 for TVSD.

Default `status`/`wait-for-score` feedback is a short labeled display: pooled score
and parent delta first, selected-stream subregion scores/deltas, then identity and
execution information. `--json` returns compact structured feedback with full
numeric precision. Failures show a specific reason and suggested action first;
bounded candidate traces are available through `--details`. Detailed subject
coverage and LAION RSA source-group diagnostics (`laion`, `thingsplus`, `ood`, `nsd`) are
available through `status <id> --details`. A subregion or source-group cohort score
requires all subjects; unavailable diagnostics do not change the pooled reward.
Repeatedly optimized search scores are not independent held-out confirmation.

**Encoding:** for LAION and NSD, fit each subject's readout on that subject's private
images and test on one half of the shared images: the search half during search,
the held-out half for the final headline score. Triple-N uses source fold 0. For
final evaluation, freeze the selected model before loading any held-out responses.
LAION and NSD final evaluation reports the held-out subjects on held-out images
as the headline and the three other subject × image cells as diagnostics. Center features and normalize
their overall scale using training data; z-score targets with training statistics.
Choose a ridge penalty per neural target using five-fold training CV over the
source's 20-alpha grid. Report test Pearson r for the pooled selected stream,
averaged equally over the full cohort. Search subregion values reuse the same
per-unit predictions; they do not add readout fits. Final evaluation retains the selected
region and always uses encoding, regardless of the search metric.
Search encoding extracts features in bounded 1,024-image batches and validates
feature shape, finiteness and parameter limits after each batch, before continuing.
Isolated models receive CPU-staged inputs; only the image-only worker uploads them
to the GPU. One worker loads the model once and handles all batches and subjects
for that candidate. Encoding search does not run singleton or shuffled-context
verification probes. The worker is closed when that candidate finishes or fails.
Within one candidate's search evaluation, identical ordered test images reuse a
validated raw feature matrix (bounded to 128 MiB). Private training features and
subject-specific readouts are not shared. The cache is discarded on completion or
failure; final evaluation retains its existing batching and no feature cache.
These execution changes leave image policies, cohorts, ridge CV and rewards intact.
Evaluation-code identity still changes, so use a new run after updating the evaluator.

Both RSA and encoding are raw scores, without noise-ceiling normalization.
No comprehensive pixel-level duplicate audit across releases is claimed. The
held-out subject and image policy uses protocol v5; start a new run rather than
resuming a run scored under a previous policy.

### LAION region definitions

`early` uses **laion-EVC**: retinotopic V1–V3 intersected with the visually responsive
laion-general mask. `ventral` uses **laion-ventral**: the broad anatomical ventral
occipitotemporal sector below the inferior occipital–inferior temporal gyrus border.
It is not restricted to FFA/PPA or a pure high-level functional area. See the
[official ROI definitions](https://laion-fmri.hebartlab.com/rois.html).

The prepared response coordinates were checked against the source binary masks
and segmentation labels (EVC=1, ventral=4) for all five subjects. After the fixed
ncsnr >0.2 filter, retained voxel counts are:

| Subject | early | ventral |
| --- | ---: | ---: |
| sub-01 | 2,139 | 2,722 |
| sub-03 | 2,916 | 2,518 |
| sub-05 | 3,077 | 2,374 |
| sub-06 | 2,943 | 2,399 |
| sub-07 | 2,500 | 2,528 |

All coordinates matched their masks, with no early/ventral overlap. These are the
source sector masks, not a reconstruction from separately voxelized named areas.

Early diagnostics report **V1, V2, V3**; ventral diagnostics report **hV4, FFA, PPA**;
`visual` reports both streams and all six areas. The union is the concatenation of
both stream blocks; loading fails if the streams share a unit.
Each diagnostic mask is the intersection of that named area's retained coordinates
with the selected stream's retained coordinates. These diagnostic areas can overlap
and do not necessarily exhaust the stream. The pooled response matrix is stored
once, with subregion masks selecting columns from it; adding diagnostics does not
duplicate voxels or rerun model forwards. Triple-N ventral diagnostics use the
available **IT** units, intersected by their source unit identifiers.

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

Search agents receive a private workspace and role-bound API. Each proposer can
have one queued or running candidate. Lists and `my-best` return compact summaries;
`status <id>` and `wait-for-score <id>` supply readable pooled/subregion feedback
(`--json` preserves structured output). `status <id> --details` supplies notes,
full coverage and bounded candidate errors; `source <id>` supplies code.
Agents maintain `scratch/notes.md` as a rolling summary targeting 2–4 KB: best
active candidate, current direction, recent observations, hypotheses and next
comparison. Older details go to `scratch/archive/<candidate-id>.md` and are read
selectively. New sessions seed a template only when notes are missing. Existing
notes are never automatically truncated or replaced; prompts require preserving
long legacy notes before summarizing them. This is a working-note budget, not a
submission gate. Agents cannot read neural
responses, the queue database, other workspaces or final results directly.
The coordinator may set directions and reset proposer lineages. Corrections create
new child candidates; archived source is never overwritten. Independent-loop
mode restricts feedback to each proposer's own island.

Provider processes have network access for model requests; prompts prohibit other
network use. Codex and Claude disable unrelated integrations/tools; the installed
agy CLI does not expose a built-in tool allowlist. Candidate execution
has separate filesystem, process and network isolation and receives only source,
image inputs, installed runtime and necessary GPU access. Missing isolation support
fails closed. Numeric outputs are loaded without pickle. The trusted host compares
sampled singleton/shuffled batches from fresh isolated processes, including centered
features, to check the per-image contract within numerical tolerance. Parameter
counts use the larger of the declaration and observed retained tensor/array elements,
including registered PyTorch parameters and buffers, lazy weights, globals and NumPy
arrays. Aliases to the same object count once. Counts are provisional declarations
while queued; successful evaluation replaces them with the observed count. A failed
evaluation with no count metadata stores `null` (unknown), rather than zero.
These checks supplement isolation
and source review; they cannot prove arbitrary code is
untrained or explainable.

Each search run records its benchmark, source/code identity, runtime versions and
data-file identities in `env.json`. Candidate SHA256 hashes identify the bytes
that were scored. Final evaluation verifies the winner and writes `selection.json`,
a frozen `model.py` and `evaluation.json` under `final_results/`. It never writes
scores to the search database, and existing final artifacts are not overwritten.
Run databases, data, outputs, credentials and caches must stay out of Git.
