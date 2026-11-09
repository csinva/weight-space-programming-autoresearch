# Search operations

[README](../README.md) · [Operations](operations.md) · [Scientific protocol](protocol.md) · [Comparisons](comparisons.md)

Run all commands from the repository root. Dataset, region and metric stay fixed for each run.

## Setup requirements

Installing the Python package does not provision the complete research environment.

| Requirement | Needed for |
| --- | --- |
| Python 3.12+, the parent repository's `uv sync`, and `PYTHONPATH=src:.` | Python commands and evaluation libraries |
| Linux, Bubblewrap (`bwrap` on PATH), usable mount/PID/user/network namespaces | Search and isolated final evaluation; unsupported isolation fails preflight |
| Prepared response files and matching stimuli | Neural scoring; files are not bundled or downloaded |
| Installed, authenticated agent CLI and model access | Autonomous search; compression specifically uses Claude Code |
| Optional baseline dependencies and pretrained weights | Selected reference comparisons |

Install Bubblewrap through your Linux distribution. Check isolation independently
with `python -c 'from neuro_eval.model import check_isolation; check_isolation()'`.
The repository does not provide a raw-data preparation pipeline. Obtain compatible
prepared files before attempting a search; their directory layout is described
[below](#data-isolation-and-outputs). Copy [`.env.example`](../.env.example) to `.env`
and replace the example paths. Each dataset must retain the subjects, filters and
image policies in the [scientific protocol](protocol.md).

The default roster uses three separate CLIs/accounts. For a smaller setup, install
and authenticate Claude Code, follow the token instructions below, and export
`NEURO_MODEL_ROSTER="claude:opus5.5:medium"`. Pass `--epoch-mins 0` to disable the
coordinator. With automatic repairs disabled, that configuration requires only
Claude. Provider and model overrides belong in the shell environment; `.env` is
used here for dataset paths. Use a model available to your account.

## Setup and launch

Use Python 3.12 or newer, from this folder (`experiments/vision`):

```bash
uv sync --extra dev --extra dashboard
source "$(git rev-parse --show-toplevel)/.venv/bin/activate"
export PYTHONPATH="$PWD/src:$PWD"

# Print settings, then check data, providers and isolation without model requests.
python -m search.run --show-config
python -m search.run --preflight

# Phase 1: autonomous search, 12 hours by default (sends real model requests).
# Default: LAION, pooled early + ventral streams (--region visual), encoding.
python -m search.run --gpu 0
python -m search.run --dataset nsd --gpu 0

# Phase 2: freeze the search winner and evaluate it on held-out subjects and images.
# Use the run directory printed by the search launcher.
python -m final_eval --run artifacts/search/laion__visual__encoding__2026-11-05
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
[`src/search/config.env`](../src/search/config.env). Codex and agy reuse existing machine
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
2026 in [`src/search/pricing.json`](../src/search/pricing.json), including cache usage and
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
python -m final_eval --model model.py --search-dataset laion --region visual

# Optional read-only dashboard over artifacts/search/.
uv sync --extra dashboard
python -m dashboard

# Tests use fake providers and synthetic data; no live model requests.
python -m pytest -q
```

## Data, isolation and outputs

Prepared data is read in place. On a new machine, you must obtain compatible
response files and matching stimuli first. No acquisition or raw-data preparation
command is included. The defaults below are lab-specific examples; override paths
in the shell or an untracked `.env` at the repository root.
Editable installation makes the packages under `src/` available to the module
commands above.

| Variable | Default response file |
| --- | --- |
| `LAION_DATA_DIR` | `/data/shared/datasets/laion_fmri/processed/laion_data.pkl` |
| `TRIPLEN_DATA_DIR` | `/data/shared/datasets/triplen/processed/triplen_data.pkl` |
| `NSD_DATA_DIR` | `/data/shared/datasets/nsd_processed/nsd_data.pkl` |
| `TVSD_DATA_DIR` | `/data/shared/datasets/papale2025.tvsd/processed/tvsd_data.pkl` |

Overrides name directories; filenames stay fixed except for `NSD_DATA_FILE`.
Stimulus paths follow the source readers: `LAION_FMRI_DIR`, `NSD_ROOT`,
`NSD_STIMULI_HDF5_PATH` and `TVSD_DATA_DIR`. NSD also requires the source ncsnr
volumes for reliability filtering. Triple-N also uses the NSD stimulus HDF5 file;
set `NSD_STIMULI_HDF5_PATH` even when not evaluating NSD responses. LAION expects
`stimuli/images/task-images_stimuli.h5` beneath `LAION_FMRI_DIR`; TVSD expects
`tvsd_stimuli.h5` alongside `tvsd_data.pkl` in `TVSD_DATA_DIR`. Preflight checks
required files for readability, but does not replace a complete data preparation
and schema-validation workflow.

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
a frozen `model.py` and `evaluation.json` under `artifacts/final/`. It never writes
scores to the search database, and existing final artifacts are not overwritten.
Search state lives under `artifacts/search/`; top-level operational logs live
under `artifacts/logs/`. Legacy `runs/`, `results/` and `.local/` contents are archived in
`artifacts/archive/runs/`, `artifacts/archive/results/` and
`artifacts/archive/local/`. Research code lives in `src/`, monitoring in root
`dashboard/`, and generated outputs in `artifacts/`.
Run databases, data, outputs, credentials and caches must stay out of Git.
