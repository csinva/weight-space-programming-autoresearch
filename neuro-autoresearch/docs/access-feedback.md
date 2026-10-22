# Access and feedback contract

Search agents optimize the fixed primary RSA score. They may explore any model
within the untrained, per-image, resource and parameter contracts. No biological
motif or mechanism narrative is required. Later scientific analysis evaluates
explanations separately; predictive selection alone does not identify a mechanism.

## Roles and trust boundaries

| Role | Reads | Writes / actions |
| --- | --- | --- |
| Proposer | Own workspace; shared bounded search history, scores, diagnostics and archived source | Own `model.py` and scratch; submit source through broker |
| Conductor | Bounded fleet history, archived source, scores, diagnostics and directions | Set directions and reset lineage through broker |
| Simplifier | Search feedback and a frozen reference/input specification | Candidate edits; bounded equivalence checks against trusted frozen inputs |
| Repair | Supplied failed candidate and bounded failure context | Local repaired candidate; trusted evaluator archives a new child |
| Candidate forward process | Candidate source, current image batch and installed runtime | Numeric feature array and capped JSON metadata |
| Trusted evaluator | Fixed search images, neural responses and protocol | Search feedback and immutable model records |
| Final evaluator | Frozen selected model and paired final dataset | Separate final artifacts; no search feedback |

The launcher gives each provider session a private Unix socket whose trusted server
binds its role and run. Request fields or changed environment variables cannot grant
another role. There is no SQL endpoint, raw database access, arbitrary host-path
read, caller-selected run, or direct score-record edit. `propose-model` transmits
source text; only trusted code creates snapshots and queue records. Optional `--parent`
records a prior candidate ID, and feedback reports its primary-RSA delta when both
scores are available. Failures expose bounded categorical reasons and, when available, errors from the
image-only candidate process. Private evaluator tracebacks are never returned. `history`,
`top-models`, `random-models`, `status`, `source` and `my-best` expose bounded fields,
not raw evaluator metadata or host paths. `search_rsa` and compatibility field
`test_corr` contain the same primary RSA value. Source and history are shared across
proposers by default; optional independent-loop mode restricts them to the caller's
island and disables cross-island sampling. `my-best` remains lineage-specific.

`direction`, `set-direction` and `reset-island` use private trusted state. Only the
conductor may set directions or reset. Reset retires an island's working lineage
without deleting archives or modifying an in-flight source snapshot; local files
remain and the proposer is instructed to start afresh. Repairs create child records
with a parent ID rather than rewriting a scored or failed source snapshot. Model
hashes identify the exact bytes that produced each result.

## Enforced isolation and limits

Provider CLIs run in a Bubblewrap filesystem/process sandbox with their own
workspace, installed runtime and copied provider authentication files. They have
network access to contact their model provider, but no mounted datasets, shared
run database, archive directory, other workspaces or final results. Provider CLI
permission-skipping flags do not disable this outer sandbox. Missing or unusable
namespace support is a launch failure, not a fallback to unrestricted execution.

Exact simplification uses a broker-bound reference and input specification captured
before the session; candidate requests cannot replace them or loosen tolerances.
Equivalence checks compare synthetic inputs and do not prove equality on every image.

Candidate execution has a separate filesystem, process and network namespace.
It receives source and image inputs, an output directory, installed runtime, and
explicit GPU device nodes and read-only device-discovery metadata when available. It does not receive neural targets,
provider credentials, the broker, run state or host data directories. Trusted
scoring loads finite numeric arrays without pickle, then computes the fixed metric.
The isolated worker checks sampled singleton versus batch outputs with a 1%
relative vector-L2 tolerance, allowing CUDA rounding without changing scored features. Source
checks and materialized-weight counting supplement isolation; they do not prove
that arbitrary code has no learned constants or is independent on every image.
This boundary relies on the OS/kernel, trusted runtime and evaluator; it is not a
claim of protection from kernel or GPU-driver vulnerabilities. Provider network
access and provider login credentials remain necessary capabilities.

## Feedback and final separation

The fixed primary reward averages actual prescribed regions within each subject,
then averages all required subjects equally. Richer feedback does not replace this
objective. `rsa_diagnostics` version 1 retains primary RSA, per-subject and region
scores, target counts and coverage. LAION additionally reports RSA computed within
`laion`, `thingsplus` and `ood` source groups, after the mandatory NSD-source exclusion.
A subgroup missing any subject is unavailable, never silently averaged over fewer
subjects. Triple-N diagnostics expose its differing region coverage. Full cohorts,
source filters and primary aggregation remain fixed.

NSD and TVSD are sealed final encoding targets. Their data and scores are excluded
from proposer, conductor, repair and simplification feedback. Freeze and record the
RSA-selected model before final scoring. Final artifacts remain separate from the
search database and cannot select a new champion or trigger a search revision.
See [scientific protocol](protocol.md) for exact metric, cohort and source policies.
