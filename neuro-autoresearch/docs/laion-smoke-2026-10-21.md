# LAION live search smoke test — 2026-10-21

The bounded live search produced 14 successful Astra candidates on the complete fixed LAION cohort. Best search RSA was **0.207912**. Argon could not execute commands under this account's headless permissions, so this was not a successful two-provider integration test. No final NSD/TVSD evaluation was requested or used.

## Execution and artifacts

- Workspace: `/home/ymehta3/research/neuro-autoresearch-lean-tests`.
- Artifacts: `runs/laion__2026-10-21/` contains `search.db`, immutable model snapshots, proposer logs, `env.json`, `launcher.log`, `smoke-observations.jsonl`, and `smoke-summary.json`.
- Used the original checkout's runtime via `NEURO_PYTHON=/home/ymehta3/research/neuro-autoresearch/.venv/bin/python`; no credentials or machine-local settings were changed.
- One GPU (0), one evaluator, two proposer slots, conductor disabled (`--epoch-mins 0`). Fixed all five subjects, 1,252 shared test images excluding NSD-source images, 26,116 total neural targets; source filters and RSA protocol unchanged. Each successful row retains both region scores for every subject and protocol provenance.
- Intended budget: five minutes. Initial launch requested 300 seconds; it was interrupted to correct an unsupported provider setting. Corrected launch requested only the remaining 240 seconds. Provider sessions began at 22:43:38 UTC and resumed at 22:44:27 UTC; corrected launcher reached its deadline around 22:48:26 UTC. Thus provider-start-to-final-deadline elapsed was about 4m48s, including correction; process cleanup completed around 22:48:47 UTC (about 5m09s). Separate preflight/setup preceded these timings.
- Initial roster: `codex:astra agy:argon4`, base effort `medium`. Corrected roster: `codex:astra:medium agy:argon4:high`, base effort still `medium`. Resolved models were `gpt-6-astra` and `gemini-4-argon-high`. The appended provider logs preserve both configurations; `env.json` reflects the corrected roster.

## Candidates and results

All candidates came from Astra (`p00`), all had status `ok`. The first 13 materialized 2,592 fixed filter weights; the final champion used 7,200. These are adaptively optimized search scores, not independent transfer results. No baseline evaluation was run, so improvement is relative to the first proposed candidate.

| ID | Idea | RSA | Evaluation seconds |
| --- | --- | ---: | ---: |
| iscru | Multiscale oriented energy with coarse color layout | 0.109153 | 1.57 |
| m8tv8 | Centered multiscale oriented energy with coarse color layout | 0.156968 | 0.30 |
| 2ahk1 | Centered oriented energy without color | 0.159841 | 0.27 |
| mhfyk | Oriented energy preserving local contrast strength | 0.152885 | 0.27 |
| c89bf | Oriented energy with stronger local contrast normalization | 0.151661 | 0.29 |
| h129d | Oriented energy and texture boundaries | 0.165519 | 0.38 |
| d5r65 | Texture boundaries emphasized over oriented energy | 0.182066 | 0.39 |
| mhzqw | Texture boundary geometry without first-order energy | 0.186593 | 0.40 |
| fe8l8 | Uncompressed texture-boundary energy | 0.154553 | 0.40 |
| 9q35g | Strongly compressed texture-boundary energy | 0.199407 | 0.47 |
| hibog | Logarithmic texture-boundary energy | 0.194614 | 0.39 |
| kjtbr | Texture boundaries with orientation-invariant layout | 0.192336 | 0.48 |
| ebf0a | Fine spatial layout of compressed texture boundaries | 0.201203 | 0.31 |
| u8lod | Longer oriented filters for texture boundaries | 0.207912 | 0.32 |

Per-subject search RSA (equal region means):

| ID | sub-01 | sub-03 | sub-05 | sub-06 | sub-07 |
| --- | ---: | ---: | ---: | ---: | ---: |
| iscru | 0.120357 | 0.115754 | 0.115594 | 0.093089 | 0.100971 |
| m8tv8 | 0.174912 | 0.166831 | 0.170302 | 0.133437 | 0.139356 |
| 2ahk1 | 0.179162 | 0.169475 | 0.174532 | 0.135926 | 0.140111 |
| mhfyk | 0.167939 | 0.163697 | 0.168360 | 0.131762 | 0.132668 |
| c89bf | 0.171451 | 0.157290 | 0.163781 | 0.130035 | 0.135747 |
| h129d | 0.185336 | 0.175336 | 0.181053 | 0.140887 | 0.144983 |
| d5r65 | 0.203104 | 0.192333 | 0.200408 | 0.155285 | 0.159202 |
| mhzqw | 0.207394 | 0.196571 | 0.206632 | 0.159318 | 0.163049 |
| fe8l8 | 0.171259 | 0.164235 | 0.167700 | 0.131367 | 0.138204 |
| 9q35g | 0.220005 | 0.205517 | 0.221665 | 0.173555 | 0.176295 |
| hibog | 0.218966 | 0.197602 | 0.214135 | 0.167289 | 0.175078 |
| kjtbr | 0.214610 | 0.197947 | 0.210112 | 0.164861 | 0.174149 |
| ebf0a | 0.222028 | 0.207677 | 0.224225 | 0.175203 | 0.176883 |
| u8lod | 0.230597 | 0.215206 | 0.229480 | 0.179456 | 0.184820 |

Best model: `u8lod` — Longer oriented filters for texture boundaries. Its source and SHA256 are retained in the scored row. Across the lineage, centering feature blocks, removing the color contribution, emphasizing texture boundaries, and stronger amplitude compression improved this search reward; several contrast/compression alternatives regressed. These observations support further controlled search ablations, not claims of final-dataset generalization.

## Timing and provider failures

- Corrected Astra start to first submission: about 40 seconds (22:44:27 to 22:45:07 UTC). The GPU was idle before that submission. Subsequent candidate creation intervals include proposer inference, editing, queue polling, and feedback wait.
- Recorded evaluation time totals **6.24s**: first evaluation **1.57s**, subsequent median **0.38s**, range **0.27–0.48s**. `evaluate_subjects` times whole-cohort data loading and scoring; it excludes earlier Python/module import and launcher/provider work. Existing persistent stimulus caches were reused, so this is not a clean-cache benchmark. Neural ranks and splits are reused within the evaluator. Sparse GPU observations showed 0% utilization and about 1,196 MiB allocated after initialization; they cannot measure peak utilization.
- Initial Argon error, despite successful preflight: `invalid model selection (--model "gemini-4-argon-high" --effort "medium"): --model gemini-4-argon-high conflicts with --effort=medium`. Five structured `ERROR` results occurred. `agy models` lists separate high/medium/low Argon IDs; generic CLI help lists effort choices but does not explicitly explain the coupling.
- With corrected high effort, Argon warned `--mode auto is not available for this account; continuing in the default mode`. Its initialization reported `permission_mode=request-review`; the first command was auto-denied because headless mode cannot prompt. All **15** completed corrected attempts returned structured **`status=SUCCESS` with `denied_actions` naming `command`**, but submitted no candidate. These attempts reported **197,108 total native tokens** cumulatively; no monetary cost is inferred. No permission bypass or settings change was attempted.
- Codex emitted unrelated tldraw MCP authentication warnings but continued producing and revising models successfully.
- Both launches exited with status 1 during shutdown. The final log reaches budget exhaustion and stopping loops but lacks the final summary; `usage.json` was not written. A subsequent reproduction using harmless process groups confirmed that `set -e` aborts cleanup when `kill` targets an already-exited group, before the helper reaches its trailing `return 0`. The zero-PID sentinel also incorrectly appeared alive, exhausting the grace period with the conductor disabled. Both helper defects were fixed after this run. One regression fails against each original defect independently and passes with the fixes; the live search was not repeated.
- After termination, all four known proposer/evaluator process groups from the two launches were absent. A `/proc` check found no remaining provider process or process with this run as its working directory (excluding the inspection command itself). The DB has 14 `ok` rows and zero queued/running/error rows. No additional scoring followed shutdown.

## Prioritized pipeline improvements

1. **Validate real provider compatibility before launch.** Reject Argon model/effort mismatches locally; distinguish CLI availability from account permission readiness. Report the effective per-slot effort in launcher logs (the current worker label still shows the base effort).
2. **Stop retrying nonrecoverable provider failures.** Parse native results, including `denied_actions` even when status says `SUCCESS`; stop a slot on permission/configuration errors. This prevents repeated token consumption with zero proposals. Configure narrowly scoped command permissions through the supported account workflow before another two-provider run.
3. **Shutdown cleanup fixed after this run.** The signal helpers now tolerate already-exited processes under `set -e`, and `alive` rejects the zero-PID sentinel. A bounded process-group regression covers both defects; confirm final summary/usage output on the next live run.
4. **Measure end-to-end latency.** Record submission, claim, score-completion, and feedback-delivery timestamps plus setup/cache phases. The evaluator polls at five seconds, which can exceed the roughly 0.4-second warm scoring time; reduce idle polling latency or use notification-based wakeups. Current evidence favors fixing provider/polling latency over adding GPU workers.
5. **Keep successful evaluation diagnostics.** Successful evaluator logs are currently deleted and stdout is buffered; retain compact per-candidate phase timings and provider outcome summaries for reliable operational analysis. The existing scored metadata already preserves the scientific cohort and regions.

The only production change resulting from this smoke test is the small shutdown-helper fix in `search/lib.sh`; the provider/latency improvements above remain recommendations. Raw run data and logs remain local and must not be committed.

## Follow-up: fixed three-provider profile

A second five-minute LAION smoke in `/home/ymehta3/research/neuro-autoresearch-agy-fix` validated the fixed profile: GPT-6 Astra, Claude Opus 5.5 and Gemini 4 Argon proposers at medium effort, with GPT-6 Astra/xhigh conducting. All CLIs skipped permissions as requested; agy inherited the existing ADC/GCP login and Claude used the existing Claude.ai login. Sessions started at **2026-10-22 02:49:53 UTC** (October 21 locally). Scientific cohort, source filters and RSA protocol stayed unchanged; no final-dataset scores were used.

| Slot / provider | Successful candidates | Best RSA | First submission after session start |
| --- | ---: | ---: | ---: |
| p00 / GPT-6 Astra | 14 | 0.147735 | 45 seconds |
| p01 / Claude Opus 5.5 | 40 | 0.190032 | 54 seconds |
| p02 / Gemini 4 Argon | 1 | 0.031115 | 217 seconds |

All three submitted candidates and received RSA feedback. The champion was Claude's `lolsb`, a log-Gabor ablation variant. Each provider's best row retained all five subject scores. These short, adaptive search results establish integration, not a reliable provider-quality ranking; the previous smoke's higher champion also came from a different search trajectory.

The conductor completed its first turn and wrote all three directions: oriented energy for p00, contour grouping/relational shape for p01, and non-Gabor random hierarchies for p02. It issued no resets. Directions were written after search had begun, so this does not establish that every submitted candidate followed them. Native conductor usage reported 203,823 input tokens, including 167,808 cached, and 2,865 output tokens. The fixed profile has **three top-level proposer slots**; native helper agents can still be spawned (observed for Codex), so it does not impose a hard runtime agent-count cap.

No permission denials or model/effort failures were observed. Unused tldraw MCP authentication warnings persisted without blocking search. Argon spent several minutes inspecting harness/model code and repeatedly querying prior candidates before its first submission. The launcher reached its budget, stopped the loops, printed its final DB summary and `done`, produced `usage.json`, and exited **0**. All six tracked loop/worker process groups were empty afterward; agy settings and ADC credential hashes were unchanged. Final DB state was **55 ok, zero queued/running/error**. Raw artifacts remain local under `runs/laion__2026-10-21/`; the launcher log is `.local/provider-verification/laion-search.log`.

Follow-up priorities:

1. Require a prompt first valid submission before extended code exploration, especially for Argon; provide concise model/hub contracts to reduce repeated source inspection.
2. Disable irrelevant MCP/plugin startup for search invocations while preserving the existing provider logins. Distinguish harmless connector warnings from model/tool failures.
3. Measure submission-to-score and feedback latency explicitly, then reduce evaluator polling overhead if it remains material. Candidate counts alone mix model latency, batching and evaluation throughput.
4. Write initial island directions before proposers start, then verify later adoption; clarify whether native helper agents are desired and account for their usage separately.

The earlier permission and shutdown failures did not recur under this explicitly authorized profile. Provider adapters now reject incompatible Argon settings and stop retrying nonrecoverable outcomes; no account credentials or machine settings were changed.
