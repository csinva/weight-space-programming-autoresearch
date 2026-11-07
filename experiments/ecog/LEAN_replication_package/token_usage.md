       1 # Compute and token usage: LEAN vs Whisper autoresearch (patient1)
       2
       3 The hand-built audio-feature search (LEAN v1 to v20, with the clean-slate side workers and the melspec
       4 precursor loops) ran as autonomous Claude Code agents in SLURM jobs from June 1 to 8, 2026. The
       5 November 5 to 6 follow-up added variance partitioning, the v21 bug-fix rebuild and a grid rescore of 510
       6 early trials. Together these used about **300 GPU-hours** (about 250 A100, 50 V100) over 355 jobs.
       7 Most of that time went to agent jobs that held a GPU while Claude read, reasoned and edited code. Each
       8 trial's evaluation needs only about 4.7 A100-minutes, so the ~930 June waveform trials account for
       9 about 70 GPU-hours of actual GPU work, and the GPUs were busy roughly 30% of the time they were allocated.
      10
      11 On the language-model side, 47 agent transcripts (11,617 model responses) processed about **2.8 billion
      12 tokens**. 99% of these were cache reads: each turn re-reads a long working context of about 240k tokens.
      13 The agents wrote about **7 million output tokens**, including thinking; the true figure may be up to
      14 ~9.6 million, since the melspec share is uncertain. About 57% of the output figure comes straight from
      15 Claude Code's per-run usage records. The rest is estimated, for runs killed at walltime, from the text and
      16 code those runs emitted. At API list prices, the June agents' usage comes to about **$1,540**, about 80%
      17 of it for cache reads. These figures are lower bounds. They leave out the June interactive sessions that
      18 launched the workers, whose transcripts are gone, and the separate openaudio trained-model follow-on
      19 (another 86 GPU-hours).
      20
      21 Details, scripts and per-phase tables: `COMPUTE_SUMMARY.md` in this folder.
      
LEAN vs Whisper autoresearch: compute and token usage (as of 2026-11-06)

Regenerate: `python3 compute_tally.py` (SLURM) and `token_tally.py` (Claude Code transcripts).

## SLURM compute (`compute_tally.out`)

GPU-hours = elapsed time x GPUs allocated. Every job had 1 GPU.

| phase | jobs | GPU-h | GPU type |
|---|---:|---:|---|
| melspec precursor (Jun 1-3) | 8 | 32.1 | V100 21.7, A100 10.4 |
| waveform main line, LEAN v1-v20 + exceed missions (Jun 2-7) | 34 | 134.3 | A100 116.7, V100 17.6 |
| clean-slate workers A-I (Jun 4-5) | 11 | 85.8 | A100 |
| June eval / diagnostics (Whisper comparison, verify, bootstrap) | 18 | 3.3 | A100 |
| Nov 5 variance partitioning v20 | 50 | 1.5 | V100, A100 |
| Nov 5 LEAN-v21 + variance partitioning v21 | 79 | 1.2 | V100, A100 |
| Nov 5-6 grid rescore of 510 early trials | 155 | 41.7 | A100 |
| **total** | **355** | **299.8** | ~250 A100, ~50 V100 |
| (separate: openaudio trained follow-on, Jun) | 335 | 85.6 | A100 |

- The June agent jobs hold their GPU for the agent's whole lifetime, including the time the GPU sits idle
  while Claude reads, reasons and edits. Eval-only cost is about 4.7 A100-min per trial (median 4.3; measured
  over the 510-trial rescore). For the ~930 logged June waveform trials (main line + clean-slate) that comes to
  ~70 GPU-h of real GPU work against ~220 GPU-h allocated to those agents, so GPUs were busy roughly 30% of the time.
- Not counted: interactive `bash` dev-partition jobs (24.9 GPU-h in Jun 1-8; can't be attributed to a project),
  login-node work, and the Whisper feature extraction (done earlier for other projects).
- Job lists: `sacct_june_byid.txt`, `sacct_nov_byid.txt`. Re-pull by job ID: a date-window `sacct -S/-E` query
  missed ~10 array tasks of the November rescore.

## Claude tokens (`token_tally_v2.out`)

Worker model: claude-opus-4-7. This session: claude-opus-5-5. Each response counted once.

**Correction (Nov 6):** the first tally (`token_tally.py`, 1.27 M output) undercounted output ~5.6x. In
the June stream-json worker logs, each message carries the usage recorded when the response starts
streaming, where `output_tokens` is a placeholder (e.g. 8 for a response with a thinking block and a tool
call). Input and cache counts there are final. `token_tally_v2.py` instead uses each `claude -p` run's
final `result` event. Those are cumulative over a resumed session, so a run's usage = its result minus
the previous result of the same session. Check: summed differences give 854.5 M cache reads vs 852.0 M in
the logged messages, with 0 resets. Runs killed at walltime (48 of 241) have no result. Their output is
estimated as visible chars emitted (text + tool inputs) x the finished runs' output-tokens-per-char for
that group (0.78-1.08; thinking is redacted in the logs, so the ratio absorbs it). A per-response estimate
agrees within 12% for every group except melspec, which has only one finished run to calibrate on.

| group | logs | responses | cache read | cache write | output: reported + estimated | API-price cost |
|---|---:|---:|---:|---:|---:|---:|
| melspec precursor (Jun 1-3) | 6 | 3,514 | 1,235.9 M | 3.28 M | 1.08 M (0.22 + 0.86)* | ~$666 |
| waveform main line + trim (Jun 2-4) | 13 | 1,462 | 188.0 M | 2.73 M | 0.73 M (0.47 + 0.26) | ~$129 |
| clean-slate workers A-I (Jun 4-5) | 9 | 2,988 | 538.5 M | 5.42 M | 2.71 M (1.69 + 1.02) | ~$371 |
| graft / interactions / banded (Jun 5) | 3 | 898 | 88.5 M | 2.12 M | 0.60 M (0.46 + 0.15) | ~$73 |
| exceed (grid) missions (Jun 6-8) | 14 | 2,257 | 497.1 M | 3.24 M | 1.14 M (0.37 + 0.77) | ~$297 |
| Nov 5-6 analysis session (2 transcripts) | 2 | 498 | 258.5 M | 3.40 M | 0.80 M (reported) | not priced |
| **total** | **47** | **11,617** | **2,806 M** | **20.2 M** | **7.07 M (4.01 + 3.06)** | **~$1,540 (June)** |

\* melspec per-response estimate would give 3.6 M instead; per-char is used because it tracks what was
actually written (melspec turns were short, ~270 visible chars each).

- About 2.83 B tokens processed in total; 99% are cache reads (~240 k context re-read per response).
- June waveform/LEAN workers alone: 7,605 responses, 1.31 B cache read, 13.5 M cache write, 5.19 M output
  (2.99 M reported + 2.20 M estimated), ~$870.
- Cost = Claude Code's `total_cost_usd` (API list-price equivalent) for finished runs, plus killed runs priced
  at the per-token rates implied by those reports ($6.25 / $0.50 / $25 per M for cache write / cache read /
  output). Cache reads are ~80% of the cost.