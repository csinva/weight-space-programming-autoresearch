# Overnight Unattended-Run Readiness Assessment

> **Historical snapshot (pre-rework).** Several "open" items below have since
> landed: the append-only single-row ledger (`harness/results.py`), the
> `status="error"` safety net (`evaluate_on_split_safe`), the `idea`/`summary`
> columns, the trained-AlexNet baseline + beat-AlexNet goal, the
> `copilot-keepgoing` launcher, and V2/V3 in the ROI menu. The main remaining
> item is a more stable reward (multi-seed / k-fold or a tie band). This document
> also predates the harness rename (`config.py`→`datasets.py`, `eval.py`→`reward.py`,
> `evolve/`→repo-root `model.py`) — treat its file paths as historical.

Scope: can the NeuralSet-based autoresearch loop run **overnight, unattended**,
driven by `copilot-keepgoing` wrapping the Copilot CLI, iterating
propose-model → evaluate → record across the active dataset?

> **Verdict: NOT READY (conditionally) for a hands-off overnight run.**
> The loop *mechanics* are sound and verified to run end-to-end offline in
> ~13s/iteration with no prompts or downloads. But several issues make a
> long unattended run lose history, chase noise, or silently stall/crash. The
> blockers below are small, well-scoped fixes — once addressed, this is READY.

This assessment is based on reading `README.md`, `harness/{config,data,eval}.py`,
`evolve/{model.py,program.md}`, `scripts/run_nsd_trial.py`, `pyproject.toml`, and
on actually running the harness end-to-end (numbers reproduced below).

---

## What works today (the good news)

- **Runs offline, no manual gates.** Deps (`neuralset[all]`, `torch`, `sklearn`,
  `pandas`, `matplotlib`) are installed; `copilot-keepgoing` is installed and on
  `PATH`. The active dataset (NSD `Allen2022MassiveSample` sample) is
  pre-downloaded locally (622 MB) with `download=False`, so a run needs **no aws
  CLI, no NSD T&C acceptance, no network, no credentials, no interactive input.**
- **End-to-end verified.** A fresh `scripts/run_nsd_trial.py --iterations 3` ran
  clean: `X_train(200,768)`, `Y_train(200,31448)`, three models scored, exit 0.
- **The reward is genuinely held-out.** Only the ridge readout is fit; the model
  is never trained. The split is a fixed seeded 200/50 hold-out, and `test_corr`
  is mean per-voxel Pearson r on the **test** set — a real out-of-sample reward,
  not a training metric.
- **The reward correctly rejects regressions.** `spatial_pyramid` (0.0202) scored
  below `identity` (0.0417); a continued loop would correctly revert it. The
  reward direction is trustworthy.
- **`.copilot-worktrees/` is gitignored**, so orchestration worktrees won't
  pollute the main checkout.

### Measured runtime (reproduced this run)

| Stage | Time | Notes |
| --- | --- | --- |
| `load_split` (data + segmentation + fMRI extraction) | **~9.3 s** | dominates; re-run **every iteration** |
| model forward | 0.2–71 ms | trivial |
| ridge fit | 94–262 ms | grows with activation dim (768 → ~262 ms) |
| ridge predict | 18–43 ms | trivial |
| **wall per iteration** | **~13 s** | almost entirely the one-off data load |

Per-iteration compute is cheap. Over an 8-hour overnight run, **wall-clock will be
dominated by the agent's own think/edit latency, not by this 13 s harness cost** —
so raw throughput is not the limiting factor. The data load being repeated every
iteration (below) is wasteful but not a blocker.

---

## Blockers — fix before an unattended overnight run

Ordered by impact on an unattended multi-iteration run.

### P0-1 — Result history is **silently overwritten**, so overnight progress is lost
`save_results()` upserts keyed by `(dataset, model_name)`. The agent iterates by
**editing the internals of a model while keeping its `name`** (e.g. repeatedly
improving `"spatial_pyramid"`). Every such iteration **overwrites** the previous
row — `overall_results.csv` keeps only the *latest* score per name, not the
trajectory. After a full night you get one row per model name, no history, no way
to see the reward curve, and the bar plot shows only the last variant.
**Fix:** append every evaluation as a new row with a monotonic `iter`/`timestamp`
(and ideally a short params/description hash), instead of upsert-by-name. Keep the
"best per name" view as a derived query, not as the stored truth.

### P0-2 — Hardcoded absolute data path makes the run **machine-specific and stall-prone**
`config.py` pins `path="/home/t-ymehta/Research/neuralset/data/nsd_sample"`. If
that path is ever absent (different machine, moved cache, the orchestrator running
in a worktree on another host), `load_split` hits `FileNotFoundError` and its
`except` branch calls `study.download()` — which for full NSD needs the **aws CLI
and NSD T&C acceptance**, i.e. it would **hang/fail mid-night with no human to
unblock it**. **Fix:** resolve the path from an env var / repo-relative location
with an explicit existence check that **fails fast with a clear message** rather
than silently falling through to a gated network download.

### P0-3 — The reward is **too noisy to safely optimize** on 50 held-out samples
The reward signal is real but small and noisy: test_corr ≈ 0.02–0.04 with only
**50 test stimuli**, a **single seed-0 split**, averaged over **31,448 voxels**
most of which are not visually driven (diluting the mean toward 0). The
differences the agent is asked to optimize (0.0417 vs 0.0382 vs 0.0202) are
plausibly **within the sampling noise of a 50-sample correlation**. Overnight, the
agent can spend hours **chasing noise** — "improving" the reward by luck and
locking in changes that don't generalize. (Note: the classic p>n overfitting shows
up in `train_corr` = 0.92 for `identity`; that does *not* contaminate the held-out
reward, but the held-out reward's *variance* is the real problem.)
**Fix:** make the reward more stable before trusting it as an overnight objective —
e.g. k-fold / multi-seed averaging and report a std, restrict targets to a
visually-driven voxel subset (or a noise-ceiling-normalized score), and/or treat
changes within ±1 noise band as ties so the agent doesn't ratchet on noise.

### P1-4 — A bad model edit / shape error **crashes the run with no recorded failure** ✅ DONE
`evaluate()` originally had **no try/except around `model(...)`** and never wrote the
`status` field as anything but `"ok"`. A syntax error, exception, or wrong output
shape in the edited model made the run exit non-zero with a traceback and **no row**.
**Fixed:** `harness/reward.py` now exposes `evaluate_on_split_safe`, which catches any
exception from the model forward pass or the ridge readout (raise, wrong output shape,
NaNs, ...), prints the traceback for the agent to read, and returns a
`status="error: <ExcType>: <msg>"` row with NaN scores (region columns kept as NaN so
the schema stays stable even when a run's first row fails). `harness/run.py` uses it,
saves the row, prints the failure, and exits cleanly — so the loop keeps a durable
record of the failed proposal. The analyzer already excludes non-`ok` rows.

### P1-5 — Empty / fully-dropped segment is an **edge-case crash**
`load_split` uses `drop_incomplete=True`. If a (future) dataset/segment config
drops all events, `n=0`; `torch.randperm(0)` is empty and `perm[:1]` indexes an
empty tensor → crash before any model runs. Not reachable with the current NSD
sample, but it is a latent stall the moment a dataset switch goes slightly wrong.
**Fix:** assert `n > 0` (and `n_test < n`) with a clear error in `load_split`.

### P1-6 — Switching datasets is a **manual, possibly-gated** step
Only the local NSD sample is wired for offline use. The full NSD
(`download=True`) needs **aws CLI + T&C**; the "third dataset"/EEG paths
(`Grootswagers2022HumanSample`, `Fake2025Meg`) referenced in `config.py` are not
the active, pre-staged dataset, and there are **unmerged branches**
(`copilot/third-dataset`, `copilot/nsd-download`, `copilot/harness-multimodal`)
indicating that multi-dataset support is in-flight, not landed. The README frames
dataset switching as a one-line *manual* edit. So a "run across datasets overnight"
goal is **not automated** today and any switch to a gated study would stall.
**Fix:** pre-download/stage every dataset intended for the overnight run as a
local `download=False` path, and decide whether dataset rotation is the agent's
job (scripted) or stays a human pre-step. Do **not** point an unattended run at any
`download=True` study.

---

## Lower-priority / hygiene (won't stall the run, but worth fixing)

- **Data reloaded every iteration (~9.3 s each).** `evolve/model.py` reloads the
  split on each invocation. Harmless given agent latency dominates, but a cached
  split would cut most per-iteration compute.
- **No automated loop *driver*.** "The loop" is the Copilot agent reading
  `program.md`, editing `model.py`, and running it — there is no deterministic
  loop script. Overnight autonomy therefore rests entirely on agent reliability +
  `copilot-keepgoing` nudging; there is no harness-side iteration counter,
  stopping criterion, or budget. Consider a thin driver that records iteration
  count and a stop condition (e.g. N iterations or no-improvement plateau).
- **Binary plot churns git.** `runs/test_corr.png` is regenerated every run; if
  the agent commits per iteration, expect a noisy binary diff in every commit.
  Consider regenerating the plot on demand rather than committing it each time, or
  gitignoring it.
- **Commit/branch hygiene is unspecified for the loop.** The harness never
  commits; persistence-to-git depends on whatever the agent does. For a clean
  overnight run, define the commit cadence and branch policy explicitly (one
  commit per accepted improvement is a reasonable default).

---

## Minimal go/no-go checklist before launching overnight

1. [ ] Make result rows **append-only with iteration/timestamp** (P0-1).
2. [ ] Resolve the data path robustly and **fail fast** instead of falling through
       to a gated download (P0-2).
3. [ ] Confirm the **reward is stable enough** to optimize (multi-seed/k-fold or a
       tie band), or accept that overnight gains may be noise (P0-3).
4. [x] Wrap evaluation so a **bad edit records `status="error"`** instead of
       crashing silently (P1-4). **DONE** — `evaluate_on_split_safe`.
5. [ ] Pre-stage **every dataset** for the night as local `download=False`; never
       point the run at a gated `download=True` study (P1-6).
6. [ ] Decide and document **commit cadence / plot handling** so the night's git
       history stays clean (hygiene items).

With P0-1, P0-2, and P0-3 addressed (and P1-4 as a safety net), the loop is
**READY** for an unattended overnight run on the local NSD sample. Until then it
will *run*, but it is liable to **lose its history, optimize noise, or stall on a
moved path / gated download** without anyone watching.
