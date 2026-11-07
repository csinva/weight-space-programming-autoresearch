# LEAN autoresearch rerun on a new held-out test set

*patient1 high-gamma, waveform → ECoG hand-built encoder (LEAN). Rerun Nov 6–7, 2026.*

This package documents a rerun of the whole LEAN search history, and of the LEAN-v20 variance
partitioning, on a test set that the search never saw. It pairs with `LEAN_replication_package.zip`,
which documents the original June 2026 search and its Nov 5 analyses. Everything needed to repeat
the rerun is here except the patient data (audio, high-gamma) and the 4.8 GB of aligned audio
segments the extraction script rebuilds.

---

## 1. Summary

**Question.** Every one of the ~600 autoresearch trials was scored and selected on the same 23-chunk
test set (`450a36e4_0190`–`0212`, one podcast segment). After that many selections, are the scores
and the progression optimistic?

**Answer.** The ranking of trials holds; the absolute level, and LEAN's distance to Whisper, do not.

| | Old test (23 chunks) | New test (317 chunks) |
|---|---|---|
| LEAN-v16, grid r | 0.0852 | 0.0693 |
| LEAN-v17 | 0.0866 | 0.0707 |
| LEAN-v18 | 0.0890 | 0.0722 |
| LEAN-v19 | 0.0903 | 0.0734 |
| LEAN-v20 | 0.0904 | 0.0735 |
| Model the search finally kept (LEAN-v35-AUTOLAGZ) | 0.0909 | 0.0737 |
| Best trial on the new test (LEAN-v31-LOGEPROD8) | 0.0904 | 0.0739 |
| Whisper features, same ridge and delays | 0.0933 | 0.0819 |
| **LEAN-v20 − Whisper** | **−0.0029** | **−0.0084** |
| LEAN-v20 variance explained (r·\|r\|, grid) | 1.89% | 1.48% |

Grid = mean over the 125 surface-grid electrodes. All-166-electrode values are in §7.

- **The ranking holds.** Across 593 trials, old-test and new-test scores correlate at r = 0.981. The
  model the search kept by its old-test score is within 0.0002 of the best trial on the new test, so
  selecting on the old test did not pick the wrong models.
- **Most of the improvement is real.** v16 → v20 gained +0.0052 on the old test and +0.0042 on the new
  one (about 80% transfers). The early front-end work (cochleagram, compression and adaptation,
  multiple windows, modulation filters) transfers clearly. The small steps between about trial 150
  and 290 (trimming, literature features, grafts) mostly do not.
- **The gap to Whisper depends on the test set.** It widens from −0.003 to −0.008 on the new test, by
  about the same amount for v16 (−0.0081 → −0.0126) as for v20 (−0.0029 → −0.0084). So the old podcast
  block favored the hand-built features in general; the late tuning is not the cause.
- **Variance partitioning.** About 88% of the explained variance is still shared between feature
  families. One block looks fitted to the old test: "cochleagram core + slow context + lookahead"
  falls from 3.1% unique to 0.5% (CI includes 0), driven by the slow-context section (2.4% → 0.6%).
  Modulation filters rise from 4.0% to 6.8%.

Figures: `figures/fig1_progression_newtest.png`, `figures/fig2_breakdown_newtest.png` (captions in
`figures/CAPTIONS.md`). The original old-test figures are included for comparison.

---

## 2. What changed and what did not

**Held fixed (asserted in every run):**
- Training data: the harness's 500 training chunks, `protocol/train_frozen.json` (sha256 prefix
  `7d317eca03e19253`). The scoring code asserts the training list equals this file (or, for 5
  data-scaling trials, a prefix of it). The new test chunks come from session `450a36e4`, which every
  autoresearch run excluded from training (`ECOG_EXCLUDE_PREFIX=450a36e4`), so every fit is
  identical to the original.
- Model: each trial's own feature code, 4 FIR delays, lag 3 frames, trim 30 frames, per-chunk z-score,
  bootstrapped ridge with alphas logspace(1, 4, 12), 5 bootstraps × 20 blocks × 40 frames, seed 42.
- Metric: for each test chunk and electrode, Pearson r between prediction and high-gamma within that
  chunk; averaged over chunks, then over electrodes. This is **average per-chunk** performance, not a
  correlation over concatenated chunks. Variance shares use r_e·|r_e| per electrode (r_e = the
  chunk-averaged r), averaged over electrodes.

**Changed:** only the test chunks.

---

## 3. The new test set (`protocol/`)

Fixed before any trial was scored on it (`protocol/PROTOCOL.md`, `freeze_test_set.py`):

- **Candidates (493):** every `450a36e4` chunk in `master_list.json` with high-gamma and audio, other
  than the 23 old test chunks. Session 450a36e4 has 517 chunks; only the old 23 had ever been used.
- **Primary (317):** candidates in `master_list_speech150.json` (at least 150 clean words; a February
  2026 list, matching the speech-rich selection of the old set), excluding chunk indices 180–222 so
  that no continuation of the old podcast block (±10 chunks) is reused.
- **Secondary:** all 493 candidates. Per-chunk r is saved for every candidate. Primary and all-493
  scores correlate at r = 0.999 across trials.
- Hashes of all four lists: `protocol/frozen_hashes.txt`.

**Audio segments.** The harness reads pre-aligned waveform segments from one directory, and it builds
the training list by shuffling *every* chunk that has a segment there. Adding the new segments to that
directory would have reshuffled and silently changed the training set. So the segments were written to
a separate directory (`runs-wave/newtest/wave_aligned/`, 4.8 GB) by `extraction/extract_newtest_wave.py`,
using the harness's own `build_segment` (copied as `extraction/harness_extract_wave_aligned.py`). The
script first rebuilt two existing segments bit-identically.

---

## 4. Recovering the code for every trial (`code_recovery/`)

To rerun a trial, its exact `wave_encoder.py` is needed. 601 trials are logged on the Figure 1 main
line (same selection as `plot_progress_grid.py`: same runs and order, references and baselines
excluded, (name, score) duplicates removed).

**Sources (`make_trials_newtest.py` → `trials.json`):**
1. **218 trials:** the Nov 5 `rescore-grid` snapshot, used only when that snapshot had already
   reproduced its logged score.
2. **378 trials:** the exact file rebuilt from the agent logs (`replay_snapshots.py`). The June 6+ grid
   missions saved code only for versions they kept, and some earlier snapshots had been overwritten, so
   most trials had no usable saved code.
3. **5 trials** (graft-lean `GRAFT-v2b/v3/v4/v5/v5b`): no evaluation of that name appears in the
   run's log; not rerun.

**How the replay works.** The agents ran Claude Code in stream-json mode, and every successful `Edit`
result in those logs carries `originalFile`: the full file before the edit. Every complete `Read`
carries the full file. Replaying each run in order:

- `Edit` → anchor the state on `originalFile`, then apply the edit.
- `Write` → set the state to the written content.
- `Read` (complete) → anchor the state.
- `cp X wave_encoder.py` → state = replayed content of X (library snapshots are themselves tracked).
- Any other shell write to the file (`sed -i`, redirect, Python write) → state unknown until the next
  anchor.
- `python wave_encoder.py ...` → an evaluation. Its snapshot is the current state, or, if the state is
  unknown, the next anchor, provided nothing modified the file in between.

An evaluation is **verified** when a later anchor, with no modification in between, equals its
snapshot. Per run (`replay_<run>.json`):

| Run | Evals | Recovered | Verified | Contradicted |
|---|---|---|---|---|
| wave-slurm-23628821 | 164 | 164 | 157 | 0 |
| wave-trim | 28 | 28 | 23 | 0 |
| wave-lean | 29 | 29 | 29 | 0 |
| graft-lean | 11 | 11 | 11 | 0 |
| lean-interp | 86 | 86 | 86 | 0 |
| exceed-dense | 6 | 6 | 6 | 0 |
| exceed-frontend | 120 | 120 | 27 | 0 |
| exceed-scatter | 115 | 115 | 114 | 1 |
| exceed-combine | 26 | 26 | 26 | 0 |
| exceed-v18 | 65 | 65 | 64 | 0 |
| exceed-v20-context | 11 | 11 | 9 | 0 |
| exceed-v20-deep | 18 | 18 | 17 | 0 |
| exceed-v20-speaker | 17 | 17 | 14 | 2 |

Most unverified evals (mainly exceed-frontend) were followed by a revert-by-copy, which leaves no later
anchor to compare against; the reproduction gate below decides them.

**Matching an evaluation to a logged row.** Jun 2–5 runs write results with an upsert keyed by model
name (a rerun under the same name replaces the row), and the Jun 6+ runs append. In both cases, a row
is matched to the last evaluation of that name whose score, as read back by the agent from `run.log`,
equals the logged score (4 dp); failing that, the last non-crashed evaluation of that name. 331 of
the 378 replayed trials had a matching read-back score.

**Flags.** Command-line flags of each evaluation are kept (e.g. `--interact-set`, `--cond-mode` for
lean-interp). Five trials were evaluated with fewer training chunks (100, 50, 200, 300, 400; a
data-scaling probe) and are rerun with the same count. That count is a prefix of the frozen list.

---

## 5. Scoring harness (`rerun/`)

Each trial runs its own `__main__` in a sandbox (`work/<id>/`) whose `src/` package is
`rerun/src_patched/`. Original harness files are in `rerun/harness_src/`.

- **`rescore_stream.py`.** A streamed copy of the harness ridge (`encoding.bootstrap_ridge_gpu`):
  training features are accumulated chunk by chunk into XᵀX and XᵀY in float64, with the bootstrap
  tune rows kept, so the 1.75M × P matrix is never held in memory. It then scores the old 23-chunk test
  and, with the same weights, streams the 493 new chunks one at a time. It asserts that the training
  list equals (a prefix of) `train_frozen.json` and that the old test list equals `test_old.json`.
- **`eval.py`.** `run_encoding` → `rescore_stream.rescore` when `RESCORE_OUT` is set (the Jun 2–5
  trials go through this path).
- **`features.py`.** The Jun 6+ grid encoders do not call `run_encoding`: their own `__main__`
  computes features and fits on the 125 grid electrodes. Ridge is per-electrode, so that fit equals the
  all-166 fit restricted to electrodes 0–124. This file runs the original `features.py` unchanged and
  wraps `get_features_per_chunk`: when it sees the frozen training list, it hands the encoder's own
  embedder (and its ndelays, lag, trim) to `rescore_stream`. Those encoders also monkey-patch this
  function with a disk-memmap version to save RAM. While `RESCORE_OUT` is set, a module-level guard
  ignores that assignment; this affects memory only.
- **`__init__.py`.** Expandable-segments CUDA allocator (avoids fragmentation OOMs on the largest
  encoders; changes no values).
- **`rescore_task_newtest.py`.** The work-stealing driver: atomic `mkdir` claims, stale-owner
  takeover, and a stop when under 45 min of walltime remains. Trials logged before the Jun 5 venv
  reinstall that fail are retried once with a NumPy 1.26.4 overlay (NumPy 2's NEP 50 promotion
  broke some of them). No retry was needed; one trial (`GRAFT-v2c-reservoir`) ran with the overlay
  from the start, as in the Nov 5 rescore.
- **GPU type.** A100 only, the same as the Nov 5 rescore. A100s use TF32 convolutions, so feature
  values can differ slightly between GPU types.

**Whisper reference (`rerun/whisper_only_encoder.py`).** The same features as the June Whisper-only
fit: the 1,280-dim `{chunk}_whisper.npy` rows, 4 delays, the same ridge. The harness passes only audio
windows, so the chunk is identified by the MD5 of its first 4,096-sample window. The June map
(`evolve-wave/ling_features/fingerprint_map.json`) is extended with the new chunks. An unknown
fingerprint raises an error instead of returning zeros, which is what the June encoder did.

---

## 6. Validation gates

1. **Reproduction.** A trial is used only if its old-test score, recomputed in this run, matches its
   logged score within ±0.00015, using the metric it was logged with (all-166 for Jun 2–5, grid for
   Jun 6+). Result: **593 of 596 pass.** Across all 596, |rerun − logged| has median 0.000015,
   95th percentile 0.000049 and maximum 0.000266. The 3 misses:

   | Trial | Logged | Rerun |
   |---|---|---|
   | exceed-v18 `LEAN-v18.q49-SLOWQUADCOARSE` | 0.08993 | 0.09020 |
   | exceed-v20-context `LEAN-v21-c2` | 0.0905 | 0.09035 |
   | exceed-v20-context `LEAN-v21-c6` | 0.08965 | 0.08982 |

   Each is the only evaluation of its name, the replay verified its code, and the agent's read-back
   score matched the log. So these look like numerical tails, not wrong code. They are excluded from
   the figures under the protocol; all three score about 0.073 on the new test.
2. **Pilot.** Five trials covering every code path passed before the main array was released:

   | Trial | Path | Logged | Rerun old | New (grid) |
   |---|---|---|---|---|
   | LEAN-v16 | rescore-grid snapshot, all-166 | 0.0760 | 0.07604 | 0.0693 |
   | Gammatone-BEST+highlevel-mt | replayed, all-166 | 0.0612 | 0.06116 | 0.0574 |
   | LEAN-v18.q45 (= v20) | replayed, grid `__main__` | 0.0904 | 0.09045 | 0.0735 |
   | SCAT-v103 | replayed, scattering | 0.0882 | 0.08823 | 0.0714 |
   | COMBINE-v1, 200 training chunks | replayed, prefix training | 0.0772 | 0.07720 | 0.0664 |

3. **Whisper.** The rerun reproduces the June reference exactly: 0.08172 all-166, 0.09330 grid.
4. **Variance partitioning.** Each of the 135 refits reproduced its stored old-test per-chunk,
   per-electrode r to within 1e-13, so the weights are the Nov 5 weights.

---

## 7. Results: progression (`results/`, `figures/fig1_progression_newtest.*`)

In the figure, dots are the new-test scores. The black step line is the model the search was keeping
at each point (the best so far by old-test score), scored on the new test. It is not monotonic, because
a model that won on the old test can score lower on the new one. The gray step line is that same model
scored on the old test, and the dashed lines are Whisper on each test set.

**End of each search phase** (grid):

| Phase | Kept model, old | Kept model, new | Best new so far |
|---|---|---|---|
| hand-design exploration | 0.0862 | 0.0706 | 0.0709 |
| trim + literature → v16 | 0.0862 | 0.0706 | 0.0709 |
| graft + interactions → v17 | 0.0867 | 0.0713 | 0.0714 |
| front end + scattering → v18 | 0.0894 | 0.0722 | 0.0725 |
| quadrature banks + gates → v19, v20 | 0.0904 | 0.0736 | 0.0736 |
| context, coupling, speaker | 0.0909 | 0.0737 | 0.0739 |

**All 166 electrodes and the all-493 set:**

| Model | Old all-166 | New all-166 | New grid, all 493 chunks |
|---|---|---|---|
| LEAN-v16 | 0.0760 | 0.0610 | 0.0649 |
| LEAN-v20 | 0.0808 | 0.0646 | 0.0690 |
| Kept model (v35-AUTOLAGZ) | 0.0815 | 0.0649 | 0.0691 |
| Whisper | 0.0817 | 0.0703 | 0.0752 |

On all 166 electrodes LEAN-v20 and Whisper were nearly tied on the old test (−0.0009). On the new
test the gap is −0.0057.

---

## 8. Results: variance partitioning (`varpart/`, `figures/fig2_breakdown_newtest.*`)

The same 135 LEAN-v20 ridge fits (full model; each unit alone; full model without each unit), scored on
the 317-chunk test. 95% CIs come from a paired bootstrap over test chunks (B = 2000). Full-model
variance explained (grid): 1.476% new vs. 1.894% old. Values are % of the full model:

| Family (raw dims) | Alone, old | Alone, new | Unique, old | Unique, new [95% CI] |
|---|---|---|---|---|
| Modulation filters (1803) | 90.6 | 93.8 | 4.05 | **6.78** [6.05, 7.51] |
| Interactions, v3e (405) | 82.1 | 82.1 | 1.47 | 1.96 [1.72, 2.20] |
| Wavelet scattering (287) | 77.5 | 75.1 | 1.60 | 1.14 [0.88, 1.42] |
| Spectral/cochleagram contrast (192) | 71.8 | 70.2 | 0.97 | 0.71 [0.50, 0.92] |
| Cochleagram core + slow context + lookahead (960) | 83.9 | 76.9 | 3.07 | **0.50** [−0.13, 1.14] |
| Pitch, voicing, voice quality (122) | 60.7 | 53.6 | 0.35 | 0.42 [0.33, 0.49] |
| Spectral shape, flux, formants (83) | 69.4 | 68.6 | 0.37 | 0.41 [0.32, 0.51] |

The sum of unique shares is 11.9% at the family level on both test sets, 9.4% for sections (10.1%
old) and 10.5% for groups (12.0% old). About 88–91% is shared.

**Sections that changed most** (unique %, old → new): H slow context 2.41 → 0.59; M quadrature bank
0.93 → 1.70; F STMF −0.08 → 0.79; J interactions 1.47 → 1.96; A cochleagram core 0.30 → 0.06; E 1-D
modulation 1.56 → 1.22; P scattering 1.60 → 1.14. Full tables: `varpart/RESULTS.md` (new) and
`varpart/old_test_reference/RESULTS.md` (old).

---

## 9. Interpretation and caveats

- **Same session.** The new chunks come from the same held-out session as the old test, the only one
  the search never trained on. They share that session's recording conditions, so this tests over-fitting
  to the specific old chunks, not generalization to a new session. A session-level test would need a
  different training set, which would change every fit.
- **Different content.** The old test was one podcast block, while the new set spans the rest of the
  session with a ≥150-clean-word filter. Lower absolute scores are expected (an earlier check found
  random-chunk splits about 0.015 below the old set). That is why the comparisons above are between
  trials and against Whisper on the same chunks.
- **Selection was benign.** Old-test selection picked models within 0.0002 of the new-test optimum,
  and the black line rises with the gray one. What did not transfer is the LEAN-vs-Whisper margin and
  the slow-context block's unique contribution.
- **Excluded trials.** 3 trials missed the reproduction tolerance (numerical, §6), and 5 had no code. All
  conclusions are unchanged if those 3 are included: they score 0.073 on the new test.
- **Primary vs. all-493.** The ranking is the same either way (r = 0.999 across trials). On all 493
  chunks LEAN scores are 0.002–0.005 lower (median 0.0044) and Whisper 0.0067 lower, so the
  LEAN−Whisper gap is −0.0062 there (v20).

---

## 10. Reproducing

All scripts use absolute paths under `/path/to/ecog_data/`; on another
system, change `REPO`/`HERE` at the top of each. Requirements: the autoresearch repo (with
`evolve-wave/src`, the run directories and the agent logs in `autoresearch_loop/logs/`), its `.venv`
(`ENVIRONMENT.txt`), the data in `wav_files/` and the original aligned segments in
`/path/to/scratch/patient1_wave_aligned`.

1. `python protocol/freeze_test_set.py` — writes the four lists; check the hashes against
   `frozen_hashes.txt`.
2. `sbatch extraction/run_extract.sh` — 20 CPU tasks, builds `wave_aligned/` (a few minutes).
3. `python code_recovery/replay_snapshots.py <run> ...` (CPU, a few minutes) →
   `python code_recovery/make_trials_newtest.py` → `trials.json`. `trials_portable.json` points to the
   snapshots inside this package.
4. `sbatch rerun/run_pilot.sh` (5 A100 tasks), then
   `sbatch --dependency=afterok:<pilot> rerun/run_array.sh` (100 A100 tasks, work-stealing), and
   `sbatch rerun/run_whisper.sh`.
5. `sbatch varpart/run_phase2.sh` (15 A100 tasks; reads `runs-wave/varpart-v20/stats/`), then
   `run_phase3.sh`.
6. `python figures/plot_progress_newtest.py` and `python figures/plot_breakdown_newtest.py`.

Job IDs of this run: `rerun/jobs.txt`, `varpart/jobs.txt` (extract 29263192 + 29263490, pilot
29263798, main array 29263799, Whisper 29264153, variance partitioning 29263980 → 29263981).

**Compute.** 71.7 A100 GPU-hours (main array 68.8, variance partitioning 2.2, pilot 0.6, Whisper
0.1), plus under 1 CPU-hour. The main array peaked at 16 concurrent tasks, limited by the per-user
A100 caps of 8 GPUs per partition.

---

## 11. Files

| Path | Contents |
|---|---|
| `protocol/` | PROTOCOL.md, freeze script, frozen train/old-test/candidate/primary lists, hashes |
| `extraction/` | Segment builder for the new chunks + its SLURM script + the harness extractor it imports |
| `code_recovery/` | Replay script and per-run replay logs (`replay_<run>.json`: every eval, its args, read-back scores, verification); trial-list builder; `trials.json` (cluster paths), `trials_portable.json` (package paths), `trials_missing.json`; `snapshots_replayed/` (all 696 replayed evals) and `snapshots_rescore_grid/` (the 218 reused snapshots) |
| `rerun/` | Driver, SLURM scripts, pilot IDs, Whisper-only encoder, patched `src/` and the original harness `src/` |
| `results/results_table.csv` | One row per trial (and Whisper): code source, logged score, old/new scores (grid, all-166, all-493), status, feature dim, runtime |
| `results/results_compact.npz` | Per-trial per-electrode chunk-averaged r (old; new primary; new all-493), and per-chunk grid-mean r (old 23; new 493), with chunk lists and the primary mask |
| `results/key_models_full.npz` | Full per-chunk × per-electrode r (old and all 493 new chunks) for LEAN-v16, LEAN-v20, the kept model and Whisper |
| `results/per_trial/`, `results/trial_logs/` | Per-trial JSON records and run logs |
| `varpart/` | Phase-2/3 scripts, `RESULTS.md`, `results_{family,section,group}.csv`, `perelec_R2.npz`, `fits_primary_float32.npz` (all 135 fits, 317 chunks × 166 electrodes; enough to rerun phase 3), `old_test_reference/` |
| `figures/` | New-test Figures 1 and 2 (PNG/PDF) and their scripts, plotted data, captions, the old-test figures for comparison |
| `both_lean_whisper/` | LEAN-v20 + Whisper ("Both") fit and the LEAN-vs-Whisper variance partitioning, old and new test (§12) |

Not included (on disk): full per-chunk per-electrode arrays for every trial
(`runs-wave/newtest/results/*_rold.npy`, `*_rnew.npy`, 225 MB), full variance-partitioning fits with
all-493 and old-test r (`runs-wave/varpart-v20-newtest/fits/`, 152 MB), and the aligned segments
(`runs-wave/newtest/wave_aligned/`, 4.8 GB).

---

## 12. LEAN + Whisper ("Both") and the LEAN-vs-Whisper variance partitioning (`both_lean_whisper/`)

*Added Nov 7, 2026.* This reruns the June "Both" model (`june_diag_combined_v20.py`: LEAN-v20 and
Whisper features in one ridge) on the held-out test, together with the partitioning between the two
feature sets.

**Method (`both_varpart.py`).**
- Canonical LEAN-v20 features (3,852 raw) with the 1,280 Whisper features appended per frame:
  5,132 raw columns, 20,528 after 4 delays.
- Whisper rows are found by the MD5 fingerprint of each chunk's first window. The map is extended
  to the new chunks, and an unknown chunk raises an error.
- One streamed pass over the frozen 500 training chunks accumulates XᵀX and XᵀY. Three fits use the
  harness ridge: Both (all columns), LEAN-only and Whisper-only (column subsets of the same pass;
  per-chunk z-scoring is per column, so these equal standalone fits, as in June).
- Each model is scored on the old test and on all 493 new chunks; results are reported for the
  317-chunk primary set.
- Per electrode, R² = r·|r|:
  - unique LEAN = R²(Both) − R²(Whisper)
  - unique Whisper = R²(Both) − R²(LEAN)
  - shared = R²(LEAN) + R²(Whisper) − R²(Both)
- Summaries average R² over electrodes. 95% CIs come from a paired bootstrap over test chunks (B = 2000).

**Validation.** All six old-test values reproduce the June runs: Both 0.09204 / 0.10375 (June
0.0920 / 0.1038), LEAN 0.08079 / 0.09045 (0.0808 / 0.0904), Whisper 0.08172 / 0.09330
(0.0817 / 0.0933), all-166 / grid. The new-test LEAN-only and Whisper-only scores match the
independent fits in §7 (0.0735 and 0.0819, grid).

**Results** (unique and shared as % of R²(Both), [95% CI]):

| Test | Electrodes | r Both | r LEAN | r Whisper | Unique LEAN | Unique Whisper | Shared |
|---|---|---|---|---|---|---|---|
| old (23) | grid | 0.1038 | 0.0904 | 0.0933 | 6.5% [3.0, 11.0] | 25.4% [21.3, 29.0] | 68.2% [63.3, 73.9] |
| old (23) | all-166 | 0.0920 | 0.0808 | 0.0817 | 7.8% [4.1, 12.8] | 24.2% [20.5, 27.4] | 68.0% [63.3, 73.5] |
| **new (317)** | **grid** | **0.0871** | **0.0735** | **0.0819** | **5.0% [4.0, 6.1]** | **26.0% [24.4, 27.5]** | **68.9% [65.6, 72.2]** |
| new (317) | all-166 | 0.0755 | 0.0646 | 0.0703 | 5.7% [4.6, 6.9] | 25.3% [23.8, 26.8] | 68.9% [65.4, 72.4] |
| new (493) | grid | 0.0810 | 0.0690 | 0.0752 | 5.8% [4.9, 6.8] | 25.3% [23.8, 26.8] | 68.9% [65.1, 72.9] |

- **The partition holds on held-out data.** About 69% of the Both model's variance is shared,
  about a quarter is unique to Whisper and 5–6% is unique to LEAN. The old-test values fall inside or
  next to the new CIs, which are much narrower (317 chunks vs. 23).
- **Adding LEAN to Whisper still helps.** Both beats Whisper by +0.0052 r on the grid (new test), down
  from +0.0105 on the old test; the unique-LEAN share itself barely moves (6.5% → 5.0%).

**Files.**
- `RESULTS.md` and `results.csv`: all tables above, with the CIs and R² values.
- `perelec.npz`: per-electrode arrays for brain maps. Keys are `{old,new_primary,new_all}__{r_both,
  r_lean,r_whisper,R2_both,R2_lean,R2_whisper,unique_lean,unique_whisper,shared}`, each of length 166
  (electrodes 0–124 grid, 125–131 strips, 132–165 depth).
- `perchunk_r.npz`: per-chunk, per-electrode r for each model on the old test and all 493 new chunks,
  with the chunk lists.
- Scripts and job records: `run_fit.sh`, `run_agg.sh`, `jobs.txt`. Fit 29293608: 20 minutes on one
  A100, about 0.35 GPU-hours. A first submission, 29293422, stopped on the frozen-training-list check
  because the harness environment variables were missing; nothing was fit.
