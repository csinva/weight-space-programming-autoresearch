# LEAN: hand-built audio features vs. Whisper for patient1 ECoG — complete replication report

Written Nov 5 2026. This document describes everything needed to understand and replicate the
"waveform → ECoG" autoresearch project for patient patient1: the evaluation harness, the
autoresearch loop that produced ~900 hand-built encoder attempts (Jun 1–8 2026), the comparison
against Whisper, and the follow-up analyses done on Nov 5 2026 (feature catalog, variance
partitioning, bug audit and the LEAN-v21 bug-fix release, the grid-electrode rescore of the
whole history, and the summary figures). Patient identity: only the study ID **patient1** is used
anywhere; no names appear in code, data paths, or this document.

The package mirrors the repository root `circuit-building-autoresearch/`, so every relative
path below (and every `src` symlink inside the run folders) resolves inside the unzipped
package. Absolute data paths refer to the cluster these runs were done on (a university cluster,
SLURM); see §12 to point them elsewhere.

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [Data, task and metric](#2-data-task-and-metric)
3. [The evaluation harness (`evolve-wave/src`)](#3-the-evaluation-harness)
4. [The autoresearch loop](#4-the-autoresearch-loop)
5. [Chronology of runs](#5-chronology-of-runs)
6. [The LEAN encoder lineage v16 → v20 and the Whisper comparison](#6-the-lean-lineage-and-the-whisper-comparison)
7. [LEAN-v20 feature catalog](#7-lean-v20-feature-catalog)
8. [Variance partitioning (LEAN-v20)](#8-variance-partitioning-lean-v20)
9. [Bug audit and LEAN-v21](#9-bug-audit-and-lean-v21)
10. [Rescoring the whole history on the 125 grid electrodes](#10-rescoring-the-whole-history-on-the-grid-electrodes)
11. [Summary figures](#11-summary-figures)
12. [How to replicate, step by step](#12-how-to-replicate-step-by-step)
13. [Gotchas (read before running anything)](#13-gotchas)
14. [Key numbers in one place](#14-key-numbers-in-one-place)
15. [File map of this package](#15-file-map-of-this-package)
16. [SLURM job log](#16-slurm-job-log)
17. [Limitations and open directions](#17-limitations-and-open-directions)

---

## 1. What this project is

**Question.** How close can a fully *hand-designed*, interpretable audio front-end get to a
large pretrained speech model (Whisper) at predicting cortical high-gamma activity, when both
are read out by the same linear ridge regression?

**Constraint ("no training").** Every feature is computed in closed form from the raw audio
waveform: filterbanks, compression, adaptation with fixed time constants, modulation filters,
etc. Nothing in the front-end is fit to data — no gradient descent, no PCA/ICA/NMF/k-means, no
pretrained weights. The *only* data-fit step is the final per-electrode ridge readout, which is
identical for the hand-built features and for Whisper. (Rules: `evolve-wave/program.md`.)

**How the features were found.** An "autoresearch" loop: Claude Code agents (model `opus`,
non-interactive sessions launched inside SLURM jobs) repeatedly edited a single encoder file
(`wave_encoder.py`), ran the fixed evaluation, logged the score to
`results/overall_results.csv`, snapshotted the file to `wave_encoders_lib/<name>.py`, and
kept or reverted the change. The best encoders were periodically "promoted" to a canonical
baseline named **LEAN-vN** ("LEAN" = the lean hand-built encoder line).

**Headline result.** On the 125 surface-grid electrodes the hand-built line rose from 0.045
(bare gammatone cochleagram) to 0.0909 (best single trial) / 0.0904 (LEAN-v20), against 0.0933
for Whisper encoder features with the same ridge. Combining LEAN-v20 with Whisper in one ridge
gives 0.1038, so the two carry partly different information. On all 166 electrodes the gap
closed to 0.0009 (LEAN-v20 0.0808 vs Whisper 0.0817). Variance partitioning shows ~89% of
LEAN's explained variance is shared across feature families; the largest unique contributors
are the modulation filters and the slow-context ("tonic") cochleagrams.

---

## 2. Data, task and metric

**Subject.** patient1, intracranial EEG (ECoG grid + strips + depth), 7 days of continuous
recording (Aug 1–7 2019, 15 sessions), segmented into ~116 s "chunks" with IDs
`{8-hex session}_{4-digit index}` (e.g. `2c159415_0003`).

**Responses.** High-gamma ECoG at 30 Hz, 166 good channels:
`/path/to/ecog_data/wav_files/{chunk}_ecog_hasson_highgamma.npy`,
shape (166, T). Electrode order: **0–124 = surface grid (G2–G128), 125–131 = strips,
132–165 = depth.** "Grid" in this document always means electrodes 0–124.

**Stimulus.** The raw room audio at 44.1 kHz, pre-cut per chunk into aligned segments:
`/path/to/scratch/patient1_wave_aligned/{chunk}_wave.npy` (float16, 1-D). Made by
`evolve-wave/extract_wave_aligned.py` from the chunk wav files so that feature frame *i* is
centered at wav sample (i + 30)·1470 (the main project's 1 s extraction crop is baked in), with
2048 samples (WINMAX/2) of real context on each side. `--verify <chunk>` proves alignment.

**Framing.** `src/data.frame_windows(seg, win, n_frames)` returns one window of `win` samples
(≤ 4096 = 92.9 ms) per 30 Hz frame (hop 1470 samples = 33.3 ms), centered on the mel/ECoG frame
grid. Encoders must not re-window the audio themselves.

**Chunk lists.** `master_list.json` (main project) lists 2,293 valid chunks; the harness keeps
the 723 that have an aligned wave segment and a high-gamma file (700 outside the test session), shuffled with seed 42.
- **Test set (fixed):** `evolve-wave/speech_rich_test_chunks.json` — 23 chunks
  `450a36e4_0190 … 450a36e4_0212`, a speech-rich block (a podcast playing in the room).
- **Training set:** the first 500 of the remaining chunks after excluding every chunk of
  session `450a36e4` (env `ECOG_EXCLUDE_PREFIX=450a36e4`), i.e. a session holdout.
- Environment variables that select this split (set by every loop script):
  `ECOG_TEST_CHUNKS_FILE=<repo>/evolve-wave/speech_rich_test_chunks.json`,
  `ECOG_EXCLUDE_PREFIX=450a36e4`, `ECOG_RIDGE_GPU=1`; call `data.get_chunk_names(500, 100)`
  (the 100 is ignored in fixed-test mode). Training rows after alignment: **1,750,593 frames**.

**Pipeline per chunk** (`src/features.get_features_per_chunk`): encoder → (T, H) features →
4 FIR delays (0–3 frames, delay-major column order `d*H + j`) → 100 ms neural lag (features at
t predict high-gamma at t+3 frames) → drop the first 30 frames → z-score every column and every
electrode **within the chunk** (zero-variance columns stay 0).

**Readout** (`src/encoding.bootstrap_ridge_gpu`): per-electrode ridge with
`alphas = logspace(1, 4, 12)` (the code uses α² as the penalty, i.e. λ = 1e2 … 1e8). The alpha
for each electrode is the argmax of mean tune-set correlation over 5 bootstrap splits, each
holding out 20 random blocks of 40 consecutive training frames (`_temporal_chunk_splits`,
seed 42). Final weights from the eigendecomposition of the full-train XᵀX (float64).

**Metric** (`src/eval._per_chunk_elec_corr`): for each test chunk, Pearson r between predicted
and actual high-gamma for each electrode; average over electrodes; average over chunks. This is
the project's canonical "per-sample per-electrode correlation". Grid scores average electrodes
0–124 only. Because alpha is chosen independently per electrode, a grid score computed from an
all-166 fit is identical to a fit on the grid electrodes alone (verified: v20 0.0904 both ways).

**Caveat on absolute numbers.** The fixed test set is speech-rich and from one session; on
random 100-chunk splits both LEAN and Whisper score ~0.015 lower in r, but the LEAN-vs-Whisper
gap was the same on the fixed set and on four random splits (+0.0061 vs +0.0064 ± 0.002;
`autoresearch_loop/whisper_vs_lean_random.sh`, `random_splits/`). Relative comparisons are
trustworthy; absolute values are optimistic.

---

## 3. The evaluation harness

`evolve-wave/src/` (symlinked as `src/` into every run folder; agents were told never to edit it):

| File | Role |
|---|---|
| `data.py` | constants (SR 44.1 kHz, OUT_SR 30, HOP 1470, WINMAX 4096, 166 electrodes, data paths), `frame_windows`, `get_chunk_names`, loaders (audio cast to float32) |
| `features.py` | delays, lag (3), trim (30), per-chunk z-scoring |
| `encoding.py` | bootstrapped ridge; GPU path via streamed XᵀX eigendecomposition (default when CUDA is available; validated identical to the CPU SVD path, `evolve-wave/validate_gpu_ridge.py`) |
| `eval.py` | `EncodingConfig`, `run_encoding`, the metric, results-CSV helpers |

`evolve-wave/setup_run_wave.py <tag>` creates a run folder `runs-wave/<tag>/` with the `src`
symlink, a starting `wave_encoder.py`, `program.md`, `results/` (seeded with reference rows) and
`wave_encoders_lib/`. Each encoder file exposes `build_embedder(device=...)` returning a callable
`(T, win) windows → (T, H) features` with a `.win` attribute, a `model_shorthand_name`, a
`model_description`, and a `__main__` that calls `run_encoding` and appends a CSV row.

Results CSV columns: `subject,test_corr,train_corr,frac_test_elec_above_0.1,encoding_seconds,
status,model_shorthand_name,n_params,description`. `test_corr` is the metric above (all 166
electrodes until Jun 6, then grid-only for the "exceed" runs, whose rows end in `GRID-ONLY`).

---

## 4. The autoresearch loop

**Launchers** (`autoresearch_loop/`): each is a SLURM batch script that runs Claude Code
(`claude --model opus --dangerously-skip-permissions`) in a loop inside one run folder with a
long prompt (the research brief), restarting the session when it ends. The worker writes code,
runs `python wave_encoder.py --num-train 500 --num-test 100 > run.log 2>&1` in the foreground
on the job's GPU, reads the result, snapshots, and repeats. Prompts are in the scripts.

| Script | Run folder(s) it drove | Brief |
|---|---|---|
| `run_loop_wave.sh` | `wave-slurm-23628821` | open hand-design exploration of the waveform front-end |
| `trim_agent.sh` | `wave-trim` | remove redundant blocks + try literature features; produced LEAN-v16 |
| `loop_forever.sh` | `wave-lean` | continued lean branch |
| `reset_worker.sh A-D`, `reset_worker2.sh E-I` | `reset-A` … `reset-I` | clean-slate paradigm workers (one paradigm each, seeded with LEAN-v16 code, "novelty required") |
| `graft_worker.sh` | `graft-lean` | greedy graft of the best reset-worker blocks onto LEAN-v16 |
| `lean_interp_worker.sh` | `lean-interp` | interpretable conditioning (stage 3) and nonlinear interaction terms (stage 2); produced LEAN-v17 |
| `ling_eval.sh` | `lean-interp-ling` | linguistic (transcript-derived) features via audio fingerprint lookup |
| `exceed_worker.sh <mission>` | `exceed-dense/frontend/scatter/combine/v18/v20-context/v20-deep/v20-speaker` | grid-only "exceed Whisper" missions; produced v18–v20 |
| `whisper_vs_lean_random.sh` | `whisper-cmp` | random-split sanity check of the LEAN-vs-Whisper gap |
| `banded_probe_worker.sh`, `banded_cache.sh` | (banded-ridge probe) | banded ridge never beat uniform ridge |

Clean-slate paradigms: A biophysical auditory periphery; B cortical spectro-temporal modulation
(Chi–Ru–Shamma); C phonological features; D wavelet scattering; E divisive normalization;
F predictive coding / surprise; G sound-texture statistics; H fixed random (echo-state)
reservoir; I temporal fine structure / phase.

Agent driver logs (`autoresearch_loop/logs/`, 420 MB of session transcripts) are **not**
included; everything needed is in the CSVs, snapshots and scripts.

**Snapshot caveat (important for replication, §10).** Snapshots are written by the agent and
are not always the code that produced the logged score: some were saved after a revert, some
were overwritten later, and the `lean-interp` agent chose variants by editing argparse defaults
of a single parameterized harness, so its snapshot files often contain a later harness state.

---

## 5. Chronology of runs

All on the fixed test set; "all" = 166 electrodes, "grid" = 125. Trial counts exclude seeded
reference/baseline rows.

| Dates (2026) | Run folder | Logged trials | Metric | Best | Outcome |
|---|---|---|---|---|---|
| Jun 1 | `evolve-ecog/`, `runs-ecog/` | — | all | — | melspec → ECoG precursor (ridge floor 0.0367 on a tiny pilot) |
| Jun 3 | `wave-slurm-23628821` | 155 | all | 0.0770 | hand-design exploration (gammatone, adaptation, flux, modulation, STMF, tonic/acausal context) |
| Jun 3 | `wave-trim`, `wave-lean` | 24 + 20 | all | 0.0768 | trim + literature features; **LEAN-v16** = 0.0760 (1,558 raw dims) |
| Jun 4–5 | `reset-A` … `reset-I` | 7–46 each | all | 0.0726 (C) | clean-slate paradigms; all below LEAN-v16 |
| Jun 5 | `graft-lean` | 17 | all | 0.0764 | grafting reset blocks onto v16: +0.0004, within noise |
| Jun 5 | `lean-interp-ling` | 2 | all | 0.0766 | lexical features from transcripts |
| Jun 5 | `lean-interp` | 75 | all | 0.0778 | **LEAN-v17** = v16 + 405 interaction terms ("v3e") |
| Jun 6 | `exceed-dense` | 5 | all | 0.0778 | dense soft maps (no gain) |
| Jun 6 | `exceed-frontend`, `exceed-scatter` | 117, 79 | grid | 0.0878, 0.0882 | front-end features (only spectral contrast helped); wavelet scattering |
| Jun 6 | `exceed-combine` | 18 | grid | 0.0893 | **LEAN-v18** = contrast family + scattering (all 0.0800 / grid 0.0891) |
| Jun 7 | `exceed-v18` | 55 | grid | 0.0905 | quadrature banks + voicing gates → **LEAN-v19** (q29, grid 0.0903); slow quadrature bank → **LEAN-v20** (q45, grid 0.0904) |
| Jun 7–8 | `exceed-v20-context/deep/speaker` | 11, 18, 16 | grid | 0.0905, **0.0909**, 0.0905 | reservoir context, cross-band coupling/PAC, speaker normalization: all within noise of v20 |
| Jun 7 | `lean-whisper` | — | per-electrode | — | canonical LEAN vs Whisper diagnostics (§6) |
| Jun 8 | `openaudio`, `openaudio2` | (excluded) | grid signed R² | — | *trained* models on LEAN-v17 + Whisper + mel; separate study, not in this package |
| Nov 5 | `varpart-v20`, `lean-v21`, `varpart-v21`, `rescore-grid`, `summary_figures` | — | — | — | this document's follow-up analyses |

Noise floor: a paired bootstrap over test chunks put single-comparison noise at about ±0.0008
in r (`evolve-wave/boot_ci.py`); the loops used +0.0015 as the "real gain" threshold for larger
blocks.

---

## 6. The LEAN lineage and the Whisper comparison

Promoted versions (encoders in `runs-wave/wave-trim/wave_encoders_lib/LEAN-v16.py … LEAN-v20.py`;
**LEAN-v21.py in that folder is an unrelated older file from Jun 3** — the real v21 is
`runs-wave/lean-v21/LEAN-v21.py`). Raw dims are before the 4 delays.

| Version | What it added | All 166 | Grid 125 |
|---|---|---|---|
| v16 | gammatone cochleagram × 3 windows, multi-τ adaptation, pitch, spectral shape, flux, 1-D modulation, STMF, formants, slow/lookahead context (1,558 dims) | 0.0760 | 0.0852 (rescored, §10) |
| v17 | + 405 interpretable interaction terms (asymmetric ReLU/cube transforms, multi-scale deviations, cross-lag and gated products) | 0.0778 | 0.0866 |
| v18 | + CPP, region flatness, spectral/cochleagram contrast family (192 dims), wavelet scattering (287 dims) | 0.0800 | 0.0891 |
| v19 | + quadrature (phase-keeping) modulation bank, its deviations and acausal deviation, cross-rate energy, voicing-gated cochleagram, gate signals | 0.0805 | 0.0903 |
| v20 | + slow quadrature bank (0.5–2 Hz) — 3,852 dims | 0.0808 | 0.0904 |
| v21 | v20 with bugs fixed, no new features (3,392 dims; §9) | 0.0805 | 0.0902 |

**Whisper baseline.** Whisper-large-v3 encoder features (`last_hidden_state`, 1280-dim),
extracted by `main_project/extract_whisper_features.py` with causal right-aligned 30 s windows
(≤ 1 s look-ahead), resampled to the 30 Hz grid and stored as
`wav_files/{chunk}_whisper.npy` (T, 1280) float16. Scored with the identical harness and 4
delays (`evolve-ecog/whisper_fir.py`; in `runs-wave/lean-whisper/` the Whisper block is injected
into the encoder by audio-fingerprint lookup so it goes through exactly the same pipeline).

Canonical per-electrode comparison (`runs-wave/lean-whisper/diag_*.out`, same split and ridge):

| Model | All 166 | Grid | Strips | Depth |
|---|---|---|---|---|
| Whisper | 0.0817 | 0.0933 | 0.0268 | 0.0505 |
| LEAN-v17 | 0.0778 | 0.0866 | 0.0243 | **0.0561** |
| LEAN-v20 | 0.0808 | 0.0904 | 0.0259 | 0.0566 |
| LEAN-v20 + Whisper (one ridge) | **0.0920** | **0.1038** | 0.0318 | 0.0614 |

LEAN beats Whisper on depth electrodes; all of Whisper's advantage is on the surface grid,
which is why the Jun 6+ search optimized the grid score. (The Jun 4–5 notes quote "Whisper
0.0821"; that was an all-166 run with a different script, `whisper_fir.py`; the canonical
same-split diagnostic gives 0.0817. Use 0.0817 / 0.0933.)

---

## 7. LEAN-v20 feature catalog

Full catalog: `runs-wave/wave-trim/wave_encoders_lib/LEAN-v20_FEATURES.md` — every one of the 51
feature groups with exact column ranges, formulas, layouts, time support (causal / acausal /
look-ahead), filterbank center frequencies, version lineage, and quirks. The group sizes there
sum to exactly 3,852 raw dims = 15,408 delayed columns, matching the scored runs.

Families (raw dims): cochleagram core + slow context + look-ahead 960; modulation filters
(1-D, STMF, quadrature, slow quadrature) 1,803; pitch/voicing/voice quality 122; spectral shape,
flux, formants 83; contrast 192; interactions (v3e) 405; wavelet scattering 287.

**Trap:** `build_embedder()` in LEAN-v17…v20 defaults to `interact_set="none"` (3,447 dims);
the scored configuration is `interact_set="v3e"` (set by the `__main__` argparse default).
Always pass `interact_set="v3e"` when importing v17–v20.

---

## 8. Variance partitioning (LEAN-v20)

Code: `runs-wave/varpart-v20/` (`groups.py`, `phase1_stats.py`, `phase2_fit.py`,
`phase3_aggregate.py`, `run_phase{1,2,3}.sh`). Results: `RESULTS.md`,
`results_{family,section,group}.csv`, `perelec_R2.npz`, `fits/*.npz`.

**Method.**
1. *Phase 1 (one GPU job, ~8 min on an A100):* stream the canonical v20 features chunk by chunk
   (same `get_features_per_chunk`, `interact_set="v3e"`) and accumulate the sufficient statistics
   XᵀX (15,408², float64), XᵀY, the exact rows the harness's 5 bootstrap tune splits use
   (4,000 rows), and the test-chunk features. The group layout the embedder actually emits is
   checked against `groups.py` on the first chunk. Output `stats/` (~6.7 GB; not shipped).
2. *Phase 2 (45-task array):* for each column subset S, refit with exactly the harness
   algorithm restricted to S: XᵀX[S,S], XᵀY[S], tune rows[:, S], per-electrode alpha by the
   same 5 bootstrap splits, final eigendecomposition, per-chunk per-electrode test r saved
   as (23 × 166). 135 deduplicated fits: the full model plus "alone" and "drop" fits for each
   unit at three levels (7 families, 16 sections A–P, 51 groups).
3. *Phase 3 (CPU):* variance explained per electrode R²ₑ = rₑ·|rₑ| (signed), rₑ = canonical
   per-chunk-averaged r; summaries average over electrodes. **Alone** = R²(unit only);
   **unique** = R²(full) − R²(full without unit); both reported as % of the full model; shared =
   1 − Σ unique shares within a level. 95% CIs: paired bootstrap over the 23 test chunks
   (B = 2000).

**Validation.** The phase-2 full fit reproduced the canonical v20 run exactly: grid 0.0904,
all 0.0808, per-electrode correlation with `diag_perelec_v20.npz` 1.00000, max |diff| 0.00000.

**Results (grid, full R² = 1.894%).**

| Family | Dims | Alone (% of full) | Unique (% of full) [95% CI in R² %] |
|---|---|---|---|
| Modulation filters | 1803 | 91% | 4.0% [+0.032, +0.108] |
| Cochleagram + slow context + look-ahead | 960 | 84% | 3.1% [+0.012, +0.137] |
| Wavelet scattering | 287 | 77% | 1.6% [+0.017, +0.049] |
| Interactions (v3e) | 405 | 82% | 1.5% [+0.014, +0.040] |
| Contrast | 192 | 72% | 1.0% [+0.005, +0.034] |
| Spectral shape, flux, formants | 83 | 69% | 0.4% |
| Pitch, voicing, voice quality | 122 | 61% | 0.4% |
| **Shared** | | | **88.1%** (all-166: 86.4%) |

Largest single group: `tonic` (slow 2 s / 5 s smoothed cochleagrams), unique 2.4%. Several big
blocks have no unique variance (STMF 357 dims, slow quadrature bank — v20's own addition).
`spectral_flatness` (1 dim) alone reaches 30% of the full model: the predictable signal is
low-dimensional. Sections and groups: see `RESULTS.md`.

---

## 9. Bug audit and LEAN-v21

### 9.1 How the bugs were found
1. Reading the code for the catalog (dead S2 scattering paths, a zero contrast octave,
   flatness gates that can never open, the `build_embedder` default).
2. Measuring on real audio (`best_strength` is ~6.6–10.2, so `sigmoid(8·(s − 0.3))` = 1.0 on
   100% of frames; spectral flatness ratio ≤ 0.12 even in 50–8000 Hz, so 0.2–0.6 gates never
   open; the ACF voicing peak `hnr_peak` spans 0.04–0.88).
3. **A systematic scan** (`runs-wave/lean-v21/degenerate_scan.py`, `degenerate_detail.py`):
   from the phase-1 XᵀX (500 training chunks) compute each column's variance and all pairwise
   correlations of the delay-0 block. v20: **22 constant columns and 489 columns with |r| >
   0.9999 to an earlier column** (`lean-v21/degenerate_v20_detail.txt`).

### 9.2 Root causes and fixes (`runs-wave/lean-v21/make_v21.py` derives LEAN-v21.py from v20 by asserted edits)

| Bug | Fix in v21 | Raw cols removed |
|---|---|---|
| ±0.5 cyc/ch 2-D Gabors are identical (channel Nyquist: sweep direction undefined) in STMF and both quadrature banks | keep +0.5 only | 294 (incl. their deviation / acausal copies) |
| 12 Hz quadrature cos/sin at scales 0 and 0.5 equal the 8 Hz ones (σₜ = period/2π is < 1 frame, both kernels collapse to the same 3-tap shape) | drop those components | 60 |
| scattering 2nd-order paths at 0.5/1/2 Hz have no support on the 10.8 Hz FFT grid of a 4096-sample window (dead/constant) | start 2nd-order paths at 4 Hz | 54 |
| slow-bank raw 2 Hz rate duplicates the quadrature bank's 2 Hz | drop raw 2 Hz (keep its τ=30 deviation, which is not a duplicate) | 36 |
| flatness gates never open | remove | 12 |
| short-window contrast 62–125 Hz octave has 1 FFT bin → always 0 | skip bands with too few bins | 4 |
| voicing gates driven by `best_strength` (saturated) → `cochlea_voiced_gated(_dyn)` were exact copies of `cochleagram` / `adapt_delta` | drive gates with `hnr_peak` (0–1), same thresholds | — (rewired) |
| `build_embedder()` default `interact_set="none"` | default `"v3e"` | — |
| dead code (unused filterbanks, DoG, replication machinery, computed-but-unused scattering blocks) and wrong comments | removed / corrected | — |
| **Total** | 3,852 → **3,392** raw dims | **460** |

**Not fixed (design choices, flagged):** the 2-D Gabor time window σₜ = period/2π is too narrow
for real rate tuning at ≥ 8 Hz (source of the residual near-duplicates); the "vowel" and
"plosive" templates are nearly sign-flipped copies.

### 9.3 Verification
- `lean-v21/verify_v21.py` (A100, same process, 2 test chunks): every v21 column mapped to its
  v20 source is **bit-identical**, except scattering S2-entropy (2e-5 relative, expected —
  removed near-zero paths change its normalization) and the groups changed by design.
- Rescan of v21 (`varpart-v21/degenerate_v21.txt`): **0 constant, 31 near-duplicates**
  (29 from the narrow Gabor window, 2 vowel/plosive).

### 9.4 v21 scores and breakdown (`runs-wave/varpart-v21/`, same pipeline as §8)
Grid 0.0902 vs v20 0.0904 (Δr −0.0003 [−0.0006, −0.0000]); all 0.0805 vs 0.0808 (Δr −0.0003
[−0.0006, +0.0000]). Duplicated columns had effectively received less ridge penalty; removing
them costs a hair. Breakdown structure unchanged: shared 89.2% (families), 92.8% (sections),
94.9% (groups); unique shares modulation 4.0% [2.1, 5.6], cochleagram + context 3.6% [1.2, 7.8],
interactions 1.3%, scattering 1.0%, contrast 0.8%, pitch/voicing 0.1%, shape ~0. The rewired
voicing gates now carry signal (alone 32% of full, was 0%) but no unique variance. Because v21
scores slightly lower, Figure 1 ends at v20; Figure 2 shows the v21 breakdown (cleaner features).

---

## 10. Rescoring the whole history on the grid electrodes

**Why.** Jun 3–5 trials were scored on all 166 electrodes and Jun 6+ trials on the 125 grid
electrodes, so the history could not be drawn on one axis. The early runs only logged the
166-electrode mean, so every early trial had to be re-run.

**Code:** `runs-wave/rescore-grid/`.
- `make_trials.py` → `trials.json` (510 early trials with a recoverable snapshot) and
  `trials_missing_snapshot.json` (85 with none: wave-slurm 51, wave-trim 5, wave-lean 4,
  graft-lean 7, lean-interp 5, reset-H 5, reset-I 2, reset-A/C/E/F 1 each, lean-interp-ling 2).
  Snapshots are matched by the `model_shorthand_name` written *inside* the file (filenames are
  not reliable), falling back to the filename.
- `src_patched/`: a copy of the harness package in which `eval.run_encoding` is redirected
  (when `RESCORE_OUT` is set) to `rescore_stream.rescore`, which streams the features, runs
  exactly the harness ridge, and saves per-chunk per-electrode test r. Every other file is a
  symlink to the original `evolve-wave/src`. `src_patched/__init__.py` sets PyTorch's
  `expandable_segments` allocator (see below).
- `rescore_task.py`: for each trial, makes a sandbox `work/<id>/` with `src → src_patched`,
  copies the snapshot to `wave_encoder.py`, and runs the snapshot's **own `__main__`**
  (`python wave_encoder.py --num-train 500 --num-test 100` + optional per-trial `args`) so the
  encoder is configured exactly as it was originally. Then it checks that the recomputed
  all-166 score reproduces the logged score within **±0.00015** (logged values have 4 dp);
  status `ok`, `mismatch` or `failed`; results in `results/<id>.json` and `<id>_r.npy`
  (23 × 166 per-chunk per-electrode r).
- Array mode is **work-stealing**: each task takes its own share, then keeps claiming any
  unclaimed trial (atomic `mkdir claims/<id>`; flock is unreliable on this NFS) until none
  remain or < 45 min of walltime is left; claims whose owner job has left the queue are taken
  over.

**Problems found and how they were handled** (each was caught by the reproduction check):

| Problem | Trials | Diagnosis | Handling |
|---|---|---|---|
| `lean-interp` snapshots overwritten by later harness states; variants were chosen by edited argparse defaults | 55 of 69 | pilot trial INTERP-v2zz reproduced v2z's score | `recover_interp.py`: run each trial from a harness file that contains its branch (`lean-v17-clean/enc_orig.py` = the original 3,665-line harness, or the own/neighbor snapshot for v3f–v3n) with explicit `--interact-set <set> --cond-mode none`; v1a/v1b (stage-3 harness without `--interact-set`) use only `--cond-mode rep_target60_k8 / rep_target120_k16` → all 69 reproduce |
| **NumPy 2 type promotion (NEP 50).** The venv's NumPy/torch were reinstalled on Jun 5 08:08 (now NumPy 2.2.6); e.g. `kernel.astype(np.float32) / np.sqrt(...)` is float32 under NumPy 1.x but float64 under NumPy 2 → `conv1d`/`matmul` dtype errors or silent numeric shifts | all 38 reset-D (scattering), reset-H (reservoir, pre-emptive), GRAFT-v2c (reservoir), reset-G TEXTURE-v22 | confirmed directly: the same line gives float64 under 2.2.6 and float32 under 1.26.4 | per-trial `PYTHONPATH` overlay with NumPy 1.26.4 installed to an isolated directory (`uv pip install --python .venv/bin/python --target rescore-grid/np126 --no-deps numpy==1.26.4`); torch 2.10 works with it; shared venv untouched → all 61 overlay trials reproduce |
| GPU OOM on 80 GB A100 for large BIOPHYS (reset-A) encoders | 4 | fragmentation (~19 GB reserved-unallocated) from keeping XᵀX on the GPU across chunks | `expandable_segments:True` set in `src_patched/__init__.py` → all 4 reproduce |
| `wave-trim` old-numbering LEAN-v17…v20 (Jun 3) | 4 | snapshot files overwritten on Jun 5–7 by the promoted v17–v20 of the same name | excluded (original code gone) |
| `wave-slurm` chanvar3, `reset-G` TEXTURE-v8 | 2 | snapshot saved after a revert (chanvar3 reproduces its predecessor exactly; TEXTURE-v8 gives the same value under both NumPy versions) | excluded |
| GPUs listed free but always "busy" | nodes node-X, 4042, 4012, 4037 | repeated self-requeues landing on the same nodes | added to `ExcNodeList` |

**Outcome:** 510 / 510 run; **504 reproduce** (median |Δ| 0.00003, max 0.00013); 6 excluded.
Including the 85 never-snapshotted trials, 811 of 902 attempts in the whole history (595 early,
307 grid-phase) have a grid score.

---

## 11. Summary figures

Folder `runs-wave/summary_figures/` (captions in `CAPTIONS.md`).

**Figure 1 — `fig1_progression.{png,pdf}`** (`plot_progress_grid.py`; data `fig1_data.csv`):
the layout of the June progress plots (`analyze/waveform_progress.png`). x = every main-line
encoder trial in chronological order; y = test r on the 125 grid electrodes (rescored values for
Jun 3–5, logged grid values for Jun 6+); faint dots = attempts, black step line = running best,
shaded bands = phases (hand-design exploration; trim + literature → v16; graft + interactions →
v17; front end + scattering → v18; quadrature banks + gates → v19/v20; context, coupling,
speaker), callouts at the major feature additions and versions v16–v20, dashed line = Whisper
0.0933. 525 of 601 main-line trials plotted; the clean-slate side workers are not on the main
line (as in the June figure). Best single trial 0.0909 (`LEAN-v35-AUTOLAGZ`, exceed-v20-deep).

**Figure 2 — `fig2_breakdown.{png,pdf}`** (`make_summary_figures.py`): LEAN-v21 variance
partitioning on the grid: (a) each family alone vs. unique with CIs; (b) unique vs. shared at
three granularities; (c) unique variance per section, v21 with CIs and v20 for comparison;
(d) group size vs. variance alone (blue = unique CI above zero).

Style: Okabe–Ito audio blue (#0072B2) as the single accent, grays for context, American English,
no viridis-family colormaps.

---

## 12. How to replicate, step by step

### 12.1 Environment
- Python 3.10.18; create the venv with `uv sync` from `pyproject.toml` / `uv.lock` at the package
  root (`.venv/`). Key versions (`ENVIRONMENT.txt`, `ENVIRONMENT_pip_freeze.txt`): torch 2.10.0
  (CUDA 12.8), NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.8.
- **GPU: NVIDIA A100 for every scored number.** Do not mix GPU types for feature extraction:
  cuDNN convolutions use TF32 on A100 by default, so features differ slightly on other GPUs.
  The ridge itself is float64 and GPU-independent (V100s are fine for fits that fit in 16 GB).
- NumPy 1.26 overlay for pre-Jun-5 encoders (reset-D, reset-H, GRAFT-v2c, TEXTURE-v22):
  `uv pip install --python .venv/bin/python --target runs-wave/rescore-grid/np126 --no-deps numpy==1.26.4`.

### 12.2 Data
On the original cluster nothing needs to change. Elsewhere, edit the constants at the top of
`evolve-wave/src/data.py` (`PROJECT_ROOT`, `WAV_DIR`, `MASTER_LIST`, `WAVE_DIR`) and provide:
`{chunk}_ecog_hasson_highgamma.npy` (166, T); `{chunk}_wave.npy` aligned audio segments
(`evolve-wave/extract_wave_aligned.py`); `master_list.json`; Whisper features
`{chunk}_whisper.npy` for the Whisper comparison (`main_project/extract_whisper_features.py`).
No ECoG, audio, Whisper features or transcripts are included in this package.

### 12.3 Score any encoder (the canonical number)
```bash
cd runs-wave/<some-run>            # any run folder: has the src -> ../../evolve-wave/src symlink
cp wave_encoders_lib/<name>.py wave_encoder.py
export ECOG_TEST_CHUNKS_FILE=$PWD/../../evolve-wave/speech_rich_test_chunks.json
export ECOG_EXCLUDE_PREFIX=450a36e4 ECOG_RIDGE_GPU=1
python wave_encoder.py --num-train 500 --num-test 100          # appends a row to results/
```
LEAN-v18…v20 `__main__` take `--n-grid 125` for the grid score (default 166) and
`--interact-set v3e` (default). Memory: v20 needs ~150–200 GB host RAM in the in-memory harness;
the streamed scripts below need < 20 GB. Expect 5–15 min per encoder on an A100.

### 12.4 Whisper comparison
`runs-wave/lean-whisper/diag_canonical.py` (LEAN-v17 vs Whisper, per electrode),
`diag_v20.py` (v20), `diag_combined_v20.py` (v20 + Whisper); SLURM wrappers `diag_*.sh`
(gpu_short, 400 GB RAM because the combined design matrix is ~144 GB).
`evolve-ecog/whisper_fir.py` sweeps delays for Whisper alone.

### 12.5 Variance partitioning (§8)
```bash
cd runs-wave/varpart-v20
sbatch run_phase1.sh                                  # streamed stats (writes stats/, ~6.7 GB)
sbatch --dependency=afterok:<phase1> run_phase2.sh    # 45 tasks, 135 fits
sbatch --dependency=afterany:<phase2> run_phase3.sh   # tables -> RESULTS.md
```
Check `RESULTS.md` "Full model (validation)": grid 0.0904 / all 0.0808, per-electrode corr 1.0.
Phase 2 on 16 GB V100s needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in
`run_phase2.sh`) and keeps XᵀX in host RAM (already the case in `phase2_fit.py`).

### 12.6 LEAN-v21 (§9)
```bash
cd runs-wave/lean-v21
python make_v21.py                     # regenerates LEAN-v21.py from LEAN-v20.py (asserted edits)
sbatch run_verify.sh                   # v20 vs v21 column mapping check -> verify_*.out "RESULT"
cd ../varpart-v21 && sbatch run_phase1.sh   # then run_scan.sh, run_phase2.sh, run_phase3.sh as in 12.5
```

### 12.7 Grid rescore (§10)
```bash
cd runs-wave/rescore-grid
python make_trials.py && python recover_interp.py   # rebuilds trials.json from the run folders
```
The shipped `trials.json` is the **final** version and already includes the manual corrections
made during the run (skip flags for the 4 overwritten `wave-trim` trials, the v1a/v1b
`--cond-mode`-only flags, and `pythonpath` → `np126` for reset-D, reset-H, GRAFT-v2c and
TEXTURE-v8/v22). Use it rather than regenerating. Then:
```bash
sbatch run_array.sh      # work-stealing tasks; 24 h walltime on gpu_short/gpu_long
sbatch run_fix.sh        # optional: dedicated tasks for the overlay trials (fix_ids.txt)
```
Each trial writes `results/<id>.json`; `status == "ok"` means it reproduced its logged score.
`rescore_task.py --only id1,id2 --check` runs a pilot and exits nonzero on any mismatch.

### 12.8 Figures
```bash
cd runs-wave/summary_figures
python plot_progress_grid.py     # Figure 1 (needs rescore-grid/results and the run CSVs)
python make_summary_figures.py   # Figure 2 (needs varpart-v20/ and varpart-v21/ results)
```

---

## 13. Gotchas

1. **Snapshots ≠ scored code** in several runs (§10). Always verify by reproducing the logged
   score before trusting a re-run.
2. **NumPy NEP 50** silently changes dtypes in pre-Jun-5 encoders; use the 1.26 overlay for
   them. There is no legacy-promotion switch in NumPy 2.2.
3. **`interact_set` default trap** in LEAN-v17…v20 `build_embedder()` (§7).
4. **GPU type**: A100 only for features (TF32 convolutions).
5. **Memory**: the in-memory harness holds the whole 1.75M × P design matrix (v20: ~108 GB
   float32) — request ≥ 192 GB or use the streamed scripts. 16 GB V100s OOM on float64 eigh at
   P ≈ 13–15k without expandable segments.
6. **SLURM on this cluster**: some a100 nodes expose a GPU that SLURM lists as free but that is
   always busy (`CUDA-capable device(s) is/are busy or unavailable`); self-requeue loops back to
   the same node, so add such nodes to `ExcNodeList`. `scontrol update ... ExcNodeList=` replaces
   the list — write back the union. A100 QOS caps (~8 GPUs per partition per user; memory caps
   912 GB / 1500 GB) are often the binding constraint. `gpu_dev` has a 4 h limit.
7. **Scratch quota**: `/path/to/scratch` was out of quota during the runs; outputs were
   written under `/path/to/ecog_data/...`.
8. **Fixed test set is optimistic** by ~0.015 r (§2).
9. **Metric name collision**: `exceed-*` CSV `test_corr` values are grid-only; earlier runs are
   all-166.
10. **`wave_encoders_lib/LEAN-v21.py` … `v23.py` in `wave-trim`** are unrelated Jun 3 files.

---

## 14. Key numbers in one place

| Quantity | Value |
|---|---|
| Whisper (4 delays, same ridge), all / grid / depth | 0.0817 / 0.0933 / 0.0505 |
| LEAN-v20, all / grid / depth | 0.0808 / 0.0904 / 0.0566 |
| LEAN-v20 + Whisper, all / grid | 0.0920 / 0.1038 |
| LEAN-v16 grid (rescored) → v20 grid | 0.0852 → 0.0904 |
| Best single hand-built trial (grid) | 0.0909 (exceed-v20-deep, LEAN-v35-AUTOLAGZ) |
| Best clean-slate paradigm (grid) | C phonological 0.0808; H reservoir 0.0803; D scattering 0.0802 |
| v20 full-model R² (grid) | 1.894% (mean r·\|r\|) |
| Shared variance (v20 families, grid) | 88.1% |
| v20 degenerate columns | 22 constant + 489 duplicates (of 3,852) |
| v21 | 3,392 dims; grid 0.0902; Δ vs v20 −0.0003 |
| Grid rescore | 510 run, 504 reproduce (max \|Δ\| 0.00013); 811 / 902 attempts on the grid axis |
| LEAN vs Whisper gap on random splits | +0.0064 ± 0.002 (fixed set +0.0061) |

---

## 15. File map of this package

```
LEAN_REPLICATION_REPORT.md        this document
ENVIRONMENT.txt, ENVIRONMENT_pip_freeze.txt
pyproject.toml, uv.lock           uv environment (repo root)
readme.md, CLAUDE.md, launch.md, lit_review_unexplored.md   original repo docs
main_project/extract_whisper_features.py   Whisper-large-v3 feature extraction (main project)
evolve-wave/                      waveform harness: src/, setup_run_wave.py, extract_wave_aligned.py,
                                  program.md (loop rules), speech_rich_test_chunks.json, boot_ci.py,
                                  validate_gpu_ridge.py, extract_ling_features.py (code only)
evolve-ecog/                      melspec precursor harness; whisper_fir.py, whisper_baseline.py
autoresearch_loop/                SLURM launchers + agent prompts for every run (logs excluded)
analyze/                          June progress plots (waveform_progress.png, branch_progress.png) + scripts
random_splits/                    4 random test splits used for the representativeness check
runs-wave/<run>/                  every run folder: results/overall_results.csv, wave_encoders_lib/
                                  (snapshots), wave_encoder.py (final state), program.md, run.log, src symlink
runs-wave/wave-trim/wave_encoders_lib/LEAN-v16..v20.py, LEAN-v20_FEATURES.md   promoted encoders + catalog
runs-wave/lean-v17-clean/         original LEAN-v17 harness (enc_orig.py), cleaned version, verification
runs-wave/lean-v19-clean/         LEAN-v19 cleanup + bit-identity check
runs-wave/lean-whisper/           LEAN vs Whisper diagnostics (diag_*.py/.sh/.out, per-electrode npz, glass brains)
runs-wave/varpart-v20/, varpart-v21/   variance partitioning code, fits/, RESULTS.md, CSVs (stats/ excluded)
runs-wave/lean-v21/               LEAN-v21.py, make_v21.py, verify_v21.py, degenerate scans
runs-wave/rescore-grid/           grid rescore: trials.json, src_patched/, rescore_task.py, recover_interp.py,
                                  run_*.sh, results/ (510 json + 504 r arrays), logs/, jobs.txt (np126 excluded)
runs-wave/summary_figures/        Figure 1 and 2 (png/pdf), scripts, captions, plotted data
```
Excluded on purpose: raw/derived neural and audio data, Whisper features, transcript-derived
features (privacy); `varpart-*/stats/` (12.5 GB, regenerated by phase 1); the NumPy overlay
(one command); agent session logs; the `openaudio*` trained-model follow-on (13 GB, separate study).

---

## 16. SLURM job log

| Job | What |
|---|---|
| 23628821 | `wave-slurm` exploration loop (Jun 3) |
| 23846181–23846184; 23848997, 23849010, 23849000, 23849011, 23849001 | reset workers A–D; E, F, G, H, I (Jun 4; B was cancelled early) |
| 23851856 | graft-lean worker |
| 23865792 | lean-interp worker |
| 23959669, 23978741, 23978742 | exceed-v18, exceed-v20-context, exceed-v20-deep |
| 23988279, 23989056, 23996756 | canonical LEAN-v17 vs Whisper, LEAN-v20, LEAN-v20 + Whisper diagnostics |
| 29206095 → 29206096 → 29206097 | varpart-v20 phase 1 → 2 → 3 |
| 29207508 / 29208175 | v21 verification (second run with per-block breakdown) |
| 29207509, 29207510, 29207511 (+ 29208176 rerun), 29207512 (+ 29208177) | varpart-v21 phases, scan |
| 29208846 | rescore pilot (gate) |
| 29208847 | rescore array (cancelled automatically when the pilot gate failed) |
| 29210855 | rescore array (work-stealing; idle pending tasks cancelled once its pool was exhausted) |
| 29222968 | rescore fixer array (NumPy 1.26 overlay trials + v1 flag fixes) |

---

## 17. Limitations and open directions

- Single subject, single fixed speech-rich test session; absolute numbers optimistic.
- Snapshot provenance limits the rescore to 811 / 902 attempts; the missing ones are mostly from
  the first exploration run and do not change the running best.
- Untested directions: widen the 2-D Gabor time window (σₜ ≈ one period) so the 4–12 Hz
  quadrature/STMF filters have genuine rate tuning (would change ~1,000 columns; "v21b");
  prune the blocks with negative unique variance (formants, spectral shape, voicing gates,
  pitch/voicing dynamics); evaluate on random splits and other subjects; electrode-level
  (glass-brain) maps of unique variance from `perelec_R2.npz` and of the per-trial grid scores
  in `rescore-grid/results/*_r.npy`.
