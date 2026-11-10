# LEAN follow-ups: held-out test sets, five more subjects, and two reviewer controls

*In this repository, part 1 (`1_heldout_and_multisubject/`) is `experiments/ecog/LEAN_newtest_package/`; parts 2 and 3 are in this folder.*

LEAN is the hand-built auditory feature space that an autoresearch search (`claude-opus-4-7`, about 600
trials) produced for predicting ECoG high-gamma from sound in patient patient1. LEAN-v20 is the version analyzed
in the paper; LEAN-v35-AUTOLAGZ is the last model the search kept. This package holds three follow-ups. Each
folder has its own detailed report; this page summarizes them.

| Folder | Question | Start with |
|---|---|---|
| `1_heldout_and_multisubject/` | Do the search's scores, ranking and variance partitioning hold on data it never saw: a new patient1 test set and five more subjects? | `NEWTEST_REPORT.md` |
| `2_text_bottleneck/` | Is the final model interpretable: can a fresh session rebuild it from a 200-word description (the paper's Fig. 2g protocol)? | `RESULTS.md`, `PROTOCOL.md` |
| `3_textbook_baseline/` | Could an expert have built these features from the literature alone? | `RESULTS.md`, `PROTOCOL.md` |

The original June search (code, logs and the Nov 5 analyses) is documented separately in
`LEAN_replication_package.zip`.

## Shared evaluation

Every model in all three parts is fit and scored exactly like every LEAN trial: 500 training chunks (about
16 h per subject), 4 FIR delays, 100 ms lag, per-chunk z-scoring, per-electrode ridge with bootstrapped
regularization. Metric: per-chunk Pearson r between predicted and actual high-gamma, averaged over chunks and
over the subject's surface-grid electrodes. Confidence intervals: paired bootstrap over test chunks
(B = 2000).

Each subject is tested on a whole recording session that was never used for training or for selection,
fixed before anything was scored (`1_heldout_and_multisubject/protocol/`, `multisubject/code/make_splits.py`):

| Subject | Held-out session | Test chunks | Test hours | Grid electrodes (all) |
|---|---|---|---|---|
| patient1 | 450a36e4 | 493 (317 primary) | 15.98 (10.29) | 125 (166) |
| patient2 | 236d910e | 318 | 10.33 | 127 (235) |
| patient3 | e3d31b95 | 416 | 13.49 | 63 (116) |
| patient4 | 2eeb170d | 107 | 3.44 | 96 (155) |
| patient5 | 5e416143 | 275 | 8.93 | 124 (188) |
| patient6 | 5d2842ef | 222 | 7.21 | 127 (167) |

patient1's 493 test chunks are every usable chunk of its session except the 23 the search was scored on; the
primary 317 are the speech-rich ones, excluding the neighbors of the old test block. The tables below report
patient1 on all 493 (primary-set results are in `NEWTEST_REPORT.md`). Each other subject trains on 500 chunks
from its own remaining sessions.

## 1. Held-out rerun and five more subjects (`1_heldout_and_multisubject/`)

The search scored and selected every trial on one fixed 23-chunk patient1 test set. Here every trial on the
Figure 1 main line (593 of 596 reproduced their logged scores and were rerun), the LEAN-v20 family variance
partitioning and the LEAN-v20 + Whisper ("Both") model are rerun on patient1's held-out session and on five more
subjects. patient1's training data and fits are unchanged.

| Subject | v16 | v20 | Whisper | Both | Unique LEAN [95% CI] | Unique Whisper | Shared | r(trial scores, patient1 old test) |
|---|---|---|---|---|---|---|---|---|
| patient1 | 0.0649 | 0.0690 | 0.0752 | 0.0810 | 5.8% [4.9, 6.8] | 25.3% | 68.9% | 0.98 |
| patient2 | 0.0661 | 0.0643 | 0.0776 | 0.0752 | −0.1% [−3.3, 3.3] | 26.2% | 74.0% | 0.80 |
| patient3 | 0.0592 | 0.0604 | 0.0710 | 0.0717 | 1.4% [0.2, 2.7] | 26.4% | 72.2% | 0.84 |
| patient4 | 0.0575 | 0.0578 | 0.0591 | 0.0604 | 5.8% [−0.8, 13.6] | 9.1% | 85.2% | 0.85 |
| patient5 | 0.0628 | 0.0637 | 0.0678 | 0.0778 | 17.2% [12.6, 22.6] | 34.6% | 48.2% | 0.94 |
| patient6 | 0.0606 | 0.0614 | 0.0649 | 0.0704 | 11.8% [7.9, 16.3] | 23.6% | 64.6% | 0.89 |

Unique and shared are percentages of the Both model's variance explained (R² = r·|r|).

- **On patient1 the ranking holds.** Old- and new-test trial scores correlate at r = 0.981, and the model the
  search kept is within 0.0002 of the best trial on the new test. About 80% of the v16 → v20 gain transfers
  (+0.0052 old, +0.0042 new, primary set).
- **The gap to Whisper widens on held-out data**, from −0.003 to −0.008 on patient1, by the same amount for v16
  as for v20: the old podcast block favored hand-built features in general, not the late tuning.
- **The front end transfers to new subjects; the late search does not.** On every subject most of the gain
  comes in the first ~60 trials (cochleagram, compression and adaptation, multiple windows, modulation
  filters). v16 → v20 adds +0.0041 on patient1 but only −0.0018 to +0.0012 elsewhere.
- **Whisper beats LEAN-v20 on every subject, but LEAN still adds information Whisper lacks** on four of six
  (unique-LEAN CI above zero): most on patient5 and patient6.
- **Modulation filters are the one LEAN family with a robust unique contribution** (2.0–7.2% on every
  subject). The "cochleagram + slow context + lookahead" block, which already shrank on patient1's new test, has
  negative unique variance on four of the five new subjects.

Details: `NEWTEST_REPORT.md` §1–11 (patient1 rerun), §12 (Both, patient1), §13 (five more subjects);
`multisubject/summary/SUMMARY.md`.

## 2. Text bottleneck (`2_text_bottleneck/`)

Following the paper's protocol with Opus 5.5 (`claude-opus-5-5`):
1. **Describe.** A fresh session saw only the final model's code and wrote a description of at most 200 words.
   The code (`shared/model.py`) is LEAN-v20 with comments, run-history strings (old scores) and harness glue
   removed mechanically (`make_clean_model.py`); it produces bit-identical features (3,852 dims).
2. **Rebuild** (5 sessions): only the task and the description (197 words, `frozen/DESCRIPTION.md`, quoted below).
3. **Control** (5 sessions): the identical task with no description. Control k got the same example clips as
   rebuild k.

Audits mirroring the paper's (i)–(iii): no rebuilt or control model fits anything to data (code audit); each
produces bit-identical outputs with different example data present (`INVARIANCE.md`); no session read files
outside its sandbox (`AUDIT.md`; the automated flags were all false positives, reviewed there).

| Subject | Original | Rebuild mean | Control mean | Rebuild − original [95% CI] | Rebuild − control [95% CI] |
|---|---|---|---|---|---|
| patient1 | 0.0688 | 0.0659 | 0.0608 | −0.0029 [−0.0036, −0.0022] | +0.0051 [+0.0041, +0.0061] |
| patient2 | 0.0643 | 0.0683 | 0.0622 | +0.0040 [+0.0029, +0.0050] | +0.0061 [+0.0049, +0.0072] |
| patient3 | 0.0602 | 0.0597 | 0.0605 | −0.0005 [−0.0015, +0.0005] | −0.0007 [−0.0025, +0.0012] |
| patient4 | 0.0578 | 0.0563 | 0.0521 | −0.0015 [−0.0028, −0.0002] | +0.0043 [+0.0020, +0.0064] |
| patient5 | 0.0637 | 0.0630 | 0.0604 | −0.0007 [−0.0016, +0.0002] | +0.0026 [+0.0013, +0.0039] |
| patient6 | 0.0615 | 0.0594 | 0.0576 | −0.0021 [−0.0030, −0.0012] | +0.0018 [+0.0004, +0.0033] |

| Closeness to the original (patient1) | Rebuilds | Controls |
|---|---|---|
| Correlation of held-out predictions with the original's | 0.633 (0.609–0.658) | 0.503 (0.484–0.513) |
| Dimensions relative to the original, geometric mean | 0.995× (3,769–3,852 dims) | 0.216× (382–1,478 dims) |

- A 197-word description is enough to rebuild LEAN-v20 closely: rebuilds keep its size, track its held-out
  predictions much more closely than controls, and lose 0–0.003 r (on patient2 they score higher than the original).
- Rebuilds beat the controls on five of six subjects; on patient3 all three are tied (about 0.060).
- The original, scored through the same wrapper, reproduces its stored scores within 0.0002 on every subject.

**The description the rebuilders received** (`frozen/DESCRIPTION.md`):

> # 3,852 dims, win=4096
> 
> LP_τ = causal one-pole lowpass over frames (α = 1/(1+τ)); "dyn" = x − LP_τ, τ ∈ {4.5, 15, 60}. All concatenated.
> 
> **Cochleagrams.** Hann power spectra, centred 1024/2048/4096-sample sub-windows, weighted by 20 fourth-order gammatone power responses (ERB-spaced, 50–8000 Hz) → log. Add E − LP_τ for τ ∈ {1.5, 4.5, 15, 60, 150}; tonic LP_60 and LP_150 of log and cube-root energies, acausal LP_60; past and future differences over 3/6/12 frames, plus rectified halves.
> 
> **Spectrotemporal modulation.** 2-D Gabor quadrature energy on the 4096 cochleagram (17 rate/scale pairs: 0–12 Hz, 0–±0.5 cycles/channel). On a 40-channel cochleagram: rates 0.5–12 Hz × scales {0, ±0.25, ±0.5}, cosine, sine, magnitude averaged over frequency thirds; dyn, acausal residuals cross-rate energy products.
> 
> **Descriptors** (4096 spectrum; each with dyn): 8-harmonic log-comb pitch salience (11 F0s, 75–450 Hz), best F0/strength, ACF HNR, cepstral peak prominence, spectral moments/flatness/rolloff, octave and channel-group spectral contrast, softmax formants, spectral flux, voicing-gated cochleagram. Plus 2/4/6/10 Hz temporal-Gabor envelopes of these scalars.
> 
> **Interactions.** Key scalars, their dyn, pairwise products (phonetic band contrasts × pitch/loudness/flux/formants), z-scored within the segment and expanded as [z]₊, [−z]₊, z³.
> 
> **Within-window scattering.** Morlet scattering (Q=4 then Q=1) on 9 channels plus instantaneous frequency/coherence.

## 3. Textbook baseline (`3_textbook_baseline/`)

Five isolated agents ran `claude-opus-4-7`, the model that ran the search, so prior knowledge is matched.
Each built, in one shot, "the feature space a knowledgeable expert would build from standard,
well-established acoustic features" (`shared/TASK.md`). Initializations differed only by a random session
tag and the three example clips. All five chose plain numpy/scipy (no packages) and built 301–856-dimensional
textbook designs: mel/ERB spectrograms with deltas, MFCCs, spectral shape, pitch and voicing, chroma,
loudness and simple modulation filterbanks (`frozen/agent_k/FEATURES.md`). The audit is clean.

| Subject | Textbook mean (best of 5) | Search, trial 52 | LEAN-v20 | LEAN-v20 − textbook [95% CI] | Whisper |
|---|---|---|---|---|---|
| patient1 | 0.0549 (0.0562) | 0.0626 | 0.0690 | +0.0140 [+0.0127, +0.0153] | 0.0752 |
| patient2 | 0.0598 (0.0609) | 0.0660 | 0.0643 | +0.0045 [+0.0029, +0.0062] | 0.0776 |
| patient3 | 0.0551 (0.0564) | 0.0605 | 0.0604 | +0.0052 [+0.0034, +0.0070] | 0.0710 |
| patient4 | 0.0507 (0.0539) | 0.0580 | 0.0578 | +0.0071 [+0.0048, +0.0093] | 0.0591 |
| patient5 | 0.0559 (0.0584) | 0.0612 | 0.0637 | +0.0077 [+0.0062, +0.0092] | 0.0678 |
| patient6 | 0.0513 (0.0546) | 0.0600 | 0.0614 | +0.0102 [+0.0085, +0.0117] | 0.0649 |

- The search beats the textbook baseline on every subject, with every CI above zero. By trial 52 (of 593) it
  already beats even the best of the five agents everywhere (+0.003 to +0.0065).
- The model the search kept stays above the textbook mean from about trial 37–41 on every subject (from
  trial 6 on patient6).
- What the textbook designs lack is what the search found early: multi-window cochleagrams with divisive
  adaptation at several time constants, and 2-D spectro-temporal modulation filters.

## Isolation of the agent sessions (parts 2 and 3)

Every agent session ran inside a bubblewrap sandbox (`3_textbook_baseline/run_agent.sh`,
`2_text_bottleneck/run_session.sh`):
- It sees only its own `/work` (task, a smoke test, three audio-only example clips), an empty home with the
  Claude credentials file, read-only system directories, and Python (numpy, scipy, torch; any PyPI package
  installable with `pip install` into `/work/pkgs`). The project filesystem (`/gpfs`: search code, logs,
  results, notes, memory) is not mounted. SLURM is absent; ssh, scp, sftp, rsync, wget, curl, ncat and
  telnet are masked.
- Claude Code runs with only the Bash, Read, Write and Edit tools (no web tools, no subagents, no messaging
  other sessions), no MCP servers (no account connectors), skills disabled.
- One shot: no performance feedback. The only check available is a smoke test (shape, finite values, a
  dimension cap, speed) on the example audio. Nobody intervened during a run.
- Afterwards: deliverables frozen with sha256 hashes; every tool call audited (paths outside the sandbox,
  network use other than `pip install`, credential access); code audited for learning, pretrained models or
  network access (`AUDIT.md`). The task texts name no features, parameters or results.

## Files

| Path | Contents |
|---|---|
| `1_heldout_and_multisubject/NEWTEST_REPORT.md`, `README.md` | Full report (methods, validation, results, reproduction, file map) |
| `1_heldout_and_multisubject/protocol/` | Test-set rule (fixed before scoring), frozen train and test lists, hashes |
| `1_heldout_and_multisubject/code_recovery/`, `rerun/`, `extraction/` | Recovered code for every trial, the scoring harness, segment extraction |
| `1_heldout_and_multisubject/results/`, `varpart/`, `figures/` | patient1 per-trial scores, family variance partitioning, Figures 1–2 on the new test |
| `1_heldout_and_multisubject/both_lean_whisper/` | LEAN-v20 + Whisper and the LEAN-vs-Whisper partitioning (patient1) |
| `1_heldout_and_multisubject/multisubject/` | Code, frozen splits, per-subject trial results, partitionings, summary tables and figures (five more subjects) |
| `2_text_bottleneck/PROTOCOL.md`, `shared/` | Protocol; describer, rebuild and control task texts; the cleaned model code given to the describer |
| `2_text_bottleneck/make_clean_model.py` | How `shared/model.py` was derived from LEAN-v20 (and its bit-identity check) |
| `2_text_bottleneck/run_session.sh`, `launch_rebuild.sh` | Sandbox launcher; the step that enforced the 200-word limit and started the rebuilds |
| `2_text_bottleneck/frozen/` | The description, the original-model adapter, all rebuild and control feature spaces, hashes |
| `2_text_bottleneck/logs/` | Transcripts of the describer, rebuild and control sessions |
| `2_text_bottleneck/AUDIT.md`, `INVARIANCE.md`, `freeze_audit.py`, `invariance_check.py` | Audits (i)–(iii) |
| `2_text_bottleneck/scoring/` | Scoring wrapper, driver, harness copy (`src_bn/`); per-run results and per-chunk r |
| `2_text_bottleneck/bn_compare.py`, `RESULTS.md`, `results.csv`, `closeness.json`, `fig_*` | Analysis and results |
| `3_textbook_baseline/PROTOCOL.md`, `shared/TASK.md`, `shared/check.py` | Protocol (fixed before the run), the agents' task and smoke test |
| `3_textbook_baseline/run_agent.sh`, `sandbox_pip_shim/` | Sandbox launcher; the pip shim and version pins used inside it |
| `3_textbook_baseline/frozen/` | The five frozen feature spaces (`feature_space.py`, `FEATURES.md`) and sha256 hashes |
| `3_textbook_baseline/logs/` | Full stream-json transcripts of the five agent sessions |
| `3_textbook_baseline/AUDIT.md`, `freeze_audit.py` | Audit report and script |
| `3_textbook_baseline/scoring/` | Scoring wrapper and driver; per-run results (`*.json`) and per-chunk × per-electrode r (`*_r.npy`) |
| `3_textbook_baseline/compare.py`, `RESULTS.md`, `results.csv`, `fig_*` | Analysis and results |

Not included: patient data (audio, high-gamma), the aligned audio segments and the agents' example clips
(patient-room audio); the largest per-chunk arrays (listed in `NEWTEST_REPORT.md` §11 and §13) and the saved
held-out predictions behind the bottleneck closeness measure (4.2 GB); the sandbox's `uv` binary. All of these
remain on disk. Scripts use the cluster's absolute paths.
