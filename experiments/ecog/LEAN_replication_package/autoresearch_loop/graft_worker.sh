#!/bin/bash
#SBATCH --job-name=graft_lean
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/graft_%x_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/graft_%x_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=144G
#SBATCH --cpus-per-task=8

# DIRECTED GRAFT worker. Seeded with the STRONG LEAN-v16 baseline (NOT a clean
# slate). Walks an ordered list of pre-identified NONLINEAR candidate blocks
# (harvested from the reset workers), grafting them onto LEAN one at a time as
# successive versions, greedily keeping what helps and reverting what doesn't.
#
# Usage:  sbatch autoresearch_loop/graft_worker.sh

set -uo pipefail
MODEL="${1:-opus}"
REPO="/path/to/ecog_data/circuit-building-autoresearch"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
JOBID="${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOOPDIR/tmp/${JOBID}"; mkdir -p "$TMPDIR"
export ECOG_TEST_CHUNKS_FILE="$REPO/evolve-wave/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"
export ECOG_RIDGE_GPU="1"
mkdir -p "$LOOPDIR/logs"

RUN_DIR="$REPO/runs-wave/graft-lean"
CLAUDE_LOG="$LOOPDIR/logs/claude_graft_${JOBID}.jsonl"
CACHE="/path/to/scratch/feat_cache_graft_lean"; mkdir -p "$CACHE"

PROMPT="You are running a DIRECTED feature-grafting experiment on a GPU node (cwd: $RUN_DIR). \
GOAL: hand-design a CLOSED-FORM auditory front-end that maps the pre-aligned speech WAVEFORM to \
patient patient1's ECoG high-gamma. A linear ridge readout (fit by the harness -- the ONLY data-fit step) \
decodes your features into 166 electrodes; score = per-chunk per-electrode test correlation. \
\
STARTING POINT -- THIS IS NOT A CLEAN SLATE: wave_encoder.py currently holds LEAN-v16, a STRONG, \
heavily-tuned baseline that already scores about test_corr=0.0760 under this harness. It already \
contains: multi-window gammatone log-cochleagram + multi-tau adaptation deltas, harmonic-comb pitch + \
log-F0 + pitch deltas, spectral moments/flatness/rolloff, HNR, data-free phonetic templates, spectral \
flux, a Gabor temporal modulation filterbank, a Chi-Shamma 2D rate-scale STMF bank, 2-formant tracking \
+ contrastive formants, cross-band comodulation + DoG lateral inhibition, per-channel modulation depth, \
slow tonic causal+acausal lowpass cochleagrams, and future+past per-band deltas. DO NOT re-add or \
duplicate any of those -- they are already present, and a LINEAR function of the existing bands adds \
NOTHING because the ridge already spans it (this was proven: an earlier graft of linear manner/place/ \
peakRate detectors gave a bit-identical NULL). \
\
YOUR TASK: graft a fixed, ORDERED list of pre-identified NONLINEAR candidate blocks onto LEAN, ONE AT \
A TIME, as successive versions. Each candidate was harvested from a sibling experiment and chosen \
specifically because it is NONLINEAR and ABSENT from LEAN (a quantity the linear readout cannot \
reconstruct from the existing bands). For each candidate: ADD it on top of the BEST encoder so far, \
eval, then KEEP or REVERT it by the acceptance rule, then move to the next candidate on top of the \
retained best. This is greedy forward selection. \
\
THE ORDERED CANDIDATE LIST (port the actual computation by READING the cited source files): \
  CANDIDATE 1 = PHASE / TEMPORAL-FINE-STRUCTURE block (most orthogonal to LEAN, ~290 dim). Source: \
    $REPO/runs-wave/reset-I/wave_encoder.py (and the highest TFS*-vNN snapshot in reset-I/ \
    wave_encoders_lib/). Port the '_tfs_block' pieces: instantaneous frequency + instantaneous \
    bandwidth (log_if / if_dev_norm / if_std), FM-glide up/down slope detectors (fm_up / fm_dn), \
    inter-band phase-locking value (plv1 / plv2), and phase-amplitude coupling (pac). These come from \
    the analytic signal (Hilbert via FFT) of per-band signals -- phase info LEAN entirely lacks. \
  CANDIDATE 2 = ECHO-STATE RESERVOIR state (~highest-upside, but SHRINK IT). Source: \
    $REPO/runs-wave/reset-H/wave_encoders_lib/RES-v09.py. Port the tanh+ReLU BIDIRECTIONAL fixed- \
    random reservoir (state x[t] = (1-leak)x[t-1] + leak*phi(W_res@x[t-1] + W_in@u[t]); random fixed \
    weights are LEGAL, not a data-fit), driven by z-scored [log_e, cube_e, flux] at a few lags. CRUCIAL: \
    SHRINK to about N=300 per family (4 families: tanh-fwd, tanh-bwd, relu-fwd, relu-bwd) so the \
    reservoir adds about 1200 dims, NOT 2400 -- you must keep TOTAL raw feature dim under ~2600 (RAM is \
    capped at 144 GB). The recurrence is intrinsically sequential, so a per-frame time-scan is PERMITTED \
    here (vectorize across units, run on GPU). \
  CANDIDATE 3 = DIVISIVE-NORMALIZATION / LEVEL-DEPENDENT COMPRESSION (~250 dim). Source: \
    $REPO/runs-wave/reset-A/wave_encoder.py ('_periphery_fast'): DRNL level-dependent compressive gain \
    (gain that DROPS as level rises, e.g. (theta/(theta+env))^(1-c)), multi-threshold AN fiber paths, \
    saturating rate-level ihc/(ihc+half_sat), and audio-rate divisive normalization ihc/(eps+slow_rate). \
    Also see $REPO/runs-wave/reset-E/wave_encoder.py for divisively-normalized cochleagrams \
    log(e)-log(sigma+pool). These are genuine nonlinearities (ratios/divisions) LEAN lacks. \
  CANDIDATE 4 = SOUND-TEXTURE STATISTICS (~250 dim). Source: \
    $REPO/runs-wave/reset-G/wave_encoder.py: higher-order CAUSAL RUNNING moments (skew + kurtosis per \
    band), cross-band running correlations, C1 cross-channel + C2 cross-rate modulation correlations, \
    and per-frame spectral entropy / Gini sparsity. Nonlinear cross-products + 3rd/4th moments LEAN \
    lacks. Compute every statistic as a CAUSAL running quantity (leaky accumulators) -- never re-window. \
  CANDIDATE 5 = NONLINEAR PRODUCT TERMS (~100-220 dim). Source: \
    $REPO/runs-wave/reset-C/wave_encoders_lib/PHON-v4.py (phonetic BIGRAM transition detectors: lagged \
    products of two template-score series, ~36 dim) and the highest PHON-vNN snapshot (v17 co-occurrence \
    products, v29 cross-band quadratic). Explicit products-of-features the ridge cannot synthesize. \
\
ACCEPTANCE RULE (gain-per-dimension): the paired-bootstrap noise floor is about +/-0.0008, so a real \
gain needs roughly >= +0.0015. After grafting a candidate, compare test_corr to the current BEST: \
  - KEEP if test_corr improves by >= +0.0015 (clearly real), OR if the block is SMALL (<= ~150 dim) AND \
    non-harmful (>= best - 0.0003). \
  - REVERT (delete the block, restore the prior best) if it HURTS (< best - 0.0003) or is a bit- \
    identical / within-noise gain on a LARGE block (redundant -- not worth the dims). \
A candidate may need a couple of variants (e.g. different scaling/normalization, or shrink dims) before \
you judge it -- a badly-SCALED block can wreck per-voxel ridge conditioning (symptom: train_corr jumps \
but test_corr collapses) even when the feature is good, so z-score / standardize new blocks. Give each \
candidate up to ~3 attempts before reverting and moving on. \
\
NAMING / BOOKKEEPING: iteration 1 = run LEAN AS-IS unchanged to CONFIRM the ~0.0760 baseline under this \
harness (call it GRAFT-v0-baseline). Then GRAFT-v1-tfs, GRAFT-v2-reservoir, GRAFT-v3-divnorm, \
GRAFT-v4-texture, GRAFT-v5-products (append '-keep' or '-revert' as you decide). Each description MUST \
state: what block was added, its dim, the measured test_corr, the delta vs prior best, and your KEEP/ \
REVERT decision. On startup, READ results/overall_results.csv and wave_encoders_lib/ to see which \
candidates you have already processed and CONTINUE from there (do not restart from scratch). After the \
ordered single-graft pass, you MAY try stacking the retained winners together and lightly tuning them. \
\
HARD RULES (never break): closed-form weights/filters ONLY -- NO training/optimizer/gradient/backprop, \
and NO parameter estimated from data (no PCA/ICA/NMF/k-means/sparse-coding/pretrained weights). Fixed \
RANDOM reservoir weights are fine (not estimated from data). The ridge readout is the ONLY data-fit \
step. NEVER re-window or re-time the audio -- the alignment is fixed; operate on the windows given. Keep \
total raw feature dim under ~2600 (RAM cap). \
\
PERFORMANCE RULES: 80 GB A100, 24h job, 144 GB host RAM. Implement filterbanks/convolutions/Hilbert/2D \
modulation via FFT or cumsum -- no naive Python time-loops (the ONLY exception is the reservoir \
recurrence). Run each eval as a SINGLE FOREGROUND command and BLOCK until it returns, with a GENEROUS \
Bash timeout (e.g. timeout 1800000 ms = 30 min); NEVER use a timeout below ~900000 ms and NEVER kill a \
healthy eval; if it auto-backgrounds, WAIT for it. CACHE expensive reusable per-chunk components to \
$CACHE/ keyed by a hash of the exact computation code+params plus chunk id. \
\
EACH ITERATION: edit wave_encoder.py (add or revert ONE candidate block), set a UNIQUE \
model_shorthand_name + a description per the bookkeeping rule, then run \
'python wave_encoder.py --num-train 500 --num-test 100 > run.log 2>&1' in the FOREGROUND, block until \
done, read 'tail -n 25 run.log', record your KEEP/REVERT decision, and snapshot the file to \
wave_encoders_lib/<name>.py. Do NOT background the eval, do NOT write any until/while/sleep/grep polling \
loop, and do NOT end your turn while an eval runs. Work through ALL FIVE candidates; do not stop early."

echo "=== GRAFT-LEAN worker on $(hostname) at $(date) -> $RUN_DIR ==="
nvidia-smi -L || true
cd "$RUN_DIR"

session=0; fastfails=0
while true; do
  session=$((session + 1)); echo "===== graft-lean session #$session start $(date) ====="
  start=$(date +%s)
  stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
    --model "$MODEL" --dangerously-skip-permissions --add-dir "$REPO" \
    --verbose --output-format stream-json 2>&1 | stdbuf -oL tee -a "$CLAUDE_LOG"
  rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))
  echo "===== graft-lean session #$session exited rc=$rc after ${dur}s at $(date) ====="
  if [ "$dur" -lt 120 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
  if [ "$fastfails" -ge 5 ]; then echo "5 consecutive <120s exits -- aborting"; break; fi
  sleep 10
done
echo "=== graft-lean worker ended at $(date) ==="
