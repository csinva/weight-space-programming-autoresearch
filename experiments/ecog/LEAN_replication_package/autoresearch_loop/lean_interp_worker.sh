#!/bin/bash
#SBATCH --job-name=lean_interp
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/lean_interp_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/lean_interp_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# INTERPRETABLE-LEAN branch worker. Seeded with LEAN-v16. Goal: an INTERPRETABLE
# feature space that rivals Whisper (0.0821). Works (3) banded-ridge-style per-
# feature-group conditioning, then (2) interpretable nonlinear feature expansion.
# (1) linguistic/semantic features come LATER via a separate side-channel.

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

RUN_DIR="$REPO/runs-wave/lean-interp"
CLAUDE_LOG="$LOOPDIR/logs/claude_lean_interp_${JOBID}.jsonl"
CACHE="/path/to/scratch/feat_cache_lean_interp"; mkdir -p "$CACHE"

PROMPT="You are improving an auditory front-end on a GPU node (cwd: $RUN_DIR). GOAL: map the pre- \
aligned speech WAVEFORM to patient patient1's ECoG high-gamma; a linear ridge readout (fit by the harness, \
the ONLY data-fit step) decodes your features into 166 electrodes; score = per-chunk per-electrode test \
correlation. wave_encoder.py currently holds LEAN-v16, a strong hand-built baseline scoring about 0.0760. \
\
OVERARCHING OBJECTIVE: build an INTERPRETABLE feature space that rivals the Whisper learned-feature \
ceiling (about 0.0821). INTERPRETABILITY IS THE PRIORITY: every feature you add must be EASILY DEFINED / \
NAMEABLE as a meaningful acoustic quantity (e.g. 'onset rate in the 2-4 kHz band', 'pitch acceleration', \
'ratio of high-band to low-band energy'). Do NOT add opaque random projections, learned embeddings, or \
un-nameable features. A feature may be expensive to compute, but its DEFINITION must be simple. \
\
YOU WILL DO TWO STAGES IN ORDER. Confirm the baseline first: iteration 1 = run LEAN-v16 unchanged to \
re-establish ~0.0760 (call it INTERP-v0-baseline). Then: \
\
STAGE 3 (do FIRST) = BANDED-RIDGE-STYLE PER-GROUP CONDITIONING. The single global ridge lets large or \
high-variance feature families dominate and over-shrinks small informative ones. Without changing the \
harness (you may NOT edit anything under src/), CONDITION the feature space ENCODER-SIDE so each named \
feature GROUP contributes on comparable footing: e.g. per-GROUP L2 / unit-RMS normalization, per-group \
scaling by 1/sqrt(group_dim), or a small grid of per-group weights chosen by PRINCIPLE (not fit to the \
target). Group LEAN's features by their existing families (cochleagram, adaptation deltas, pitch, \
spectral shape, HNR, phonetic templates, flux, modulation/STMF, formants, comodulation, tonic, future/ \
past deltas). Try a few principled conditioning schemes (INTERP-v1a, v1b, ...) and keep the best. This \
is a closed-form per-group rescaling -- fully interpretable. \
\
STAGE 2 (do SECOND, on top of the best from stage 3) = INTERPRETABLE NONLINEAR EXPANSION. Add explicit, \
NAMED nonlinear INTERACTION terms between existing interpretable features -- products and ratios a linear \
ridge cannot synthesize but that remain fully interpretable: e.g. pitch-strength x onset-rate, high/low \
band-energy RATIOS, formant-dispersion x voicing, flux x spectral-tilt, envelope x its own short-lag. \
Prefer a curated set of meaningful pairwise products/ratios over a blind all-pairs blow-up. (Kernel / \
random-feature expansions are allowed ONLY as a last resort and only if you LABEL them clearly as the \
non-interpretable fallback -- avoid if an interpretable interaction set works.) Try several curated \
interaction sets (INTERP-v2a, ...) and keep what helps. \
\
ACCEPTANCE RULE (gain-per-dimension): paired-bootstrap noise is about +/-0.0008, real gain needs about \
>= +0.0015. After each change compare test_corr to the current best: KEEP if it improves by >= +0.0015 \
(real), OR if the block is SMALL (<= ~150 dim) and non-harmful (>= best - 0.0003); REVERT if it hurts or \
is a within-noise gain on a large block. z-score / standardize any new block so a badly-scaled feature \
does not wreck per-voxel ridge conditioning (symptom: train_corr jumps but test_corr collapses). \
\
NAMING: INTERP-v0-baseline, then INTERP-v1*-cond (stage 3), then INTERP-v2*-interact (stage 2). Each \
description MUST state what changed, its dim, the measured test_corr, the delta vs prior best, and your \
KEEP/REVERT decision. On startup READ results/overall_results.csv + wave_encoders_lib/ to see progress \
and CONTINUE. After finishing stages 3 then 2, STOP cleanly (do not invent unrelated features; stage 1 \
linguistic features are handled separately and are NOT your job). \
\
HARD RULES: closed-form ONLY -- no training/optimizer/gradient/backprop, no parameter estimated from data \
(no PCA/ICA/NMF/k-means/sparse-coding/pretrained weights); the ridge readout is the ONLY data-fit step. \
NEVER re-window or re-time the audio. You may NOT edit anything under src/ (the harness is fixed). \
\
PERFORMANCE: 80 GB A100, 192 GB RAM. FFT/cumsum, never naive Python time-loops. Run each eval as a SINGLE \
FOREGROUND command and BLOCK: 'python wave_encoder.py --num-train 500 --num-test 100 > run.log 2>&1' with \
a GENEROUS timeout (>= 900000 ms); never kill a healthy eval; if it auto-backgrounds, WAIT. Read \
'tail -n 25 run.log'. Snapshot each version to wave_encoders_lib/<name>.py. Do MANY iterations; do not \
end your turn while an eval runs."

echo "=== LEAN-INTERP worker on $(hostname) at $(date) -> $RUN_DIR ==="
nvidia-smi -L || true
cd "$RUN_DIR"

session=0; fastfails=0
while true; do
  session=$((session + 1)); echo "===== lean-interp session #$session start $(date) ====="
  start=$(date +%s)
  stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
    --model "$MODEL" --dangerously-skip-permissions --add-dir "$REPO" \
    --verbose --output-format stream-json 2>&1 | stdbuf -oL tee -a "$CLAUDE_LOG"
  rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))
  echo "===== lean-interp session #$session exited rc=$rc after ${dur}s at $(date) ====="
  if [ "$dur" -lt 120 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
  if [ "$fastfails" -ge 5 ]; then echo "5 consecutive <120s exits -- aborting"; break; fi
  sleep 10
done
echo "=== lean-interp worker ended at $(date) ==="
