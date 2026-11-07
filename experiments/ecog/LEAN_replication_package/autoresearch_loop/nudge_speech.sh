#!/bin/bash
#SBATCH --job-name=ecog_speechnudge
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/speechnudge_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/speechnudge_%A.err
#SBATCH --partition=gpu_short,gpu_long,gpu4_short,gpu4_medium,gpu8_short,gpu8_medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# CONTINUE the plateaued speech-rich melspec loop on its EXISTING run folder with a
# fresh-context agent and a steer toward structurally new ideas (the running-mean /
# adaptation / flux family stalled at ~0.0528). The prior results table + snapshots
# persist, so the agent sees all earlier attempts.

set -euo pipefail
MODEL="${1:-opus}"
NUM_TRAIN=500; NUM_TEST=100
REPO="/path/to/ecog_data/circuit-building-autoresearch"
EVOLVE="$REPO/evolve-ecog"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"
RUN_DIR="$REPO/runs-ecog/ecog-speech-23608983"   # the existing plateaued run

mkdir -p "$LOOPDIR/logs"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
JOBID="${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOOPDIR/tmp/${JOBID}"; mkdir -p "$TMPDIR"
export ECOG_TEST_CHUNKS_FILE="$EVOLVE/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"
CLAUDE_LOG="$LOOPDIR/logs/claude_speechnudge_${JOBID}.jsonl"

echo "=== speech-rich melspec NUDGE on $(hostname) at $(date) -> $RUN_DIR ==="
nvidia-smi -L || true
cd "$RUN_DIR"

PROMPT="You are CONTINUING an autonomous research loop on a GPU node (cwd: $RUN_DIR). results/overall_results.csv \
already holds ~22 prior attempts; the best is ~0.0528 and progress has PLATEAUED. First read \
results/overall_results.csv and the recent snapshots in interpretable_transformers_lib/ to see what has been tried. \
The earlier attempts converged on one family (subtract a causal running mean of the melspec, then half-wave \
rectify / threshold, plus spectral flux), and then mostly micro-tuned block_size and thresholds. DO NOT keep \
micro-tuning that family (no more block_size / threshold sweeps of MelRm*). \
Pursue STRUCTURALLY DIFFERENT, interpretable, hand-written (NO TRAINING) ideas, e.g.: \
spectro-temporal modulation / STRF-style rate-scale filterbanks over the (freq x time) melspec; 2D convolution \
patterns over the melspec (oriented spectro-temporal edges, frequency-sweep detectors); cross-frequency \
interactions / comodulation across bands; harmonic-template / pitch-salience features; longer-timescale temporal \
modulation channels; contrast/divisive normalization across frequency. Be creative and diverse. \
Rules (unchanged): closed-form weights only, NO training/optimizer/gradient/fitting; edit interpretable_transformer.py \
with ONE idea per iteration and a unique model_shorthand_name + description; run the eval in the FOREGROUND and \
WAIT (each ~5-15 min, that is NORMAL): 'python interpretable_transformer.py --num-train $NUM_TRAIN --num-test $NUM_TEST > run.log 2>&1' \
-- do NOT background it or poll with 'echo wait'/'sleep'; then read 'tail -n 20 run.log'; keep feature dimensionality \
MODEST; snapshot each attempt to interpretable_transformers_lib/<name>.py. NEVER stop to ask; do at least 20 more \
iterations. Goal: beat 0.0528 and push toward the Whisper ceiling (0.0794) with genuinely new structure."

echo "=== launching claude ($MODEL) -> $CLAUDE_LOG ==="
stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" --model "$MODEL" --dangerously-skip-permissions \
  --add-dir "$REPO" --verbose --output-format stream-json 2>&1 | stdbuf -oL tee "$CLAUDE_LOG"
echo "=== claude exited at $(date) ==="
