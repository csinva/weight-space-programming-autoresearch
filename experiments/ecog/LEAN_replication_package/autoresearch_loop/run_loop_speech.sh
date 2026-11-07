#!/bin/bash
#SBATCH --job-name=ecog_speechloop
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/speechloop_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/speechloop_%A.err
#SBATCH --partition=gpu_short,gpu_long,gpu4_short,gpu4_medium,gpu8_short,gpu8_medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# Autonomous autoresearch loop, SPEECH-RICH variant: the held-out test set is the
# monkey-selfie-copyright podcast block (450a36e4_0190..0212); the whole 450a36e4
# recording is excluded from training. Headless Claude hand-writes (NO training)
# melspec -> ECoG high-gamma encoders to beat the linear ridge baseline, with a
# pretrained Whisper-reference ceiling for context.
#
# Usage:
#   sbatch autoresearch_loop/run_loop_speech.sh            # default model: opus
#   sbatch autoresearch_loop/run_loop_speech.sh sonnet

set -euo pipefail

MODEL="${1:-opus}"
NUM_TRAIN=500
NUM_TEST=100   # ignored in fixed-test mode, kept for the baseline call signature

REPO="/path/to/ecog_data/circuit-building-autoresearch"
EVOLVE="$REPO/evolve-ecog"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"

mkdir -p "$LOOPDIR/logs"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"

# Non-writable /tmp on compute nodes -> Claude needs a writable TMPDIR or it hangs.
JOBID="${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOOPDIR/tmp/${JOBID}"
mkdir -p "$TMPDIR"

# Speech-rich fixed test set + leakage guard (read by src/data.get_chunk_names).
export ECOG_TEST_CHUNKS_FILE="$EVOLVE/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"

TAG="ecog-speech-${JOBID}"
RUN_DIR="$REPO/runs-ecog/$TAG"
CLAUDE_LOG="$LOOPDIR/logs/claude_speech_${JOBID}.jsonl"

echo "=== ecog speech-rich autoloop ==="
echo "host:   $(hostname)"
echo "date:   $(date)"
echo "model:  $MODEL"
echo "tag:    $TAG"
echo "run:    $RUN_DIR"
echo "test:   $ECOG_TEST_CHUNKS_FILE (exclude prefix $ECOG_EXCLUDE_PREFIX)"
nvidia-smi -L || true

# 1) Create run folder + seed the linear ridge melspec baseline on the speech-rich split.
cd "$EVOLVE"
python setup_run.py "$TAG" --num-train "$NUM_TRAIN" --num-test "$NUM_TEST" --device cuda

# 2) Seed the pretrained Whisper-reference ceiling on the same split.
python whisper_baseline.py "$RUN_DIR" --num-train "$NUM_TRAIN" --num-test "$NUM_TEST" --device cuda

# 3) Launch headless Claude to drive the experiment loop.
cd "$RUN_DIR"

PROMPT="You are running an autonomous research loop on a GPU compute node (cwd: $RUN_DIR). \
The held-out TEST SET is a SPEECH-RICH block: a continuous podcast segment about the \
monkey-selfie copyright case (chunks 450a36e4_0190..0212). The whole 450a36e4 recording is \
excluded from training. This run folder already exists and is seeded in \
results/overall_results.csv with TWO reference rows: 'Ridge-melspec-baseline' (the linear \
floor to beat) and 'Whisper-reference' (a pretrained-encoder ceiling, ~0.079 here). Do NOT \
run setup_run.py or whisper_baseline.py. \
Read program.md and follow its 'Experimentation' and 'The experiment loop' sections, with \
these HARD rules: \
(1) Edit interpretable_transformer.py with ONE hand-designed, human-understandable weight \
idea per iteration (closed-form weights only -- absolutely NO training/optimizer/gradient/ \
fitting of the transformer). Set a unique model_shorthand_name + model_description. \
(2) Run the eval in the FOREGROUND and WAIT for it to finish: \
'python interpretable_transformer.py --num-train $NUM_TRAIN --num-test $NUM_TEST > run.log 2>&1'. \
Each eval takes ~10-15 minutes; that is NORMAL. Do NOT launch it in the background, do NOT \
poll with 'echo wait', 'sleep' loops, or repeated status checks -- just let the single \
foreground command block until it returns, then read 'tail -n 20 run.log'. \
(3) Keep feature dimensionality MODEST: ridge memory/time grow with d_model*ndelays and with \
stacked feature banks. Do NOT stack many large feature families at once; if an eval is very \
slow, reduce d_model or feature count. \
(4) After each eval, snapshot to interpretable_transformers_lib/<name>.py and continue. \
NEVER stop to ask whether to continue; do at least 30 iterations. \
Goal: maximize the per-chunk per-electrode test_corr on this speech-rich set, beating the \
ridge baseline and approaching the Whisper ceiling, with interpretable audio features \
(mel filter banks, onset/edge detectors, spectro-temporal modulation/STRF filters, short \
causal temporal integration)."

echo "=== launching claude ($MODEL) -> $CLAUDE_LOG ==="
echo "TMPDIR=$TMPDIR"
stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
  --model "$MODEL" \
  --dangerously-skip-permissions \
  --add-dir "$REPO" \
  --verbose \
  --output-format stream-json 2>&1 | stdbuf -oL tee "$CLAUDE_LOG"

echo "=== claude exited at $(date) ==="
echo "=== final results ==="
cat "$RUN_DIR/results/overall_results.csv" || true
