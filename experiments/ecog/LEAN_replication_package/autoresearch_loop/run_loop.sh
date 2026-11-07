#!/bin/bash
#SBATCH --job-name=ecog_autoloop
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/autoloop_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/autoloop_%A.err
#SBATCH --partition=gpu_short,gpu_long,gpu4_short,gpu4_medium,gpu8_short,gpu8_medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Autonomous autoresearch loop: headless Claude hand-writes (NO training) the
# weights of a small melspec -> ECoG high-gamma encoder for patient1, iterating to
# beat the linear ridge baseline. Everything for THIS job lives under
# autoresearch_loop/ (script + logs); the experiment run folder is created under
# runs-ecog/<tag>.
#
# Usage:
#   sbatch autoresearch_loop/run_loop.sh            # default model: opus
#   sbatch autoresearch_loop/run_loop.sh sonnet     # override model

set -euo pipefail

MODEL="${1:-opus}"
NUM_TRAIN=500
NUM_TEST=100

REPO="/path/to/ecog_data/circuit-building-autoresearch"
EVOLVE="$REPO/evolve-ecog"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"

mkdir -p "$LOOPDIR/logs"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"

JOBID="${SLURM_JOB_ID:-manual}"
TAG="ecog-slurm-${JOBID}"
RUN_DIR="$REPO/runs-ecog/$TAG"
CLAUDE_LOG="$LOOPDIR/logs/claude_${JOBID}.jsonl"

# This cluster's compute nodes have a non-writable /tmp (the SLURM prolog warns
# "TMPDIR [/tmp] is not writeable"). Claude Code needs a writable temp dir for its
# shell snapshots / runtime or it hangs silently at startup, so point TMPDIR at a
# job-specific writable dir on the cluster file system.
export TMPDIR="$LOOPDIR/tmp/${JOBID}"
mkdir -p "$TMPDIR"

echo "=== ecog autoloop ==="
echo "host:   $(hostname)"
echo "date:   $(date)"
echo "model:  $MODEL"
echo "tag:    $TAG"
echo "run:    $RUN_DIR"
nvidia-smi -L || true

# 1) Create + seed the run folder with the deterministic ridge baseline row.
cd "$EVOLVE"
python setup_run.py "$TAG" --num-train "$NUM_TRAIN" --num-test "$NUM_TEST" --device cuda

# 2) Launch headless Claude to drive the experiment loop from inside the run folder.
cd "$RUN_DIR"

PROMPT="You are running an autonomous research loop on a GPU compute node (cwd: $RUN_DIR). \
This run folder already exists and is seeded with the 'Ridge-melspec-baseline' row in \
results/overall_results.csv -- do NOT run setup_run.py. \
Read program.md in this folder and follow its 'Experimentation' and 'The experiment loop' \
sections exactly. Repeatedly: edit interpretable_transformer.py with ONE hand-designed, \
human-understandable weight idea (closed-form weights only -- absolutely NO training, \
optimizer, gradient, or fitting of the transformer), set a unique model_shorthand_name and \
model_description, then run: \
'python interpretable_transformer.py --num-train $NUM_TRAIN --num-test $NUM_TEST > run.log 2>&1', \
then read 'tail -n 20 run.log', save a snapshot to interpretable_transformers_lib/, and continue. \
Each iteration's ridge fit takes a few minutes; that is expected. NEVER stop to ask whether to \
continue; do at least 30 iterations. Goal: maximize test_corr above the ridge baseline with \
interpretable audio features (mel filter banks, onset/edge detectors, spectro-temporal \
modulation filters, short causal temporal integration)."

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
