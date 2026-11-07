#!/bin/bash
#SBATCH --job-name=ecog_waveloop
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/waveloop_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/waveloop_%A.err
#SBATCH --partition=gpu_short,gpu_long,gpu4_short,gpu4_medium,gpu8_short,gpu8_medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# Autonomous autoresearch loop, WAVEFORM track: headless Claude hand-designs (NO
# training) interpretable auditory front-ends mapping the raw audio to patient1 ECoG
# high-gamma, on the speech-rich test set (450a36e4_0190..0212, that recording
# excluded from training). Floor = Wave-melspec-floor; ceiling = Whisper-reference.
#
# Prereq: aligned segments pre-extracted to /path/to/scratch/patient1_wave_aligned
#         (evolve-wave/extract_wave_aligned.py --speech-rich).
#
# Usage:
#   sbatch autoresearch_loop/run_loop_wave.sh            # default model: opus
#   sbatch autoresearch_loop/run_loop_wave.sh sonnet

set -euo pipefail

MODEL="${1:-opus}"
NUM_TRAIN=500
NUM_TEST=100   # ignored in fixed-test mode

REPO="/path/to/ecog_data/circuit-building-autoresearch"
EVOLVE="$REPO/evolve-wave"
ECOG="$REPO/evolve-ecog"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"

mkdir -p "$LOOPDIR/logs"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"

JOBID="${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOOPDIR/tmp/${JOBID}"
mkdir -p "$TMPDIR"

# Speech-rich fixed test set + leakage guard (read by src/data.get_chunk_names).
export ECOG_TEST_CHUNKS_FILE="$EVOLVE/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"

TAG="wave-slurm-${JOBID}"
RUN_DIR="$REPO/runs-wave/$TAG"
CLAUDE_LOG="$LOOPDIR/logs/claude_wave_${JOBID}.jsonl"

echo "=== ecog waveform autoloop ==="
echo "host: $(hostname)   date: $(date)   model: $MODEL"
echo "tag:  $TAG"
echo "run:  $RUN_DIR"
nvidia-smi -L || true

# 1) Create run folder + seed the waveform floor baseline (wave -> log-mel -> ridge).
cd "$EVOLVE"
python setup_run_wave.py "$TAG" --num-train "$NUM_TRAIN" --num-test "$NUM_TEST" --device cuda

# 2) Seed the pretrained Whisper-reference ceiling on the same split (reuses evolve-ecog).
cd "$ECOG"
python whisper_baseline.py "$RUN_DIR" --num-train "$NUM_TRAIN" --num-test "$NUM_TEST" --device cuda

# 3) Launch headless Claude to drive the experiment loop.
cd "$RUN_DIR"

PROMPT="You are running an autonomous research loop on a GPU compute node (cwd: $RUN_DIR). \
GOAL: hand-design, with NO TRAINING, interpretable AUDITORY FRONT-ENDS that map the raw audio \
WAVEFORM to patient1 ECoG high-gamma -- modelling the algorithmic priors of early hearing (cochlear \
filtering, compression, adaptation, onset enhancement, modulation tuning, lateral inhibition, ...). \
The held-out TEST SET is a SPEECH-RICH podcast block (450a36e4_0190..0212; that whole recording is \
excluded from training). results/overall_results.csv is seeded with 'Wave-melspec-floor' (the linear \
floor) and 'Whisper-reference' (a pretrained-encoder ceiling, ~0.079). Do NOT run setup_run_wave.py \
or whisper_baseline.py. \
Read program.md and follow its loop, with these HARD rules: \
(1) Your front-end is handed PRE-ALIGNED audio windows (T, win); just map them to features (T, H). \
NEVER re-window/re-time the audio -- alignment is fixed by the harness. \
(2) You MAY use any deterministic DSP transform with hand-set/closed-form parameters (FFT, gammatone/ \
mel/roex filterbanks, rectification, log/power compression, fixed convolutions, adaptation with fixed \
time constants, modulation filterbanks, lateral inhibition, divisive normalization, hand-written weight \
tensors). You MAY NOT train or fit ANY front-end parameter to data (no gradient/optimizer/backprop, no \
PCA/ICA/NMF/k-means/sparse-coding). The ONLY data-fit step is the ridge readout, done by the harness. \
No pretrained weights. \
(3) Edit wave_encoder.py with ONE idea per iteration; set a unique model_shorthand_name + description. \
(4) Run the eval in the FOREGROUND and WAIT (each takes a few-to-~15 min; that is NORMAL): \
'python wave_encoder.py --num-train $NUM_TRAIN --num-test $NUM_TEST > run.log 2>&1'. Do NOT background \
it, do NOT poll with 'echo wait'/'sleep'/repeated checks. Then read 'tail -n 20 run.log'. \
(5) Keep feature dimensionality MODEST (ridge memory/time grow with H*ndelays); if slow/heavy, reduce \
filters/H. Snapshot each attempt to wave_encoders_lib/<name>.py. \
NEVER stop to ask whether to continue; do at least 30 iterations. Beat the floor and push toward the \
Whisper ceiling. The deliverable is INSIGHT: which fixed hearing priors high-gamma reflects, and by how much."

echo "=== launching claude ($MODEL) -> $CLAUDE_LOG ==="
echo "TMPDIR=$TMPDIR"
stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
  --model "$MODEL" \
  --dangerously-skip-permissions \
  --add-dir "$REPO" \
  --verbose \
  --output-format stream-json 2>&1 | stdbuf -oL tee "$CLAUDE_LOG"

echo "=== claude exited at $(date) ==="
cat "$RUN_DIR/results/overall_results.csv" || true
