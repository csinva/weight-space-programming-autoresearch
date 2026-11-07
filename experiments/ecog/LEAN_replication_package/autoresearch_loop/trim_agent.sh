#!/bin/bash
#SBATCH --job-name=wave_trim
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/trim_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/trim_%A.err
#SBATCH --partition=gpu_short,gpu_long
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# Dedicated PRUNING agent: takes the current best waveform encoder (forked into
# runs-wave/wave-trim) and ablates feature groups to find the LEANEST model that
# keeps the performance. Adds NOTHING. Runs in parallel with the explorer loop on
# its own node + run folder (no collision).

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
RUN_DIR="$REPO/runs-wave/wave-trim"
CLAUDE_LOG="$LOOPDIR/logs/claude_trim_${JOBID}.jsonl"

echo "=== wave TRIM agent on $(hostname) at $(date) -> $RUN_DIR ==="
nvidia-smi -L || true
cd "$RUN_DIR"

PROMPT="You are a model-PRUNING / ABLATION agent on a GPU node (cwd: $RUN_DIR). wave_encoder.py is the \
current BEST waveform->ECoG encoder (per-chunk per-electrode test_corr ~0.0768, embedding dim ~1200 = \
~4800 ridge features after FIR delays). It is heavily OVER-PARAMETERIZED. Your ONLY job is to TRIM it to \
the LEANEST model that KEEPS the performance. DO NOT add any new features or ideas -- removal only. \
STEP 1: run the eval once to establish the baseline: 'python wave_encoder.py --num-train 500 --num-test 100 \
> run.log 2>&1' (foreground, timeout 1800000 ms, BLOCK until done); read run.log -- note the test_corr and \
the ridge feature dim (the 'feature shapes (N, D)' line gives D). \
NOTE: a paired bootstrap over the 23 test chunks showed the noise floor is ~0.0008, so a removal that costs < 0.0015 is statistically NOT a real loss -- strip it. \
STEP 2: ABLATE. The forward() returns a big torch.cat([... many named feature blocks ...]). Repeatedly: \
REMOVE exactly ONE feature block from that concatenation (comment it out / drop it from the cat and delete its \
now-unused computation), run the eval, and compare test_corr to the current best. If removing the block drops \
test_corr by LESS than 0.0015 (i.e. it is dead weight or marginal), KEEP IT REMOVED permanently. If removal \
costs >= 0.0015, restore that block and try removing a different one. Greedily strip the lowest-value blocks \
one at a time. \
GOAL: minimize the ridge feature dim while keeping test_corr >= ~0.0748 (within ~0.002 of 0.0768). \
Each iteration set a UNIQUE model_shorthand_name = LEAN-v1, LEAN-v2, ... and a description stating WHICH block \
you removed, its measured delta test_corr, and the new ridge feature dim. After each accepted removal, snapshot \
to wave_encoders_lib/<name>.py. Run every eval in the FOREGROUND with a generous timeout and BLOCK until it \
returns; never background it, never end your turn mid-eval. Keep going until removing ANY remaining block would \
cost >= 0.0015 (a true lean core); do at least 20 ablation evals before considering yourself done, and if you \
think you are done, double-check by trying to remove more. Report the final lean dim vs the original ~4800."

session=0; fastfails=0
while true; do
  session=$((session + 1)); echo "===== trim session #$session $(date) ====="
  start=$(date +%s)
  stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" --model "$MODEL" --dangerously-skip-permissions \
    --add-dir "$REPO" --verbose --output-format stream-json 2>&1 | stdbuf -oL tee -a "$CLAUDE_LOG"
  rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))
  echo "===== trim session #$session exited rc=$rc after ${dur}s ====="
  if [ "$dur" -lt 120 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
  if [ "$fastfails" -ge 5 ]; then echo "5 fast exits -- aborting"; break; fi
  sleep 10
done
echo "=== trim agent ended $(date) ==="; cat "$RUN_DIR/results/overall_results.csv" || true
