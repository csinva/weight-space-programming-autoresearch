#!/bin/bash
#SBATCH --job-name=ling_eval
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/ling_eval_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/ling_eval_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -uo pipefail
REPO="/path/to/ecog_data/circuit-building-autoresearch"
RUN_DIR="$REPO/runs-wave/lean-interp-ling"
export ECOG_TEST_CHUNKS_FILE="$REPO/evolve-wave/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"
export ECOG_RIDGE_GPU="1"

cd "$RUN_DIR"
echo "=== ling_eval on $(hostname) at $(date) ==="
nvidia-smi -L || true

echo; echo "########## BASELINE: LEAN-v16 (matched) ##########"
python wave_encoder_base.py --num-train 500 --num-test 100

echo; echo "########## LING-v1: LEAN + 16 interpretable linguistic features ##########"
python wave_encoder.py --num-train 500 --num-test 100

echo; echo "=== done at $(date) ==="; cat "$RUN_DIR/results/overall_results.csv"
