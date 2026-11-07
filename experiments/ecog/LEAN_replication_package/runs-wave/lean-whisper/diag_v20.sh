#!/bin/bash
#SBATCH --job-name=diag_v20
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-whisper/diag_v20_%j.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-whisper/diag_v20_%j.err
#SBATCH --partition=gpu_short
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node-X
set -uo pipefail
REPO="/path/to/ecog_data/circuit-building-autoresearch"
cd "$REPO/runs-wave/lean-whisper"
export ECOG_TEST_CHUNKS_FILE="$REPO/evolve-wave/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"
export ECOG_RIDGE_GPU="1"
export TMPDIR="$REPO/runs-wave/lean-whisper/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
"$REPO/.venv/bin/python" diag_v20.py
