#!/bin/bash
#SBATCH --job-name=both_fit
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-whisper-newtest/logs/fit_%j.out
#SBATCH --partition=gpu_dev,gpu_short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-whisper-newtest"
export TMPDIR="/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-whisper-newtest/logs/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export ECOG_TEST_CHUNKS_FILE=/path/to/ecog_data/circuit-building-autoresearch/evolve-wave/speech_rich_test_chunks.json ECOG_EXCLUDE_PREFIX=450a36e4 ECOG_RIDGE_GPU=1
export WAVE_FEAT_CACHE="$TMPDIR" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" both_varpart.py fit
rc=$?; rm -rf "$TMPDIR"; exit $rc
