#!/bin/bash
#SBATCH --job-name=vp20_stats
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20/logs/phase1_%j.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20/logs/phase1_%j.err
#SBATCH --partition=gpu_dev,gpu_short,gpu_long
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node-X
#SBATCH --requeue
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20"
export ECOG_TEST_CHUNKS_FILE="/path/to/ecog_data/circuit-building-autoresearch/evolve-wave/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"
export ECOG_RIDGE_GPU="1"
export TMPDIR="/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20/logs/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WAVE_FEAT_CACHE="$TMPDIR"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" phase1_stats.py
rc=$?; rm -rf "$TMPDIR"; exit $rc
