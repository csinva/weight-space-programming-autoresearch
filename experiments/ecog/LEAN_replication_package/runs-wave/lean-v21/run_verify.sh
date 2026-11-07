#!/bin/bash
#SBATCH --job-name=v21_verify
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-v21/verify_%j.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-v21/verify_%j.err
#SBATCH --partition=gpu_dev,gpu_short,gpu_long
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-v21"
export ECOG_TEST_CHUNKS_FILE="/path/to/ecog_data/circuit-building-autoresearch/evolve-wave/speech_rich_test_chunks.json"
export WAVE_FEAT_CACHE="/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-v21/tmp_${SLURM_JOB_ID}"; mkdir -p "$WAVE_FEAT_CACHE"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" verify_v21.py
rc=$?; rm -rf "$WAVE_FEAT_CACHE"; exit $rc
