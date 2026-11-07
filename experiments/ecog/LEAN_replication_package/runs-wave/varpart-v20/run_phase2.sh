#!/bin/bash
#SBATCH --job-name=vp20_fit
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20/logs/phase2_%A_%a.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20/logs/phase2_%A_%a.err
#SBATCH --partition=gpu4_short,gpu4_medium,gpu_dev,gpu_short,gpu_long
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X
#SBATCH --requeue
#SBATCH --array=0-44
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20"
export TMPDIR="/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20/logs/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WAVE_FEAT_CACHE="$TMPDIR"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" phase2_fit.py --task $SLURM_ARRAY_TASK_ID --ntasks 45
rc=$?; rm -rf "$TMPDIR"; exit $rc
