#!/bin/bash
#SBATCH --job-name=vpn_fit
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20-newtest/logs/phase2_%A_%a.out
#SBATCH --partition=gpu_dev,gpu_short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
#SBATCH --array=0-14
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20-newtest"
export TMPDIR="/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20-newtest/logs/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WAVE_FEAT_CACHE="$TMPDIR" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" phase2_newtest.py --task $SLURM_ARRAY_TASK_ID --ntasks 15
rc=$?; rm -rf "$TMPDIR"; exit $rc
