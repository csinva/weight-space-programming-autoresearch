#!/bin/bash
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj/logs/%x_%A_%a.out
#SBATCH --partition=gpu_dev,gpu_short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
# usage: sbatch -J name [--array=0-14] run_ms_gpu.sh MODE SUBJ [NTASKS]
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj"
export TMPDIR="/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj/logs/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WAVE_FEAT_CACHE="$TMPDIR" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ECOG_RIDGE_GPU=1
if [ "$1" = vp_fit ]; then "/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" ms_analyses.py vp_fit "$2" "${SLURM_ARRAY_TASK_ID:-0}" "$3"; else "/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" ms_analyses.py "$1" "$2"; fi
rc=$?; rm -rf "$TMPDIR"; exit $rc
