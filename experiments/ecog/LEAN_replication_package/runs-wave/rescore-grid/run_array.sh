#!/bin/bash
#SBATCH --job-name=rg_rescore
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid/logs/task_%A_%a.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid/logs/task_%A_%a.err
#SBATCH --partition=gpu_short,gpu_long
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
#SBATCH --array=0-99
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" rescore_task.py --task $SLURM_ARRAY_TASK_ID --ntasks 100
