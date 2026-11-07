#!/bin/bash
#SBATCH --job-name=rg_pilot
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid/logs/pilot_%A_%a.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid/logs/pilot_%A_%a.err
#SBATCH --partition=gpu_dev,gpu_short,gpu_long
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
#SBATCH --array=0-2
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid"
IDS=(wave-trim__LEAN_v16_953005 reset-C__PHON_v10_560f1d lean-interp__INTERP_v2zz_interact_071c69)
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" rescore_task.py --only ${IDS[$SLURM_ARRAY_TASK_ID]} --check
