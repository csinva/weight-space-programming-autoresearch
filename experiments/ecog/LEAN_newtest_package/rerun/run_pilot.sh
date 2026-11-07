#!/bin/bash
#SBATCH --job-name=nt_pilot
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/newtest/logs/pilot_%A_%a.out
#SBATCH --partition=gpu_dev,gpu_short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
#SBATCH --array=0-4
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/newtest"
ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" pilot_ids.txt)
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" rescore_task_newtest.py --only "$ID" --check
