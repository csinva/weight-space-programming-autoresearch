#!/bin/bash
#SBATCH --job-name=ms_rescore
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj/logs/task_%A_%a.out
#SBATCH --partition=gpu_short,gpu_long
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
#SBATCH --array=0-99
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" rescore_task_multi.py --task $SLURM_ARRAY_TASK_ID --ntasks 100
