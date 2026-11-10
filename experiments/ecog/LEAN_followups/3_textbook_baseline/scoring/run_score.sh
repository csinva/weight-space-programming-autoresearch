#!/bin/bash
#SBATCH --job-name=tb_score
#SBATCH --output=/path/to/ecog_data/lean_textbook_baseline/scoring/logs/task_%A_%a.out
#SBATCH --partition=gpu_dev,gpu_short,gpu_long
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
#SBATCH --array=0-29
/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python /path/to/ecog_data/lean_textbook_baseline/scoring/tb_score_task.py $SLURM_ARRAY_TASK_ID
