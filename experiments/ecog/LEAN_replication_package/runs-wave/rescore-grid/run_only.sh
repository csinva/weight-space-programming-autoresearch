#!/bin/bash
#SBATCH --job-name=rg_only
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid/logs/only_%j.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid/logs/only_%j.err
#SBATCH --partition=gpu_dev,gpu_short,gpu_long
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/rescore-grid"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" rescore_task.py --only "$1"
