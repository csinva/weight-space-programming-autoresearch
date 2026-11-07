#!/bin/bash
#SBATCH --job-name=nt_whisper
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/newtest/logs/whisper_%j.out
#SBATCH --partition=gpu_dev,gpu_short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node-X,node-X,node-X,node-X
#SBATCH --requeue

set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/newtest"
ID=whisper-reference
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" rescore_task_newtest.py --only "$ID" --check --trials extra_trials.json
