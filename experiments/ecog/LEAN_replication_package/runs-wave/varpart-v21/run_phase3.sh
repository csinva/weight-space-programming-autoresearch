#!/bin/bash
#SBATCH --job-name=vp21_agg
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v21/logs/phase3_%j.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v21/logs/phase3_%j.err
#SBATCH --partition=cpu_short
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v21"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" phase3_aggregate.py
