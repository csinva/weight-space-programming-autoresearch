#!/bin/bash
#SBATCH --job-name=both_agg
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-whisper-newtest/logs/agg_%j.out
#SBATCH --partition=cpu_short
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/lean-whisper-newtest"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" both_varpart.py aggregate
