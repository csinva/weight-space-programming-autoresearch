#!/bin/bash
#SBATCH --job-name=vpn_agg
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20-newtest/logs/phase3_%j.out
#SBATCH --partition=cpu_short
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20-newtest"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" phase3_aggregate.py
