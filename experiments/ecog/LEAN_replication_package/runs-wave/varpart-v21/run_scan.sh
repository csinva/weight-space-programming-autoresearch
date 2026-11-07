#!/bin/bash
#SBATCH --job-name=vp21_scan
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v21/logs/scan_%j.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v21/logs/scan_%j.err
#SBATCH --partition=cpu_short
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
set -uo pipefail
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v21"
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" ../lean-v21/degenerate_scan.py "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v21" > degenerate_v21.txt
"/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" ../lean-v21/degenerate_scan.py ../varpart-v20 > ../lean-v21/degenerate_v20.txt
