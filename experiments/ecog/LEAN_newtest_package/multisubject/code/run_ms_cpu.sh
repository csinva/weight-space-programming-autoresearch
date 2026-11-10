#!/bin/bash
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj/logs/%x_%j.out
#SBATCH --partition=cpu_short
#SBATCH --time=1:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
# usage: sbatch -J name run_ms_cpu.sh both_agg SUBJ  |  run_ms_cpu.sh vp_agg SUBJ
cd "/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj"
if [ "$1" = vp_agg ]; then "/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" varpart_aggregate.py "$2"; else "/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python" ms_analyses.py "$1" "$2"; fi
