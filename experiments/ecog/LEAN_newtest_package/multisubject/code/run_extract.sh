#!/bin/bash
#SBATCH -J ms_extract
#SBATCH -p cpu_short
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --array=0-49
#SBATCH -o logs/extract_%a.out
cd /path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj
../../.venv/bin/python extract_multi.py $SLURM_ARRAY_TASK_ID 50
