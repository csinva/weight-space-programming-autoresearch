#!/bin/bash
#SBATCH -J nt_extract
#SBATCH -p cpu_short
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --array=0-19
#SBATCH -o logs/extract_%a.out
cd /path/to/ecog_data/circuit-building-autoresearch/runs-wave/newtest
../../.venv/bin/python extract_newtest_wave.py $SLURM_ARRAY_TASK_ID 20
