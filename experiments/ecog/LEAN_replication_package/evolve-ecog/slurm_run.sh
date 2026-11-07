#!/bin/bash
#SBATCH --job-name=ecog_autoresearch
#SBATCH --output=logs/ecog_autoresearch_%A.out
#SBATCH --error=logs/ecog_autoresearch_%A.err
#SBATCH --partition=gpu_short,gpu_long,gpu4_short,gpu4_medium,gpu8_short,gpu8_medium
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Run ONE encoding experiment (the current interpretable_transformer.py) on a GPU.
#
# Usage:
#   sbatch slurm_run.sh <run_dir> [num_train] [num_test]
# e.g.
#   sbatch slurm_run.sh /path/to/circuit-building-autoresearch/runs-ecog/ecog-jun01-run1 200 40
#
# For interactive autoresearch looping, instead grab a GPU and run the loop there:
#   srun --partition=gpu_short --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=04:00:00 --pty bash
#   cd <run_dir> && python interpretable_transformer.py --num-train 200 --num-test 40

set -euo pipefail

RUN_DIR="${1:?usage: sbatch slurm_run.sh <run_dir> [num_train] [num_test]}"
NUM_TRAIN="${2:-200}"
NUM_TEST="${3:-40}"

mkdir -p "$(dirname "$0")/logs"
cd "${RUN_DIR}"

echo "ECoG autoresearch experiment in ${RUN_DIR} on $(hostname) at $(date)"
echo "num_train=${NUM_TRAIN} num_test=${NUM_TEST}"

python interpretable_transformer.py --num-train "${NUM_TRAIN}" --num-test "${NUM_TEST}" --device cuda
