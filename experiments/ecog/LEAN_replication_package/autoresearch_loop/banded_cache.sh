#!/bin/bash
#SBATCH --job-name=banded_cache
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/banded_cache_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/banded_cache_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --cpus-per-task=8

set -uo pipefail
REPO="/path/to/ecog_data/circuit-building-autoresearch"
export ECOG_TEST_CHUNKS_FILE="$REPO/evolve-wave/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"
export ECOG_RIDGE_GPU="1"
cd "$REPO/banded_probe"
echo "=== build cache $(date) ==="; nvidia-smi -L || true
python cache_features.py

echo; echo "=== validate baseline (uniform band scaling must reproduce ~0.0760) ==="
python - <<'PY'
import probe
c = probe.load_cache()
cv, test, _ = probe.fit_banded(probe.expand_scale([1.0]*len(c["band_cols"]), c["band_cols"], c["p"]), c)
print(f"BASELINE uniform: cv={cv:.5f}  test_corr={test:.5f}  (target ~0.0760)")
PY
echo "=== done $(date) ==="
