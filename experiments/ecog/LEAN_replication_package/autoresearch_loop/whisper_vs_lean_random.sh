#!/bin/bash
#SBATCH --job-name=whisper_vs_lean
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/whisper_vs_lean_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/whisper_vs_lean_%A.err
#SBATCH --partition=gpu_short,gpu_long
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

# Compare LEAN (hand-built waveform) vs Whisper (pretrained) test_corr on the
# FIXED session-holdout test AND on several RANDOM train/test splits, to check
# whether the fixed 23-chunk single-session test is idiosyncratic.
set -uo pipefail
REPO="/path/to/ecog_data/circuit-building-autoresearch"
export ECOG_RIDGE_GPU="1"
LEANDIR="$REPO/runs-wave/whisper-cmp"
RES="$REPO/autoresearch_loop/logs/whisper_vs_lean_results.txt"
: > "$RES"

run_lean () {  # $1 = test file ("" for fixed) ; $2 = exclude prefix
  local tf="$1" excl="$2"
  cd "$LEANDIR"
  ECOG_TEST_CHUNKS_FILE="$tf" ECOG_EXCLUDE_PREFIX="$excl" \
    python wave_encoder.py --num-train 500 --num-test 100 2>/dev/null \
    | grep -E "^test_corr:" | head -1 | grep -oE "0\.[0-9]+" | head -1
}
run_whisper () {  # $1 = test file ; $2 = exclude prefix  -> nd=4 row
  local tf="$1" excl="$2"
  cd "$REPO/evolve-ecog"
  ECOG_TEST_CHUNKS_FILE="$tf" ECOG_EXCLUDE_PREFIX="$excl" \
    python whisper_fir.py 2>/dev/null \
    | awk '$1==4 {print $3}' | head -1
}

echo "=== whisper vs LEAN: fixed session-holdout vs random splits — $(hostname) $(date) ===" | tee -a "$RES"
FIXED="$REPO/evolve-wave/speech_rich_test_chunks.json"

echo "--- FIXED session-holdout (test=450a36e4, excluded from train) ---" | tee -a "$RES"
L=$(run_lean "$FIXED" "450a36e4"); W=$(run_whisper "$FIXED" "450a36e4")
echo "FIXED   LEAN=$L   Whisper=$W" | tee -a "$RES"

for s in 1 2 3 4; do
  TF="$REPO/random_splits/random_test_seed${s}.json"
  echo "--- RANDOM split seed$s (100 random test chunks, no session holdout) ---" | tee -a "$RES"
  L=$(run_lean "$TF" ""); W=$(run_whisper "$TF" "")
  echo "RAND$s   LEAN=$L   Whisper=$W" | tee -a "$RES"
done

echo "=== SUMMARY ===" | tee -a "$RES"
python - "$RES" <<'PY' | tee -a "$RES"
import sys, re
lines=open(sys.argv[1]).read().splitlines()
lean=[]; whis=[]; fixed=None
for ln in lines:
    m=re.match(r"(FIXED|RAND\d)\s+LEAN=([0-9.]+)\s+Whisper=([0-9.]+)", ln)
    if m:
        l=float(m.group(2)); w=float(m.group(3))
        if m.group(1)=="FIXED": fixed=(l,w)
        else: lean.append(l); whis.append(w)
import statistics as st
if fixed: print(f"fixed session-holdout:  LEAN={fixed[0]:.4f}  Whisper={fixed[1]:.4f}  gap={fixed[1]-fixed[0]:+.4f}")
if lean:
    lm,ws=st.mean(lean),st.mean(whis)
    ls=st.pstdev(lean) if len(lean)>1 else 0; wsd=st.pstdev(whis) if len(whis)>1 else 0
    print(f"random splits (n={len(lean)}): LEAN={lm:.4f}±{ls:.4f}  Whisper={ws:.4f}±{wsd:.4f}  gap={ws-lm:+.4f}")
PY
echo "=== done $(date) ===" | tee -a "$RES"
