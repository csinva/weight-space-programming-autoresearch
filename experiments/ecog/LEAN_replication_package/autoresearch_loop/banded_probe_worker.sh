#!/bin/bash
#SBATCH --job-name=banded_probe
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/banded_probe_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/banded_probe_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# PROBE-ONLY loop. Features are FROZEN (cached LEAN-v16). The worker may improve
# ONLY the linear read-out (banded ridge) in banded_probe/probe.py.

set -uo pipefail
MODEL="${1:-opus}"
REPO="/path/to/ecog_data/circuit-building-autoresearch"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
JOBID="${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOOPDIR/tmp/${JOBID}"; mkdir -p "$TMPDIR"
mkdir -p "$LOOPDIR/logs"
RUN_DIR="$REPO/banded_probe"
CLAUDE_LOG="$LOOPDIR/logs/claude_banded_probe_${JOBID}.jsonl"

PROMPT="You are improving the LINEAR PROBE (read-out) of an audio->ECoG encoding model on a GPU node \
(cwd: $RUN_DIR). THE FEATURES ARE FROZEN: a strong hand-built acoustic front-end (LEAN-v16) has already \
been computed and cached under $RUN_DIR/cache/ (XtX, XtY, the 5 bootstrap CV tuning subsets, the per- \
chunk test features, and the raw feature-GROUP column indices in bands.json). You may NOT recompute or \
change features. You may edit ONLY the read-out in $RUN_DIR/probe.py. \
\
THE PROBLEM: the harness read-out is a per-voxel ridge with a SINGLE global regularization strength per \
electrode -- so every feature group shares the same shrinkage. But LEAN's feature groups have WILDLY \
DIFFERENT statistics and signal/noise (a 20-channel cochleagram vs a 2-dim formant track vs a large STMF \
bank), so one global lambda is wrong. YOUR JOB: implement PROPER BANDED RIDGE -- give each feature group \
(band) its own effective regularization -- to beat the uniform baseline. \
\
WHAT IS ALREADY THERE (in probe.py, all CORRECT -- keep the math): load_cache(); per_chunk_elec_corr() \
(the exact project metric -- do NOT change it); fit_banded(s, cache) which fits the per-voxel ridge with \
a per-COLUMN scale vector s and returns (cv_score, test_corr, wt) -- banded ridge IS column scaling: \
scaling band b by theta_b is mathematically equivalent to giving band b its own lambda. expand_scale() \
maps per-band thetas -> the column vector s. The UNIFORM baseline (all thetas = 1) reproduces the \
harness's plain ridge (~0.0760). \
\
WHAT TO IMPROVE: the function search_probe(cache) -- the band DEFINITION and the per-band scale SEARCH. \
Ideas, in rough order: (1) MERGE the ~25 raw groups in bands.json into a smaller number of meaningful \
bands (e.g. by feature family) so the search is tractable and each band is statistically coherent; \
(2) better search than greedy coordinate ascent -- e.g. random/log-uniform search over per-band scales, \
multiplicative coordinate ascent with refinement, or a himalaya-style banded-ridge scaling search; \
(3) consider per-band normalization (scale each band so its columns have comparable total variance \
BEFORE the ridge) as a principled starting point; (4) optionally let bands share groups sensibly. \
\
CRITICAL RULES: SELECT the band scales by CV (cv_score from the tuning subsets), NEVER by test_corr -- \
test is reported only, never optimized against (no test leakage). Keep per_chunk_elec_corr and the \
fit_banded ridge math intact -- you are choosing s, not changing how the ridge is solved. The features, \
the cache, the metric, and the data split are all FIXED. The paired-bootstrap noise floor is about \
+/-0.0008; a real probe improvement needs to clear it and must come with a higher CV score. \
\
EACH ITERATION: edit search_probe (or add a banded scheme it calls), then run \
'python probe.py > run.log 2>&1' in the FOREGROUND with a GENEROUS Bash timeout (>= 900000 ms); block \
until it returns; read 'tail -n 30 run.log' and the appended row in probe_results.csv (label, test_corr, \
cv_score). Keep changes that RAISE cv_score (and report their test_corr). Snapshot good versions by \
copying probe.py to probe_lib/<label>.py (mkdir -p probe_lib). Do MANY iterations; never edit anything \
outside probe.py; never end your turn while an eval runs; if an idea stalls, try a different banding or \
search strategy."

echo "=== BANDED-PROBE worker on $(hostname) at $(date) -> $RUN_DIR ==="
nvidia-smi -L || true
cd "$RUN_DIR"; mkdir -p probe_lib

session=0; fastfails=0
while true; do
  session=$((session + 1)); echo "===== banded-probe session #$session start $(date) ====="
  start=$(date +%s)
  stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
    --model "$MODEL" --dangerously-skip-permissions --add-dir "$REPO" \
    --verbose --output-format stream-json 2>&1 | stdbuf -oL tee -a "$CLAUDE_LOG"
  rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))
  echo "===== banded-probe session #$session exited rc=$rc after ${dur}s at $(date) ====="
  if [ "$dur" -lt 120 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
  if [ "$fastfails" -ge 5 ]; then echo "5 consecutive <120s exits -- aborting"; break; fi
  sleep 10
done
echo "=== banded-probe worker ended at $(date) ==="
