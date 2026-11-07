#!/bin/bash
#SBATCH --job-name=reset_worker2
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/reset_%x_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/reset_%x_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# RADICAL-RESET worker, SECOND WAVE (E/F/G/H/I). Each pursues a distinct
# NONLINEAR auditory paradigm chosen to be orthogonal to a linear-cochleagram
# span (so a linear ridge readout cannot already reconstruct it). Clean-slate
# framing: NOT told prior project history. Seeded from the lean encoder CODE.
#
# Usage:  sbatch --job-name=reset_E [--partition=gpu_short] autoresearch_loop/reset_worker2.sh E

set -uo pipefail
WORKER="${1:?usage: reset_worker2.sh <E|F|G|H|I>}"
MODEL="${2:-opus}"
REPO="/path/to/ecog_data/circuit-building-autoresearch"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
JOBID="${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOOPDIR/tmp/${JOBID}"; mkdir -p "$TMPDIR"
export ECOG_TEST_CHUNKS_FILE="$REPO/evolve-wave/speech_rich_test_chunks.json"
export ECOG_EXCLUDE_PREFIX="450a36e4"
export ECOG_RIDGE_GPU="1"
mkdir -p "$LOOPDIR/logs"

RUN_DIR="$REPO/runs-wave/reset-${WORKER}"
CLAUDE_LOG="$LOOPDIR/logs/claude_reset_${WORKER}_${JOBID}.jsonl"
CACHE="/path/to/scratch/feat_cache_reset_${WORKER}"; mkdir -p "$CACHE"

case "$WORKER" in
  E) PARADIGM="YOUR PARADIGM = DIVISIVE NORMALIZATION (the canonical cortical computation, Carandini-Heeger). Build the front-end around divisive gain control: from an auditory spectrogram, every channel's response is DIVIDED by a pooled drive from its spectrotemporal neighborhood (a semi-saturation constant PLUS a weighted sum of surrounding channels x recent time, the whole raised to a power). Tile a BANK of normalization operators with DIFFERENT pool shapes and sizes (narrow vs broad in frequency, fast vs slow in time, symmetric vs asymmetric/lagging pools) and different exponents and semi-saturation constants. This nonlinearity -- response divided by local context energy -- is believed to be THE canonical cortical operation; it produces contrast and gain invariances that a purely linear front-end can NEVER reconstruct. Make the normalized responses (and the residual of raw drive vs its normalized version) the representation. All closed-form via FFT / cumsum; NO fitted parameters." ;;
  F) PARADIGM="YOUR PARADIGM = PREDICTIVE CODING / SURPRISE. Represent NOT the stimulus but the PREDICTION ERROR. At each 30 Hz frame a FIXED closed-form predictor forecasts the current cochleagram from its recent past (fixed-coefficient linear AR over each band's history, and/or a fixed spectral-transition expectation), and the feature is the RESIDUAL = actual minus predicted -- the unpredicted, surprising part of the signal that auditory cortex / STG is thought to encode (deviance, mismatch-negativity, onsets). Build a BANK of predictors at multiple horizons and orders; keep signed AND rectified residuals; add PRECISION weighting (residual divided by its running variance, so surprise is scaled by reliability); add CROSS-BAND surprise (a band's error given the other bands). The residual is a nonlinear, context-dependent quantity a linear readout cannot synthesize from the raw bands. Fixed predictors only; the ONLY data-fit is the ridge readout." ;;
  G) PARADIGM="YOUR PARADIGM = SOUND-TEXTURE SUMMARY STATISTICS (McDermott-Simoncelli, training-free). Represent the signal by the rich NONLINEAR summary statistics that define sound texture: per-band envelope marginal moments (mean, coefficient of variation, skew, kurtosis), CROSS-BAND envelope correlations, modulation-band power within each cochlear channel, and CROSS-correlations between modulation bands and between cochlear channels at matched modulation rates (the C1 / C2 statistics). CRITICAL: you must NEVER re-window or re-time the audio -- compute every statistic as a CAUSAL RUNNING quantity at 30 Hz using leaky / lowpass moving accumulators (running mean, variance, skew; running products for the correlations), so each frame carries its LOCAL texture statistics. These nonlinear cross-products and higher moments are exactly what a linear readout cannot reconstruct from the bands. Closed-form; the ridge readout is the only fit." ;;
  H) PARADIGM="YOUR PARADIGM = ECHO-STATE / FIXED-RANDOM RECURRENT EXPANSION (reservoir computing, training-free). Drive a high-dimensional reservoir with FIXED RANDOM (never-trained) recurrent weights using the cochleagram as input: x[t] = tanh(W_res @ x[t-1] + W_in @ u[t]), with the random matrices scaled to a chosen spectral radius for fading memory. The reservoir state is a rich NONLINEAR function of the ENTIRE stimulus history -- long temporal context and multiplicative interactions a feedforward or linear front-end cannot produce. NOTE on the rules: random FIXED weights are NOT a data-fit (nothing is estimated from the data), so this is fully legal; the ONLY data-fit step remains the ridge readout. NOTE on performance: the recurrence is intrinsically sequential, so a per-frame scan over time IS permitted HERE (this is the one allowed exception to the no-time-loop rule) -- but vectorize across reservoir units, keep the reservoir modest (e.g. a few hundred units), and run on GPU. Explore reservoir size, spectral radius, leak rate, input scaling, multiple reservoirs at different timescales, and the nonlinearity. Make the reservoir state the representation." ;;
  I) PARADIGM="YOUR PARADIGM = TEMPORAL FINE STRUCTURE & FREQUENCY MODULATION (the phase dimension most front-ends discard). Most front-ends keep only the ENVELOPE (power) and throw the phase away. Instead build the representation from the ANALYTIC SIGNAL of each cochlear channel: INSTANTANEOUS FREQUENCY (derivative of the unwrapped phase), instantaneous bandwidth, GROUP DELAY, and FM sweep rate and direction per band -- the fine temporal structure the auditory nerve phase-locks to below ~1.5 kHz. Add FM-selective detectors (upward vs downward instantaneous-frequency glides), phase coherence across adjacent bands, and the envelope-to-TFS ratio. Compute the analytic signal via FFT (Hilbert transform). These phase / FM quantities are ORTHOGONAL to any envelope / power representation -- a linear readout over power bands cannot recover them. Closed-form; the ridge readout is the only fit." ;;
  *) echo "unknown worker $WORKER (use E|F|G|H|I)"; exit 1 ;;
esac

PROMPT="You are starting a FRESH research effort on a GPU node (cwd: $RUN_DIR). \
GOAL: hand-design a CLOSED-FORM auditory front-end that maps the pre-aligned speech WAVEFORM to \
patient patient1's ECoG high-gamma cortical activity. A linear ridge readout (fit by the harness -- \
the ONLY data-fit step allowed) decodes your features into the 166 electrodes; the score is the \
per-chunk per-electrode correlation on a held-out test set. \
\
IMPORTANT FRAMING -- THIS TASK IS ALREADY SOLVED: other models have succeeded at predicting this \
neural signal from speech audio; strong front-ends exist and reach well above naive baselines. The \
predictive signal is genuinely, abundantly there in the data. You are REPLICATING a known-achievable \
result, not chasing something speculative. So be BOLD and ambitious: if a representation is not \
working, assume the limitation is YOUR feature design and try a structurally different idea -- do not \
conclude the ceiling has been hit. \
\
NOVELTY IS REQUIRED -- you CANNOT win by rehashing textbook auditory features or your prior knowledge \
of the literature. The standard, well-known features (a plain cochleagram, mel bins, fixed-tau \
adaptation, generic onsets) are exactly what already plateaus; simply re-deriving them from memory \
will NOT break through. CRUCIAL DESIGN PRINCIPLE: because the readout is a LINEAR ridge, any feature \
that is just a LINEAR function of the log-cochleagram bands is ALREADY reconstructable by the readout \
and adds NOTHING. To make progress you must produce GENUINELY NONLINEAR features -- products, ratios, \
divisions, rectified residuals, higher moments, phase / instantaneous-frequency, recurrent state -- \
quantities a linear map over the bands cannot synthesize. Bias HARD toward nonlinear, original \
structure. Each iteration ask yourself: 'is this actually nonlinear and new, or is it a linear \
re-mix of bands the ridge already has?' -- if the latter, escalate to something more inventive. \
\
$PARADIGM \
\
THE STARTING CODE: wave_encoder.py currently holds a competent generic baseline front-end (a \
gammatone-cochleagram model) so you have a working harness and a reference number on iteration 1. You \
may BUILD ON it, REPLACE it wholesale, or IGNORE it -- whatever your paradigm demands. Do NOT feel \
anchored to it; your job is to push your paradigm hard and try genuinely new structural ideas, not to \
make small tweaks. results/overall_results.csv starts essentially empty -- this is a clean slate, so \
do not look for or rely on any external history. \
\
HARD RULES (never break): closed-form weights/filters ONLY -- absolutely NO training / optimizer / \
gradient / backprop, and NO parameter estimated from data (no PCA/ICA/NMF/k-means/sparse-coding/ \
pretrained weights). The ridge readout done by the harness is the ONLY data-fit step. NEVER re-window \
or re-time the audio -- the alignment to the neural data is fixed by the harness; operate on the \
windows you are given. \
\
PERFORMANCE RULES: you are on an 80 GB A100 with a 24h job. Do NOT rush and do NOT trim features for \
memory. Implement filterbanks / 2D modulation / convolutions / autocorrelation via FFT or cumsum -- \
never naive Python loops over time (the ONLY exception is an intrinsically-sequential recurrence if \
your paradigm explicitly calls for one). Feature dimensionality up to ~10000 is fine. Run each eval as \
a SINGLE FOREGROUND command and BLOCK until it returns, with a GENEROUS Bash timeout (e.g. timeout \
1800000 ms = 30 min); NEVER use a timeout below ~900000 ms and NEVER kill a healthy eval; if the \
harness auto-backgrounds it, WAIT for it rather than ending your turn. CACHE expensive reusable \
per-chunk components to $CACHE/ keyed by a hash of the exact computation code+params plus chunk id, \
and load on a hit. \
\
EACH ITERATION: edit wave_encoder.py with ONE new structural idea from your paradigm, set a UNIQUE \
model_shorthand_name + a description stating what you changed and the measured test_corr + feature \
dim, then run 'python wave_encoder.py --num-train 500 --num-test 100 > run.log 2>&1' in the FOREGROUND, \
block until done, read 'tail -n 20 run.log', and snapshot the file to wave_encoders_lib/<name>.py. Do \
NOT background the eval, do NOT write any until/while/sleep/grep polling loop, and do NOT end your turn \
while an eval runs. Do MANY iterations. NEVER stop to ask or conclude -- if an idea stalls, pivot to a \
different structural variant WITHIN your paradigm and keep going."

echo "=== RESET2 worker $WORKER on $(hostname) at $(date) -> $RUN_DIR ==="
nvidia-smi -L || true
cd "$RUN_DIR"

session=0; fastfails=0
while true; do
  session=$((session + 1)); echo "===== reset-$WORKER session #$session start $(date) ====="
  start=$(date +%s)
  stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
    --model "$MODEL" --dangerously-skip-permissions --add-dir "$REPO" \
    --verbose --output-format stream-json 2>&1 | stdbuf -oL tee -a "$CLAUDE_LOG"
  rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))
  echo "===== reset-$WORKER session #$session exited rc=$rc after ${dur}s at $(date) ====="
  if [ "$dur" -lt 120 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
  if [ "$fastfails" -ge 5 ]; then echo "5 consecutive <120s exits -- aborting"; break; fi
  sleep 10
done
echo "=== reset2 worker $WORKER ended at $(date) ==="
