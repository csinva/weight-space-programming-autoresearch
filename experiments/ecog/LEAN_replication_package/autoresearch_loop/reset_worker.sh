#!/bin/bash
#SBATCH --job-name=reset_worker
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/reset_%x_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/reset_%x_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# RADICAL-RESET worker: a CLEAN-SLATE autoresearch loop. Each worker (A/B/C/D)
# pursues a DIFFERENT auditory paradigm from scratch -- it is NOT told the prior
# project history, plateau, or which features failed. It starts from the lean
# encoder CODE as a working baseline but is encouraged to depart radically.
#
# Usage:  sbatch --job-name=reset_A autoresearch_loop/reset_worker.sh A

set -uo pipefail
WORKER="${1:?usage: reset_worker.sh <A|B|C|D>}"
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
  A) PARADIGM="YOUR PARADIGM = BIOPHYSICAL AUDITORY PERIPHERY. Build a physiologically faithful model of the auditory periphery as the front-end, on the hypothesis that a realistic periphery LINEARIZES the cortical code far better than ad-hoc compression. Include, all closed-form: (1) basilar-membrane filtering with LEVEL-DEPENDENT compressive nonlinearity (dual-resonance / compressive-gammachirp: gain that drops as input level rises), (2) inner-hair-cell transduction (asymmetric half-wave rectification + ~1 kHz lowpass), (3) auditory-NERVE synapse ADAPTATION in the Zilany-Bruce style (an exponential rapid/short-term component PLUS a dual power-law slow component), (4) a fast cochlear AGC / medial-olivocochlear efferent gain that normalizes dynamic range across ~100-1000 ms, and (5) distinct fiber populations (low- vs high-spontaneous-rate -> different thresholds/onset emphasis). Make the periphery the centerpiece." ;;
  B) PARADIGM="YOUR PARADIGM = CORTICAL SPECTRO-TEMPORAL MODULATION (Chi-Ru-Shamma). Represent the signal ENTIRELY in the modulation domain using a cortical STRF model. From an auditory spectrogram, apply a rich bank of 2D spectro-temporal modulation filters tiling RATE (temporal modulation, a log-spaced grid ~1-32 Hz, both upward- and downward-sweeping) x SCALE (spectral modulation, ~0.25-8 cycles/octave). These rate-scale selective filters are the basis auditory cortex is believed to use. Make the rate-scale modulation representation the whole model -- not an add-on to a cochleagram. Implement the 2D filtering via FFT. Explore the tiling density, phase/quadrature energy, and temporal/spectral asymmetry." ;;
  C) PARADIGM="YOUR PARADIGM = PHONOLOGICAL / LINGUISTIC FEATURES (top-down). Instead of bottom-up acoustics, build a stream of closed-form LINGUISTIC feature detectors, on the hypothesis that STG encodes phonological features and prosody directly. Include: (1) articulatory/manner matched filters (vowel / stop / fricative / nasal / approximant) and place-of-articulation and voicing detectors computed from the spectrum, (2) RELATIVE / speaker-normalized INTONATION -- log-F0 minus and divided by a slow running baseline, plus pitch velocity/acceleration (encode relative not absolute pitch), (3) SYLLABIC structure via a sparse peakRate landmark train (impulses at local maxima of the loudness-envelope derivative), and (4) vowel FORMANT-space coordinates (F1/F2/F3) expressed CONTRASTIVELY vs a running mean. Build the representation as a phonological+prosodic feature vector." ;;
  D) PARADIGM="YOUR PARADIGM = WAVELET SCATTERING TRANSFORM (Mallat / Anden-Mallat). Build a deep, translation-stable, fully training-free scattering front-end: a cascade of constant-Q wavelet filterbanks + complex modulus + lowpass averaging, computing FIRST-order (~ a log-mel-like envelope) AND SECOND-order scattering coefficients (which recover the amplitude-modulation / transient information that averaging destroys at first order). Consider JOINT time-frequency scattering (2D wavelets over the time-frequency plane) for spectro-temporal structure. Treat the (log-compressed) scattering coefficients as the entire feature stream. Tune the Q-factors per order, number of octaves, averaging scale T, and scattering depth. Implement wavelet convolutions via FFT." ;;
  *) echo "unknown worker $WORKER (use A|B|C|D)"; exit 1 ;;
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
will NOT break through. The teams that succeeded at this had to INVENT new representations -- they did \
not just implement what was already in papers. Treat the named paradigm below as a STARTING DIRECTION, \
not a recipe: your job is to push it somewhere genuinely new -- combine it with unexpected structure, \
take a known idea to an extreme it has never been taken, or design a feature nobody would find in a \
textbook. Bias HARD toward brand-new, original ideas over safe, familiar ones. Each iteration ask \
yourself: 'is this actually a new idea, or am I just re-implementing something standard?' -- if the \
latter, escalate to something more inventive. \
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
memory. Implement every filterbank / 2D modulation / scattering / autocorrelation via FFT or cumsum -- \
never naive Python loops over time. Feature dimensionality up to ~10000 is fine. Run each eval as a \
SINGLE FOREGROUND command and BLOCK until it returns, with a GENEROUS Bash timeout (e.g. timeout \
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

echo "=== RESET worker $WORKER on $(hostname) at $(date) -> $RUN_DIR ==="
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
echo "=== reset worker $WORKER ended at $(date) ==="
