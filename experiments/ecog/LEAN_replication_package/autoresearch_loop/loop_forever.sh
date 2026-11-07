#!/bin/bash
#SBATCH --job-name=ecog_forever
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/forever_%x_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/forever_%x_%A.err
#SBATCH --partition=gpu_short,gpu_long,gpu4_short,gpu4_medium,gpu8_short,gpu8_medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8

# CONTINUOUS autoresearch loop: re-launches the headless Claude agent every time it
# exits (it otherwise stops once it stops emitting tool calls), so iteration never
# halts until the SLURM time limit. Continues an EXISTING run folder (reads its
# results history). GPU ridge is on by default.
#
# Usage:
#   sbatch --job-name=speech_cont autoresearch_loop/loop_forever.sh speech
#   sbatch --job-name=wave_cont   autoresearch_loop/loop_forever.sh wave

set -uo pipefail   # NOT -e: a claude non-zero exit must not kill the while loop

TRACK="${1:?usage: loop_forever.sh <speech|wave>}"
MODEL="${2:-opus}"
REPO="/path/to/ecog_data/circuit-building-autoresearch"
LOOPDIR="$REPO/autoresearch_loop"
CLAUDE_BIN="$HOME/.npm-global/bin/claude"
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
JOBID="${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOOPDIR/tmp/${JOBID}"; mkdir -p "$TMPDIR"
export ECOG_EXCLUDE_PREFIX="450a36e4"
export ECOG_RIDGE_GPU="1"
mkdir -p "$LOOPDIR/logs"

if [ "$TRACK" = "speech" ]; then
  RUN_DIR="$REPO/runs-ecog/ecog-speech-23608983"
  export ECOG_TEST_CHUNKS_FILE="$REPO/evolve-ecog/speech_rich_test_chunks.json"
  EVAL="python interpretable_transformer.py --num-train 500 --num-test 100"
  EDIT="interpretable_transformer.py"; SNAP="interpretable_transformers_lib"
  RULES="This is the MELSPEC track: hand-write closed-form weights of the transformer feature extractor in $EDIT (running-mean adaptation, half-wave rectifiers, harmonic/pitch combs, phonetic templates, cross-band products, multi-timescale sliding stats). Pursue diverse, structurally-new interpretable ideas and recombine the best-performing prior feature blocks. \
CROSS-BRANCH UPDATE: the sibling WAVEFORM track has reached 0.0716 (you are plateaued at ~0.0613 doing rank-based tweaks). It got there by stacking, on top of its base, a FULL high-level feature suite PLUS two things you likely under-use: (1) explicit FORMANT features -- peak-pick the spectral-envelope formants F1/F2/F3 per frame (formant frequencies, bandwidths, and amplitude ratios, and their temporal change) -- and (2) per-channel/per-band VARIANCE and spectral MOMENTS. ADOPT THESE: add explicit formant detectors/tracking and channel-variance/moment features computed from the mel, recombined onto your best base. (Your front-end is the fixed 50-bin mel, so you cannot match the waveform's cochlear front-end, but formants + the full high-level stack should break you past 0.0613.) Keep each eval under ~8 minutes; if a feature is slower, simplify or cache it."
elif [ "$TRACK" = "wave" ]; then
  RUN_DIR="$REPO/runs-wave/wave-lean"
  export ECOG_TEST_CHUNKS_FILE="$REPO/evolve-wave/speech_rich_test_chunks.json"
  EVAL="python wave_encoder.py --num-train 500 --num-test 100"
  EDIT="wave_encoder.py"; SNAP="wave_encoders_lib"
  RULES="This is the WAVEFORM track: hand-design closed-form auditory front-ends in $EDIT operating on the pre-aligned audio windows (gammatone/mel/roex filterbanks, compression laws, multi-timescale adaptation, onset/offset, modulation filterbanks, lateral inhibition, AGC). NEVER re-window/re-time the audio (alignment is fixed by the harness). \
PERFORMANCE RULE: you are on an 80 GB A100 GPU with a 48h job -- DO NOT trim features for GPU memory and DO NOT rush. Evals up to ~20 minutes are FINE; your full-stack evals take ~400-700s and complete cleanly, so NEVER kill a healthy eval. Run the eval as a single foreground command with a GENEROUS Bash timeout (e.g. 1800000 ms = 30 min) and BLOCK until it returns; NEVER set a timeout below ~900000 ms (a too-short timeout kills a healthy eval and wastes the iteration). If the harness auto-backgrounds a long eval, do NOT abandon it and do NOT end your turn -- WAIT for it (block on the background task) and read the result when it returns. Feature dimensionality up to ~10000 is fine on the A100. Only abandon a feature if a single eval would genuinely exceed ~25 minutes; implement any 2D spectro-temporal / rate-scale (STMF) or autocorrelation feature via FFT/cumsum (never naive loops). Cache expensive reusable per-chunk components (gammatone cochleagram, multi-tau adaptation) to the feat cache if helpful. \
FUSION DIRECTIVE (key new result): a one-shot experiment concatenated your best front-end (gammatone 3-window x 6-tau cochleagram + AM-depth + loudness + centroid, 0.0612) with the MELSPEC branch's high-level feature classes and reached 0.0626 -- beating either alone. So your highest-value direction now is to FUSE those high-level feature classes onto your multi-tau cochleagram base and optimize the combination to beat 0.0626. Add, computed from your own cochleagram/spectrum (all closed-form, no training): harmonic-comb PITCH SALIENCE / voicing (sum of energy at F0 harmonics for a bank of F0 candidates), PHONETIC spectral templates (sibilant/vowel/nasal/plosive matched filters), CROSS-BAND comodulation (products of band energies), SPECTRAL MOMENTS (centroid/spread/skew/kurtosis), HARMONICS-TO-NOISE ratio, spectral FLATNESS and ROLLOFF, and pitch STRENGTH. Build these on top of the proven base and recombine; keep evals reasonable; cache the cochleagram base. \
STATUS: you have PLATEAUED at ~0.0768. Temporal-fine-structure, autocorrelation, and cross-channel-coherence were tried and did NOT help (all 0.0736-0.0768) -- abandon those. \
NEW LITERATURE-BACKED DIRECTIONS to pursue (closed-form, from human auditory-cortex/STG ECoG studies; most are NORMALIZATIONS of features you already compute): \
(1) RELATIVE / SPEAKER-NORMALIZED PITCH -- your harmonic-comb/autocorrelation gives a per-frame F0; do NOT use it absolutely. Encode F0 MINUS (and divided by) a slow leaky running baseline (a per-speaker z-score of log-F0), plus its temporal derivative (pitch change). Chang lab (Science 2017): 84pct of STG intonation electrodes track RELATIVE not absolute pitch. This is the single biggest untried gap. \
(2) CONTRASTIVE FORMANTS -- F1 and F2 MINUS a slow running-mean (contrast vs recent context), not absolute formants. STG vowel responses are contrastive (Sjerps/Chang 2019). \
(3) DISCRETE peakRate LANDMARK -- a SPARSE magnitude-weighted impulse train at local maxima of the FIRST DERIVATIVE of the broadband loudness envelope (not a continuous onset). Oganian/Chang 2019: the discrete event-train BEATS the continuous envelope for middle-STG. \
(4) SILENCE-GATED ONSET -- fire only on an energy rise that follows >=~200ms (6 frames) of near-silence (posterior-STG onset zone, Hamilton/Chang). \
(5) POWER-LAW (scale-free) ADAPTATION -- convolve each band with a power-law-decaying FIR kernel ~(t+t0)^-beta (NO single fixed tau), distinct from your fixed-tau leaky means; gives dynamic-range normalization within ~1s. \
(6) AUDITORY-NERVE SYNAPSE ADAPTATION -- you have the raw waveform, so apply a Zilany-Bruce-style AN adaptation (exponential + dual power-law) to the cochleagram; AN-periphery front-ends improve linear encoding of auditory cortex (2020 PNAS). \
PARSIMONY = GOOD GAIN-PER-DIMENSION (not absolute gain): you START from a trimmed lean base (wave_encoder.py = LEAN-v16, ~5000 ridge features, test_corr ~0.0760). A SMALL cheap feature is welcome to keep even if its benefit is marginal -- what bloats the model is adding HUNDREDS-to-THOUSANDS of dims for negligible benefit. \
(a) The model is ALREADY trimmed to its lean core -- do NOT re-trim it and do NOT add back the redundant blocks that were removed. \
(b) Add the new features ONE block at a time and decide by GAIN-PER-DIMENSION: KEEP a SMALL block (<= ~50 added embedding dims) as long as it does NOT hurt (delta test_corr >= ~-0.0008, the bootstrap noise floor) -- small cheap features are fine to keep even when the gain is marginal, and several may help different electrodes and add up. KEEP a LARGER block (> ~50 dims) only if it earns its size with a real gain (>= +0.0015). REMOVE any block that hurts, and any large block that does not justify its dimensions. ACCUMULATE the kept small features (build the combined model), do not test each in isolation and discard. \
(c) Report each block's measured delta and dimensionality in the description. Prefer compact, high-yield features; the goal is a LEANER model that generalizes, not a bigger one."
else
  echo "unknown track $TRACK"; exit 1
fi

CLAUDE_LOG="$LOOPDIR/logs/claude_cont_${TRACK}_${JOBID}.jsonl"

PROMPT="You are CONTINUING an autonomous research loop on a GPU node (cwd: $RUN_DIR). \
results/overall_results.csv already holds many prior attempts. FIRST read it and the recent \
snapshots in $SNAP/ to see the current best and what has been tried, then keep iterating to \
beat it. $RULES \
HARD RULES (unchanged): closed-form weights only -- absolutely NO training/optimizer/gradient/ \
backprop and NO parameter estimated from data (no PCA/ICA/NMF/k-means/sparse-coding); the ONLY \
data-fit step is the ridge readout done by the harness; no pretrained weights. \
Each iteration: edit $EDIT with ONE new idea (build on / recombine the best), set a UNIQUE \
model_shorthand_name + description, then run the eval as a SINGLE FOREGROUND command and BLOCK \
until it returns -- pass a long Bash timeout (e.g. timeout 1800000 ms): '$EVAL > run.log 2>&1'. \
ABSOLUTELY DO NOT launch the eval in the background, DO NOT write any until/while/sleep/grep \
polling loop, and DO NOT end your turn while an eval is still running -- just wait for that one \
command to finish, then read 'tail -n 20 run.log' and snapshot to $SNAP/<name>.py. \
EXPENSIVE feature extraction is fine, but to avoid recomputing the same thing every iteration, \
CACHE per-chunk feature components you are likely to reuse: write them to \
/path/to/scratch/feat_cache_${TRACK}/ (mkdir it) keyed by a hash of the exact \
feature-computation code+params plus the chunk id, and LOAD from cache on a hit instead of \
recomputing; only compute the genuinely new part each iteration. Do MANY iterations. NEVER stop \
to ask or conclude -- if you think you are done, pick a new direction and keep going."

echo "=== CONTINUOUS loop track=$TRACK on $(hostname) at $(date) ==="
echo "run_dir=$RUN_DIR  model=$MODEL"
nvidia-smi -L || true
cd "$RUN_DIR"

session=0; fastfails=0
while true; do
  session=$((session + 1))
  echo "===== claude session #$session start $(date) ====="
  start=$(date +%s)
  stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
    --model "$MODEL" --dangerously-skip-permissions --add-dir "$REPO" \
    --verbose --output-format stream-json 2>&1 | stdbuf -oL tee -a "$CLAUDE_LOG"
  rc=${PIPESTATUS[0]}
  dur=$(( $(date +%s) - start ))
  echo "===== claude session #$session exited rc=$rc after ${dur}s at $(date) ====="
  # guard against a tight crash-loop (e.g. auth failure): abort after 5 fast exits
  if [ "$dur" -lt 120 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
  if [ "$fastfails" -ge 5 ]; then echo "5 consecutive <120s exits -- aborting to avoid spin"; break; fi
  sleep 10
done
echo "=== continuous loop ended at $(date) ==="
