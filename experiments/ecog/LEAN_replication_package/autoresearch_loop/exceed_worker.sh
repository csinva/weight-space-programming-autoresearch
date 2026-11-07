#!/bin/bash
#SBATCH --job-name=exceed_worker
#SBATCH --output=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/exceed_%x_%A.out
#SBATCH --error=/path/to/ecog_data/circuit-building-autoresearch/autoresearch_loop/logs/exceed_%x_%A.err
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#
# Directed worker: ADD an audio-only block to LEAN-v17 to EXCEED Whisper on the
# SURFACE GRID. Objective = GRID-ONLY test_corr (electrodes 0-124).
#   sbatch --job-name=exc_scatter  autoresearch_loop/exceed_worker.sh scatter
#   sbatch --job-name=exc_frontend autoresearch_loop/exceed_worker.sh frontend
set -uo pipefail
MISSION="${1:?usage: exceed_worker.sh <scatter|frontend>}"
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

case "$MISSION" in
  scatter) RUN_DIR="$REPO/runs-wave/exceed-scatter"
    TASK="ADD a WAVELET SCATTERING TRANSFORM block, computed CLOSED-FORM from the waveform, appended to \
LEAN's existing features (BOTH expert reviews rate scattering the single strongest untrained family to \
add). Build a deep, translation-stable, training-free scattering front-end: a cascade of constant-Q \
wavelet filterbanks (Morlet/Gabor, log-spaced center freqs) + complex MODULUS + lowpass averaging, with \
FIRST-order coefficients AND crucially SECOND-order scattering (recovers amplitude-modulation/transient \
structure that 1st-order averaging destroys). Strongly consider JOINT TIME-FREQUENCY SCATTERING (2D \
wavelets over the time-frequency plane) -- especially relevant for speech. Tune Q-factors per order, \
octaves, averaging scale T, depth. FFT convolutions. It is a principled replacement for hand-written \
product/cube blocks: keep the (log-compressed) scattering coefficients as DENSE SOFT maps." ;;
  frontend) RUN_DIR="$REPO/runs-wave/exceed-frontend"
    TASK="ADD a much RICHER FIXED AUDITORY FRONT-END to LEAN's features. TWO independent expert reviews \
CONVERGED on the same diagnosis: the plateau is MISSING PRIMARY AUDITORY STRUCTURE, not missing scalar \
interaction products -- so add DENSE SOFT MAPS, not more summaries. Add, all closed-form (try these as \
separate iterations, keep each that helps grid test_corr): \
(1) SHORTER analysis windows (128/256/512 samples) for transients/onsets/bursts/consonant edges that \
strongly drive high-gamma; ideally multi-taper (DPSS/Thomson) power for stability, esp. short windows. \
(2) DENSE PERIODICITY as SOFT distributions (NOT argmax/best-F0): a log-spaced F0-salience grid \
(60-400 Hz, 24-48 bins/octave), a normalized AUTOCORRELOGRAM/summary-correlogram slice, a CEPSTROGRAM in \
the pitch quefrency range, CPP (cepstral peak prominence), SRH/subharmonic ratio, periodicity entropy, \
top-2 peak ratio. Give the probe the WHOLE profile (or its DCT), per electrode lag-range differences \
matter. (3) GTCC/cepstral features: USE the already-defined but UNUSED self.dct_basis on the log- \
gammatone energies -> low-order cepstra (envelope) + high-order (fine structure) + delta/delta-delta. \
(4) PER-BAND modulation filterbank: for each cochlear-band envelope, fixed modulation filters at \
1/2/4/8/16/32 Hz keeping modulation energy + pos/neg derivative energy + onset/offset asymmetry; USE the \
already-defined but UNUSED self.fb_fine (40-channel cochleagram) for a DENSER 2D rate x scale STMF bank. \
(5) SOURCE/FILTER: spectral tilt (log-power slope 200-2000 Hz), LPC/Burg formants+bandwidths, harmonic \
tilt (H1-H2 proxy), voiced/unvoiced confidence. (6) CONTRAST/EDGE: local contrast normalization (subtract \
local mean / divide local RMS, in time AND frequency), DoG/LoG on the cochleagram, separate upward vs \
downward spectral-sweep detectors. Improve harmonic_comb with fractional-bin interpolation + log-spaced \
candidates + 1/k harmonic weights. Prefer DENSE soft maps over scalar summaries throughout." ;;
  combine) RUN_DIR="$REPO/runs-wave/exceed-combine"
    TASK="COMBINE the two prior runs into one. Your wave_encoder.py is ALREADY the best FRONTEND encoder \
(LEAN-v17 + rich-front-end blocks, esp. cochleagram-CONTRAST normalization + future-cube context; grid \
test_corr ~0.0879). Now ALSO ADD the wavelet-SCATTERING block from the best SCATTER run, located at \
$REPO/runs-wave/exceed-scatter/wave_encoders_lib/SCAT-v103.py (grid ~0.0882). READ that file, diff it \
against plain LEAN-v17 to isolate the scattering additions (its 2nd-order Morlet scattering computation in \
__call__ AND any filterbank/setup it adds in __init__), and PORT those additions into THIS encoder \
alongside the existing frontend blocks (append the scattering coefficients to feat). The two families \
were developed independently (contrast-normalization vs multiscale scattering) so they likely capture \
DIFFERENT structure -- stacking them should exceed the better single (0.0882). After the merge works, \
keep iterating: tune the combination, prune redundancy, and try further small additions. Goal: push grid \
test_corr past 0.0882 toward the Whisper grid bar 0.096." ;;
  push) RUN_DIR="$REPO/runs-wave/exceed-v18"
    TASK="Your wave_encoder.py is LEAN-v18: the best hand-built encoder so far = LEAN-v17 + wavelet \
SCATTERING + cochleagram-CONTRAST normalization + a rich front-end. It already absorbed two prior runs. \
TRAJECTORY: we have HALVED the gap to Whisper each iteration (all-166: 0.0061 -> 0.0043 -> 0.0021), so \
exceeding Whisper is achievable -- keep halving. The remaining target lives on the surface GRID. Find \
AUDIO-ONLY closed-form features that capture the REMAINING Whisper-orthogonal grid signal and push grid \
test_corr past LEAN-v18 (~0.089) toward the Whisper grid bar (~0.096). You have WIDE LATITUDE -- do NOT \
just re-tune what's already there. Promising untried/under-tried directions: DEEPER scattering (3rd-order \
/ richer joint time-frequency scattering); a genuinely DENSE 2-D cortical rate x scale modulation bank on \
the high-res cochleagram with quadrature PHASE pairs; longer TEMPORAL CONTEXT (multi-second integration, \
acausal); cross-FAMILY interactions (scattering coefficients x contrast x periodicity); finer spectro- \
temporal resolution; periodicity/pitch SOFT maps done better than before. Prefer DENSE SOFT MAPS and \
biologically-plausible nonlinearities over scalar summaries. Append your block(s) to feat." ;;
  context) RUN_DIR="$REPO/runs-wave/exceed-v20-context"
    TASK="Your wave_encoder.py is LEAN-v20, the best hand-built encoder (grid ~0.0905). The single biggest \
STRUCTURAL thing Whisper has that LEAN lacks is LONG TEMPORAL CONTEXT -- its transformer integrates \
SECONDS of audio, while LEAN's features are mostly LOCAL. This is a fresh, UNDER-EXPLORED direction. Add \
AUDIO-ONLY closed-form features carrying LONG-RANGE temporal structure into each frame: (1) a fixed-RANDOM \
ECHO-STATE RESERVOIR driven by the cochleagram -- x[t]=(1-leak)x[t-1]+leak*tanh(W_res@x[t-1]+W_in@u[t]); \
random fixed weights are LEGAL (not a data-fit), the ridge is the ONLY fit; a per-frame sequential scan is \
PERMITTED here (the one allowed time-loop) -- gives nonlinear fading memory over SECONDS; try several \
spectral radii / leak rates / reservoir sizes; (2) VERY-LONG-HORIZON multi-scale context: causal AND \
acausal running statistics at long taus (1s/2s/4s/8s), long-lag deltas, running spectral/prosodic \
context; (3) phrase/utterance-scale slow envelopes and their interactions with local features. Prefer \
representations that inject SECONDS of history per frame. Append to feat." ;;
  deep) RUN_DIR="$REPO/runs-wave/exceed-v20-deep"
    TASK="Your wave_encoder.py is LEAN-v20 (grid ~0.0905). It already has 2nd-order Morlet scattering + \
cochleagram-contrast + gating/quadratic products. Go DEEPER and BROADER on nonlinear lifting -- these are \
UNDER-EXPLORED: (1) 3rd-ORDER scattering and richer JOINT TIME-FREQUENCY scattering (2D wavelets over the \
time-frequency plane); the prior 2nd-order run plateaued, so push order/JTFS structure further; (2) \
explicit CROSS-FREQUENCY COUPLING at scale: phase-amplitude coupling (low-band phase x high-band \
envelope), cross-band envelope products across DISTANT bands and lags, comodulation across the full \
spectro-temporal field (not just local); (3) a deeper FIXED nonlinear cascade -- stack \
filterbank->modulus->filterbank for a few deterministic layers (a scattering-like deep net, no learning). \
Prefer DENSE soft coefficients over scalar summaries. Append to feat." ;;
  speaker) RUN_DIR="$REPO/runs-wave/exceed-v20-speaker"
    TASK="Your wave_encoder.py is clean LEAN-v20 (grid ~0.0904). A glass-brain diagnostic localized Whisper's \
ENTIRE remaining grid advantage to a few ANTERIOR-TEMPORAL / temporal-pole electrodes (G119,G63,G125,G54, \
G113); the eval prints 'gap15_corr' over those targets (LEAN 0.2765 -> Whisper 0.3357) -- WATCH IT, but ACCEPT \
on grid test_corr. \
IMPORTANT -- A PRIOR RUN ALREADY EXHAUSTED these and they did NOT help: scalar speaker summaries (LTAS \
cepstra, spectral tilt, slow F0/formant scalars), light FiLM gating products, fine cochleagram, banded-ridge \
column-replication, and generic nonlinearities (half-wave/signed-cube/asym) on cepstral coefs. DO NOT \
re-add any of those, and DO NOT fall back to generic nonlinear lifting -- the deep/context branches already \
mined that. \
THIS RUN HAS ONE FOCUS: an explicit, AGGRESSIVE SPEAKER-NORMALIZATION / FREQUENCY-WARP (VTLN) pathway -- the \
piece the prior run only tried weakly. The mechanism: the SAME phoneme sits at different absolute \
frequencies across talkers; canonicalize it. Build it closed-form: \
(1) Estimate a slowly-varying per-frame WARP FACTOR alpha(t) (vocal-tract-length / formant-scale) from the \
running spectrum -- e.g. running-median formant spacing, the scale of the spectral envelope, or the LTAS \
log-frequency centroid over a long tau (1-4 s). alpha(t) is a running statistic, NOT a fit -- legal. \
(2) WARP THE LOG-FREQUENCY AXIS of the cochleagram by alpha(t): in the log-frequency (or real-cepstrum) \
domain a vocal-tract-length change is a SHIFT/SCALE, so resample each frame's spectral envelope along \
log-f by alpha(t) (linear interp along the freq axis -- fully vectorizable, no time-loop). This maps every \
talker onto a CANONICAL speaker. Recompute the phonetic/formant/template maps ON THE WARPED cochleagram and \
append them as DENSE soft maps. KEEP the un-warped maps too so the readout sees both. \
(3) Try a small BANK of fixed warp factors (e.g. alpha in {0.85,0.9,...,1.15}) applied to the running \
envelope -- a warp-invariance pyramid -- and also the warped-MINUS-unwarped residual maps. \
(4) Long-tau CEPSTRAL MEAN-AND-VARIANCE NORMALIZATION as the channel/speaker-normalization complement: \
subtract running cepstral mean and divide running cepstral std per quefrency over 1-4 s, then rebuild maps. \
Prefer DENSE warped soft maps over scalars. The whole point is to move gap15_corr; push grid test_corr past \
0.0904 toward the Whisper grid bar (~0.096)." ;;
  *) echo "unknown mission $MISSION (scatter|frontend|combine|push|context|deep|speaker)"; exit 1 ;;
esac
CLAUDE_LOG="$LOOPDIR/logs/claude_exceed_${MISSION}_${JOBID}.jsonl"
CACHE="/path/to/scratch/feat_cache_exceed_${MISSION}"; mkdir -p "$CACHE"

PROMPT="You are improving a hand-built CLOSED-FORM auditory front-end on a GPU node (cwd: $RUN_DIR). \
GOAL: map the pre-aligned speech WAVEFORM to patient patient1's ECoG high-gamma; a linear ridge readout \
(fit by the harness -- the ONLY data-fit step) decodes your features. THE SCORE IS GRID-ONLY: per-chunk \
per-electrode test correlation over the 125 SURFACE-GRID electrodes (0-124) ONLY -- depth and strip \
electrodes are EXCLUDED. (A diagnostic showed ALL of Whisper's advantage over LEAN lives on the surface \
grid; depth/strips are low-signal and were only diluting the metric.) wave_encoder.py already evaluates \
GRID-ONLY -- just run it and read test_corr. \
\
STARTING POINT: wave_encoder.py holds LEAN-v17, a strong interpretable baseline; run it once to get your \
GRID-ONLY baseline (about 0.084). THE BAR TO BEAT: a pretrained Whisper encoder scores about 0.096 \
GRID-ONLY; LEAN+Whisper reaches about 0.100, PROVING audio-only features can hold the grid signal Whisper \
captures that LEAN currently misses. YOUR MISSION: capture that missing signal with AUDIO-ONLY closed-form \
features and PUSH LEAN PAST 0.096 GRID-ONLY. Known-achievable: the signal is there. \
\
$TASK \
\
HOW TO ADD: LEAN-v17's __call__ builds features and ends with 'feat = self._condition(groups)' then \
'return feat.float().cpu().numpy()'. Compute your new block as a (T, K) torch tensor from the input \
'windows'/'x', then 'feat = torch.cat([feat, your_block], dim=1)' BEFORE the return. Do NOT remove or \
break existing LEAN features -- you are ADDING. Do NOT touch the __main__ grid-only eval. \
\
EACH ITERATION: edit wave_encoder.py to add/expand your block with ONE structural idea, set a UNIQUE \
model_shorthand_name + a description stating what changed + the measured test_corr + feature dim, then run \
'python wave_encoder.py --num-train 500 --num-test 100 > run.log 2>&1' in the FOREGROUND, block until \
done, read 'tail -n 20 run.log', and snapshot to wave_encoders_lib/<name>.py. ACCEPTANCE (CRITICAL): we \
are near the hand-crafted ceiling, so progress comes ENTIRELY from ACCUMULATING MANY SMALL IMPROVEMENTS. \
KEEP any change that improves grid test_corr over the running best by ANY positive amount -- do NOT \
discard a gain because it looks 'within noise'; the route from 0.084 to 0.096 is INCREMENTAL and small \
wins COMPOUND. Only REVERT a change that strictly HURTS (< running best) or is bit-identical/redundant. \
Keep stacking small wins relentlessly; never wait for a big jump. \
\
HARD RULES: closed-form weights/filters ONLY -- NO training/optimizer/gradient/backprop, NO parameter \
estimated from data (no PCA/ICA/NMF/k-means/pretrained weights). The ridge is the ONLY data-fit step. \
NEVER re-window or re-time the audio. PERFORMANCE: 80 GB A100, 192 GB RAM; FFT/cumsum, no naive Python \
time-loops; cache reusable per-chunk components to $CACHE. Run each eval as a SINGLE FOREGROUND command \
with a GENEROUS Bash timeout (>= 900000 ms); never background it or end your turn while it runs. Do MANY \
iterations; if an idea stalls, try a different structural variant. NEVER stop to conclude."

echo "=== EXCEED worker '$MISSION' (GRID-ONLY) on $(hostname) at $(date) -> $RUN_DIR ==="
nvidia-smi -L || true
cd "$RUN_DIR"
session=0; fastfails=0
while true; do
  session=$((session + 1)); echo "===== exceed-$MISSION session #$session start $(date) ====="
  start=$(date +%s)
  stdbuf -oL -eL "$CLAUDE_BIN" -p "$PROMPT" \
    --model "$MODEL" --dangerously-skip-permissions --add-dir "$REPO" \
    --verbose --output-format stream-json 2>&1 | stdbuf -oL tee -a "$CLAUDE_LOG"
  rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))
  echo "===== exceed-$MISSION session #$session exited rc=$rc after ${dur}s at $(date) ====="
  if [ "$dur" -lt 120 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
  if [ "$fastfails" -ge 5 ]; then echo "5 consecutive <120s exits -- aborting"; break; fi
  sleep 10
done
echo "=== exceed worker $MISSION ended at $(date) ==="
