# autoresearch — hand-designed auditory front-ends (patient1 ECoG, waveform)

This experiment has a coding agent autonomously research how to hand-design — with
NO training — an interpretable auditory front-end that maps the **raw audio
waveform** to a per-frame feature predicting **ECoG high-gamma** (patient patient1).
The scientific goal is to model the *algorithmic priors of early hearing*: which
fixed cochlear/brainstem-style computations (filtering, compression, adaptation,
onset enhancement, modulation tuning, lateral inhibition, ...) best explain
cortical responses, without learning anything from the data.

The held-out test set is a **speech-rich** block (a podcast segment about the
monkey-selfie copyright case, `450a36e4_0190..0212`); the whole `450a36e4`
recording is excluded from training. Results are seeded with a `Wave-melspec-floor`
(waveform reproducing a log-mel, the linear floor) and a `Whisper-reference`
(pretrained-encoder ceiling, ~0.079).

## What you start from

The pipeline hands your front-end **pre-aligned audio windows** of shape `(T, win)`:
one window per 30 Hz frame, already centered on the exact same grid as the
mel-spectrogram / high-gamma (the 1 s extraction crop is baked in). You only decide
how to turn windows into features `(T, H)`. The harness then applies the standard
FIR delays + 100 ms lag + 30-frame trim and fits a ridge readout to the 166
electrodes. **Alignment is guaranteed for you — never re-window or re-time the audio
yourself.** Set `WaveEmbedder.win` (<= 4096) to choose the analysis window length.

The metric is the project's **per-sample per-electrode correlation** (`test_corr`),
identical to the mel and Whisper rows, so everything is comparable.

The fixed pipeline lives in `src/` (do NOT edit; reached via a symlink).

## Rules of the game

**You MAY** use any deterministic signal-processing transform whose parameters are
set by formula or by hand:
- FFT / STFT, gammatone / ERB / mel filterbanks, rounded-exponential filters
- rectification, log / power-law compression, envelope extraction
- fixed convolutions, lateral inhibition, divisive (gain) normalization
- adaptation / temporal integration with fixed time constants
- modulation (rate–scale) filterbanks, onset/offset detectors
- hand-written weight tensors and hand-built transformer / conv circuits

**You MAY NOT:**
- train anything — no gradient descent, optimizer, backprop, or fitting loop.
- estimate ANY front-end parameter from data: no PCA / ICA / NMF / k-means /
  sparse-coding / regression FIT to the audio. Priors are implemented in closed
  form (evolution already "found" them). The ONLY data-fit step is the final ridge
  readout, which the harness does for you (as for the mel / Whisper references).
- load pretrained model weights.
- re-window / re-time the audio (alignment is fixed by the harness).

**Interpretability matters**: prefer human-understandable auditory computations.
A small gain that adds opaque complexity is not worth it. The deliverable is
insight — *which fixed hearing priors help, and by how much* — not just a number.

## Setup

1. Pick a unique tag, e.g. `wave-jun02-run1`.
2. `python setup_run_wave.py <tag>` → creates `runs-wave/<tag>/` (src symlink,
   `wave_encoder.py`, `program.md`, `results/` seeded with `Wave-melspec-floor`,
   `wave_encoders_lib/`).
3. `cd runs-wave/<tag>` and stay there. Read `wave_encoder.py`, `src/eval.py`,
   `src/features.py`, and `results/overall_results.csv`.

## Output columns

```
subject,test_corr,train_corr,frac_test_elec_above_0.1,encoding_seconds,status,model_shorthand_name,n_params,description
```
`test_corr` is the primary metric. `n_params` = count of any hand-written weight
tensors you add (0 for pure DSP).

## The experiment loop

You are always inside the run folder (`runs-wave/<tag>/`).

LOOP FOREVER:
1. Edit `wave_encoder.py` with ONE hand-designed front-end idea (closed-form, no
   training). Update `model_shorthand_name` (unique) and `model_description`.
2. Run the eval in the FOREGROUND and WAIT for it to finish (each eval takes a few
   to ~15 minutes — that is NORMAL):
   `python wave_encoder.py --num-train 500 --num-test 100 > run.log 2>&1`
   Do NOT background it, do NOT poll with `echo wait` / `sleep` / repeated checks.
3. Read `tail -n 20 run.log` and `grep <name> results/overall_results.csv`.
4. If it crashed, read `tail -n 50 run.log`, fix, rerun.
5. Snapshot to `wave_encoders_lib/<model_shorthand_name>.py`.
6. Keep feature dimensionality MODEST (ridge memory/time grow with H*ndelays);
   if an eval is very slow or memory-heavy, reduce filters / H.

**NEVER STOP** to ask whether to continue; run until manually stopped; do at least
30 iterations. Beat the `Wave-melspec-floor` and push toward the `Whisper-reference`
ceiling with interpretable hearing priors.

**Ideas to try** (be creative, go beyond these):
- Gammatone vs. mel vs. roex cochlear filterbanks; vary filter count / Q / spacing.
- Cochlear compression laws: log, cube-root, power-law, adaptive (AGC).
- **Adaptation**: subtract/divide a leaky running mean per channel (onset emphasis).
- **Onset/offset detectors**: half-wave-rectified temporal derivatives of channels.
- **Lateral inhibition**: difference-of-neighbors across the frequency axis.
- **Modulation tuning** (Chi/Shamma): rate (temporal) × scale (spectral) filterbank
  on the cochleagram.
- Phase / fine-structure features vs. envelope-only.
- Multi-window (multi-resolution) analysis combined into one feature.
- Divisive normalization / contrast gain control across channels.

BE CREATIVE. The question is which auditory priors the cortex's high-gamma reflects.
