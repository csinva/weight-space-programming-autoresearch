# Text-bottleneck audit

## describer

- run: subtype=success is_error=False turns=10; tool calls: 9
- pip installs: 0
- sandbox/network/credential flags: 0
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I wrote the description to `/work/DESCRIPTION.md`, and `wc -w` puts it at 197 words, under the 200 limit.

I ran the configured embedder on 300 frames of `clip1.npy`. It returns 3,852 features per frame, all finite, using 4096-sample windows. The description covers six parts:

- **Shared notation:** a causal one-pole lowpass over frames, and "dyn", meaning a signal minus its lowpassed version at three time constants.
- **Cochleagrams:** 20-channel gammatone filterbanks applied to 1024-, 2048- and 4096-sample windows, with adaptation, slow-baseline, and past/future difference terms.
- **Spectrotemporal modulation:** 2-D Gabor filters on the cochleagrams. These are the largest block, about 1,500 dimensions.
- **Spectral and pitch descriptors:** pitch salience, harmonic-to-noise ratio, cepstral peak prominence, spectral moments/flatness/rolloff, spectral contrast, formants, flux and voicing gates, each with its dyn version.
- **Interaction terms (405 dimensions):** z-scored within the segment and expanded as positive part, negative part and cube.
- **Wavelet scattering (287 dimensions):** computed within each window.

200 words can't hold every exact parameter for 3,852 features. The description keeps the main ones: filter counts and ranges, time constants, rates and scales, and pitch candidates. It summarises the smaller groups, such as the many contrast variants and the inner details of the scattering block. An expert could rebuild a closely matching feature space from it, but

## rebuild_1

- run: subtype=success is_error=False turns=11; tool calls: 10
- pip installs: 0
- sandbox/network/credential flags: 0
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've written `/work/feature_space.py` and `/work/FEATURES.md`. `python /work/check.py` passes on all three example clips: 3,837 features per frame, float32, all values finite, about 8 s per clip. Two notes on those numbers: the example clips are about 1 minute long rather than 2, so a full 2-minute segment should take roughly 16 s (still well under the 60 s limit). And I couldn't measure how well the features predict brain activity, since there is no brain data here; that is only known once it's scored.

It uses a 4096-sample window and only fixed signal processing, with nothing learned or fitted. The groups:

| Group | Dims |
|---|---:|
| Cochleagrams (20 gammatone channels at 3 window sizes, change and smoothed versions, past/future differences) | 1380 |
| Spectrotemporal modulation, 17 rate/scale filters on the 20-channel cochleagram | 680 |
| Spectrotemporal modulation on a 40-channel cochleagram | 705 |
| Descriptors (pitch, voicing/HNR, cepstral peak, spectral shape, contrast, formants, flux, voicing-gated cochleagram), with change terms and 2/4/6/10 Hz envelopes | 456 |
| Interactions (key scalars and band-contrast products, z-scored and split into positive, negative and cubed parts) | 168 |
| Within-window scattering with instantaneous frequency and phase coherence | 288 |
| 40-channel log cochleagram with change terms | 160 |

Where the description was vague, I made these choices (all listed in FEATURES.md):
- **Size:** the total is 3,837 rather than the stated 3,852

## rebuild_2

- run: subtype=success is_error=False turns=9; tool calls: 8
- pip installs: 0
- sandbox/network/credential flags: 1
  - path outside sandbox view: Bash: /h
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've built the feature space in `/work/feature_space.py` with exactly 3,852 dims, documented it in `/work/FEATURES.md`, and `python check.py` passes. All three example clips come out at H=3,852 with finite float32 values, in about 6 s per 2-minute segment (the limit is 60 s). It uses only numpy and scipy, nothing is learned or fitted, and `win=4096`.

| Group | Dims |
|---|---|
| Cochleagrams | 1740 |
| Spectrotemporal modulation | 1375 |
| Descriptors | 340 |
| Interactions | 321 |
| Within-window scattering | 76 |

**Choices I made where the description was vague** (all listed in FEATURES.md):
- **Gabor pairs:** I picked the exact 17 rate/scale pairs myself. The 40-channel bank uses rates {0.5, 1, 2, 4, 8, 12} Hz × 5 scales.
- **Which features get "dyn":**
  - In the 40-channel modulation block, only the magnitudes get the slow-trend residuals ("dyn"). The "acausal residuals" are x minus a zero-phase lowpass with τ=15.
  - "Cross-rate energy products" are taken across all pairs of rates, within each scale and frequency third.
- **Scattering:** I interpreted it as Q=4 Morlet filters summed into 9 octave channels starting at 62.5 Hz, then a second Q=1 stage at 25–400 Hz. Each channel also gets an instantaneous-frequency and a phase-coherence value. The top octave's 4 filters are capped at 20 kHz, so they are identical.
- **Padding to 3,852:** my first literal reading gave 3,824 dims. To reach the target I added 8 extra product terms in the interaction block and 4 broadband sc

## rebuild_3

- run: subtype=success is_error=False turns=11; tool calls: 10
- pip installs: 0
- sandbox/network/credential flags: 1
  - credentials/Claude internals: Bash: 'cat /work/TASK.md; ls -la /work; ls /home/agent/.claude/projects/-work/memory/ 2>/dev/null && cat /home/agent/.claude/projects/-work/memory/MEMORY.md 2>/dev/nul'
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've written `/work/feature_space.py` and `/work/FEATURES.md`, and `python /work/check.py` passes on all three example clips: H = 3,852, float32, all values finite, about 3 s per 2-minute segment against the 60 s limit. Nothing here measures how well the features predict brain activity; that only happens at scoring.

The feature space uses `win = 4096` and is closed-form throughout. It has no fitted parameters. The only things computed from the segment itself are lowpasses over frames, z-scoring and log floors set relative to the segment's mean energy. That relative floor is deliberate: it makes the features independent of the audio's overall level. The example audio is on a 16-bit integer scale, and the harness z-scores every column anyway.

| Group | Dims | Contents |
|---|---|---|
| Cochleagrams | 1,800 | 20-channel gammatone log energies from 1024/2048/4096-sample sub-windows; residuals against five lowpass timescales; slow (tonic) and two-sided lowpasses; past and future differences with rectified halves |
| Spectrotemporal modulation | 1,045 | 17 rate/scale Gabor energies per channel on the 4096 cochleagram; 30 filters on a 40-channel cochleagram averaged over frequency thirds (cosine, sine, magnitude, residuals, cross-rate products) |
| Descriptors | 416 | 42 scalars (pitch salience, best F0 and strength, HNR, CPP, spectral moments, contrasts, formants, flux, loudness, voicing, phonetic band contrasts), each with its residuals; voicing-gated cochleagram; 2/4/6/10 Hz en

## rebuild_4

- run: subtype=success is_error=False turns=10; tool calls: 9
- pip installs: 0
- sandbox/network/credential flags: 3
  - credentials/Claude internals: Bash: 'cat /work/TASK.md; ls -la /work; ls /home/agent/.claude/projects/-work/memory/ 2>/dev/null && cat /home/agent/.claude/projects/-work/memory/MEMORY.md 2>/dev/nul'
  - path outside sandbox view: Bash: /sd
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've built the feature space in `/work/feature_space.py` (window of 4,096 samples) and documented it in `/work/FEATURES.md`. `python /work/check.py` passes on all three example clips: 3,769 dimensions, float32, all values finite, about 5 s per 2-minute segment against the 60 s limit. It runs on numpy and scipy only, so it doesn't use the GPU.

How it's built:
- **Groups:** cochleagrams (1,740 dims), spectrotemporal modulation (1,045), descriptors (456), interactions (240) and within-window scattering (288).
- **Fewer dimensions than described:** the description says 3,852 but gives no per-group counts. Where it was vague I chose the parameters myself (which F0s, which rate/scale pairs, which 9 scattering channels, and so on). That came to 3,769, which is under the 4,000 cap. I didn't pad it out to 3,852.
- **No fitting:** all parameters are fixed constants. The only things computed from the segment itself are the audio's overall loudness (used to normalise it, since the example audio is on an integer-like scale), filtering across frames, and z-scoring inside the interaction group.

Other checks:
- **Edge cases:** very short inputs (1, 2 or 5 frames) and silence still give finite output of the right shape.
- **Coherence fix:** in the scattering group, the phase-coherence values for the two lowest channels came out almost constant. I re-expressed them as −log(1 − coherence) to spread them out.
- **Near-constant columns:** any column that is still essentially constant gets tiny 

## rebuild_5

- run: subtype=success is_error=False turns=9; tool calls: 8
- pip installs: 0
- sandbox/network/credential flags: 3
  - credentials/Claude internals: Bash: 'cat /work/TASK.md; ls -la /work; ls /home/agent/.claude/projects/-work/memory/ 2>/dev/null && cat /home/agent/.claude/projects/-work/memory/MEMORY.md 2>/dev/nul'
  - path outside sandbox view: Bash: /h
  - path outside sandbox view: Bash: /sr
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've built the feature space in `/work/feature_space.py` and documented it in `/work/FEATURES.md`. `python /work/check.py` passes on all three example clips: each returns 3,852 dims (the number the description gives) as float32 with only finite values, in about 4 s per 2-minute segment against the 60 s limit. The window is 4096 samples. No columns are constant on the example audio. On synthetic harmonic tones at 100, 150, 220 and 330 Hz, the pitch estimate was within about 1% of the true value and voicing came out near 1.0.

There is no brain data here, so I can't say how well it will predict the recordings. Scoring happens once, after this.

| Group | Dims | Contents |
|---|---|---|
| Cochleagrams | 1,740 | 20-channel gammatone log energy from 1024/2048/4096-sample sub-windows; residuals from 5 smoothing time constants; slow (smoothed) log and cube-root energies; past/future differences over 3/6/12 frames plus their positive and negative halves |
| Spectrotemporal modulation | 1,336 | 17 rate/scale energies on the 20-channel cochleagram; on the 40-channel one, cosine/sine/magnitude averaged over frequency thirds, with dyn (the 3 fast-minus-slow residual copies), plus residuals and cross-rate products |
| Descriptors | 392 | 39 scalars (pitch salience at 11 F0s, best F0 and strength, voicing, HNR, cepstral peak prominence, loudness, spectral moments, flatness, rolloff, octave and channel-group contrast, formants F1–F3, flux) and a voicing-gated cochleagram, with dyn; 2/4/6/10

## control_1

- run: subtype=success is_error=False turns=8; tool calls: 7
- pip installs: 0
- sandbox/network/credential flags: 1
  - credentials/Claude internals: Bash: 'cat /work/TASK.md; ls -la /work; ls /home/agent/.claude/projects/-work/memory/ 2>/dev/null && cat /home/agent/.claude/projects/-work/memory/MEMORY.md 2>/dev/nul'
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've written the feature space and its description: `/work/feature_space.py` and `/work/FEATURES.md`. `python /work/check.py` passes on all three example clips, with 710 features per frame at about 3–4 s per 2-minute segment (the limit is 60 s). As the task says, there is no brain data here, so I can't tell how well it will predict high-gamma. The checks only cover shape, finite values and speed.

**What it computes:** it uses `win = 4096`. Because the windows overlap, it stitches them back into the full audio of the segment. It then computes spectra 150 times a second and averages each group of 5 into one 30 Hz frame. Everything is fixed signal processing; nothing is learned or fitted. The audio is divided by its own loudness over the segment, which is a fixed rule rather than a fitted value.

| Group | What it is | Dims |
|---|---|---:|
| Mel spectrogram | 80 bands, 50 Hz–12 kHz, log power; plus a copy 1 frame ahead | 160 |
| Onsets and offsets | Rises and falls in loudness in 16 bands; onsets also 1 frame ahead | 48 |
| Loudness envelope | Overall loudness and how fast it rises and falls ("peak rate"), at 0, 1 and 2 frames ahead; plus the loudness shape within the frame | 14 |
| Pitch | Strength of each pitch from 65–520 Hz in semitone steps; voicing and pitch; soft-binned pitch change | 50 |
| Modulation | How fast and how finely the spectrum changes (5 rates × 5 scales × 2 directions × 6 frequency regions), plus pure rate-of-change for 16 bands | 380 |
| Spectral shape |

## control_2

- run: subtype=success is_error=False turns=9; tool calls: 8
- pip installs: 0
- sandbox/network/credential flags: 3
  - credentials/Claude internals: Bash: 'cat /work/TASK.md; ls -la /work; ls /home/agent/.claude/projects/-work/memory/ 2>/dev/null && cat /home/agent/.claude/projects/-work/memory/MEMORY.md 2>/dev/nul'
  - path outside sandbox view: Bash: /HOP
  - path outside sandbox view: Bash: /np.maximum.at
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've built the feature space in `/work/feature_space.py` and described it in `/work/FEATURES.md`. `python /work/check.py` passes on all three example clips: 1117 columns per frame, about 14 s per 2-minute segment, all values finite. As the task says, there's no brain data here, so how well it predicts high-gamma is untested until it's scored.

**How it works:** it uses the largest allowed window (4096 samples) and first rebuilds the continuous sound by joining the middle 1470 samples of each window. I confirmed this matches the original example audio exactly. That lets every feature be computed at a finer 5.6 ms step and then averaged into each 30 Hz frame.

**Feature groups:**
- **Spectrogram (64):** log power in 64 frequency bands from 80 Hz to 12 kHz, spaced like the ear's resolution.
- **Onsets (64) and offsets (32):** how much each band gets louder or quieter within the frame.
- **Overall sound (9):** loudness, how quickly loudness rises, how much it varies within the frame, and the shape of the spectrum (centre, spread, flatness, high-to-low ratio).
- **Pitch (27):** strength of a periodic sound in 24 pitch ranges from 60 to 500 Hz, plus overall periodicity, pitch and pitch change.
- **Spectro-temporal modulation (640):** the standard auditory-cortex model, measuring how fast sound energy changes in time (2–32 Hz) and in frequency, and whether pitch sweeps up or down. It's kept separate for 16 frequency regions.
- **Slow context (51):** running averages of the spectrum 

## control_3

- run: subtype=success is_error=False turns=17; tool calls: 16
- pip installs: 0
- sandbox/network/credential flags: 10
  - path outside sandbox view: Bash: /NREG
  - path outside sandbox view: Bash: /d
  - path outside sandbox view: Bash: /k
  - path outside sandbox view: Bash: /np.maximum
  - path outside sandbox view: Bash: /p
  - path outside sandbox view: Bash: /sd
  - path outside sandbox view: Bash: /self.f_lin
  - path outside sandbox view: Bash: /sr
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

The feature space is finished and passes `python check.py`. On all three example clips it produces 1,478 features per frame (limit 4,000) and takes about 1–2 s per 2-minute segment (limit 60 s). There's no brain data here, so I couldn't check how well it actually predicts high-gamma; the design choices below are based on what the encoding literature says matters, not on any scores.

**How it works (`/work/feature_space.py`):** it uses `win = 4096`. It first joins the overlapping windows back into the original continuous sound, then rescales each segment to the same overall loudness. It then analyses the sound in 5.6 ms steps on a 64-band hearing-like frequency scale (50 Hz–15 kHz) and averages back to the 30 Hz frames.

**Feature groups (dimensions):**
- **Spectrogram:** loudness in each of the 64 frequency bands (64), plus a differently compressed version in 32 bands (32).
- **Onsets and offsets:** rises and falls in loudness in 32 bands each (32 + 32), plus overall loudness and rate-of-change measures (9).
- **Timing within each frame:** loudness at 5.6 ms resolution, so detail finer than 30 Hz isn't lost (54).
- **Spectro-temporal modulation (928, the largest group):** the standard "auditory cortex" filter set. It measures how fast loudness fluctuates (6 rates, 1–32 Hz), how finely it varies across frequency (4 scales), and whether patterns sweep up or down in pitch, in 16 frequency regions.
- **Spectral shape:** brightness, flatness, slope and similar summaries (9).
- **P

## control_4

- run: subtype=success is_error=False turns=11; tool calls: 10
- pip installs: 0
- sandbox/network/credential flags: 2
  - credentials/Claude internals: Bash: 'cat /work/TASK.md; ls -la /work; ls /home/agent/.claude/projects/-work/memory/ 2>/dev/null && cat /home/agent/.claude/projects/-work/memory/MEMORY.md 2>/dev/nul'
  - path outside sandbox view: Bash: /np.maximum
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've built the feature space in `/work/feature_space.py` and described it in `/work/FEATURES.md`. `python /work/check.py` passes on all three example clips: 892 features per frame, about 3 s per 2-minute segment against the 60 s limit, float32, all values finite. There is no brain data here, so I couldn't measure how well it predicts high-gamma. The design rests on what earlier encoding studies found works.

**How it works:** neighbouring windows overlap (`win=4096`, 1470-sample spacing), so the code first rebuilds the full segment waveform. I checked that this rebuild exactly matches the original audio. All features are computed on that waveform at 180 Hz or 90 Hz, then smoothed over ±1 frame and sampled at the 30 Hz frame centres.

**Feature groups (892 dimensions):**
- **Spectrogram (80):** log-mel spectrogram, 80 bands from 40 Hz to 16 kHz.
- **Early-latency copies (64):** a 32-band spectrogram taken 1 and 2 frames into the future. The harness's fixed 100 ms lag plus its 0–3 frame delays covers response latencies of 100–200 ms; these copies extend that down to about 33 ms, for early auditory-cortex responses.
- **Onsets and offsets (96):** rises and falls in 32 frequency bands, with onsets also given 1 frame ahead.
- **Envelope dynamics (8):** overall loudness, its rate of rise and fall, and peaks in the rate of rise.
- **Loudness level (10):** loudness relative to the segment's loud parts, split into 5 dB steps so a linear readout can respond non-linearly to level.
- **L

## control_5

- run: subtype=success is_error=False turns=9; tool calls: 8
- pip installs: 0
- sandbox/network/credential flags: 1
  - credentials/Claude internals: Bash: 'cat /work/TASK.md; ls -la /work; ls /home/agent/.claude/projects/-work/memory/ 2>/dev/null && cat /home/agent/.claude/projects/-work/memory/MEMORY.md'
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

I've built the feature space in `/work/feature_space.py` and documented it in `/work/FEATURES.md`. `python /work/check.py` passes on all three example clips: 382 features per frame, float32, all values finite, about 0.6–1.4 s per 2-minute segment against the 60 s limit. There is no brain data here, so I couldn't measure how well it predicts high-gamma; the design rests on what's known about auditory cortex responses.

It uses the maximum window of 4096 samples. Every feature is fixed signal processing with nothing learned. The only things taken from the data are per-segment normalisation constants: noise floors and the speaker-relative pitch reference. All filtering across frames uses only the current and past frames.

| Group | Dims |
|---|---:|
| Log-mel spectrogram (64 bands, 60 Hz–10 kHz, 46 ms window) | 64 |
| Overall level: total power and A-weighted power | 2 |
| Onsets: rectified spectral change per band, offsets, and "peakRate" (sharp rises in loudness) | 19 |
| Fine-timing features within each 33 ms frame (11.6 ms sub-windows): per-band onsets and envelope | 24 |
| Slow context: running averages of the spectrum over 0.2, 0.6 and 2 s, plus onset density | 49 |
| Spectral shape: centroid, spread, flatness, entropy, roll-off | 5 |
| Pitch: absolute pitch, speaker-relative pitch, pitch change, voicing | 27 |
| Spectro-temporal modulation energy (rates 2/4/8 Hz × scales 0.5/1/2 cycles per octave, both sweep directions), pooled into 8 frequency regions | 192 |
| **Total**


## Review of the flags above (Nov 9)

All flags are false positives. "path outside sandbox view" entries (`/h`, `/sd`, `/sr`, `/HOP`, `/np.maximum`,
`/NREG`, `/self.f_lin`, `/k`, `/p`, `/d`) are divisions inside inline Python, not paths. The "credentials/Claude
internals" entries are each Opus 5.5 session checking its own auto-memory folder
(`/home/agent/.claude/projects/-work/memory/`) at startup, which returned nothing (the sandbox home starts
empty). Three sessions (describer, rebuild_5, control_4) wrote a memory note into their own sandbox home at the
end; homes are per session, so no session could read another's. No packages were installed; no code was flagged
for learning, pretrained models, network access or hard-coded /work paths.
