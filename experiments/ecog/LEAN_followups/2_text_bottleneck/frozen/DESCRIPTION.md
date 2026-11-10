# 3,852 dims, win=4096

LP_τ = causal one-pole lowpass over frames (α = 1/(1+τ)); "dyn" = x − LP_τ, τ ∈ {4.5, 15, 60}. All concatenated.

**Cochleagrams.** Hann power spectra, centred 1024/2048/4096-sample sub-windows, weighted by 20 fourth-order gammatone power responses (ERB-spaced, 50–8000 Hz) → log. Add E − LP_τ for τ ∈ {1.5, 4.5, 15, 60, 150}; tonic LP_60 and LP_150 of log and cube-root energies, acausal LP_60; past and future differences over 3/6/12 frames, plus rectified halves.

**Spectrotemporal modulation.** 2-D Gabor quadrature energy on the 4096 cochleagram (17 rate/scale pairs: 0–12 Hz, 0–±0.5 cycles/channel). On a 40-channel cochleagram: rates 0.5–12 Hz × scales {0, ±0.25, ±0.5}, cosine, sine, magnitude averaged over frequency thirds; dyn, acausal residuals cross-rate energy products.

**Descriptors** (4096 spectrum; each with dyn): 8-harmonic log-comb pitch salience (11 F0s, 75–450 Hz), best F0/strength, ACF HNR, cepstral peak prominence, spectral moments/flatness/rolloff, octave and channel-group spectral contrast, softmax formants, spectral flux, voicing-gated cochleagram. Plus 2/4/6/10 Hz temporal-Gabor envelopes of these scalars.

**Interactions.** Key scalars, their dyn, pairwise products (phonetic band contrasts × pitch/loudness/flux/formants), z-scored within the segment and expanded as [z]₊, [−z]₊, z³.

**Within-window scattering.** Morlet scattering (Q=4 then Q=1) on 9 channels plus instantaneous frequency/coherence.
