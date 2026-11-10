# Textbook-baseline audit

## Agent 1

- run: subtype=success is_error=False turns=12; tool calls: 11
- pip installs: 0
- sandbox/network/credential flags: 0
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

All checks pass. I built `/work/feature_space.py` with H = 301 interpretable acoustic features at ~0.9 s per 2-min segment, and documented the groups and literature in `/work/FEATURES.md`.

Feature groups (all standard, closed-form, fixed parameters):
- Log-mel cochleogram (64) + delta (64) + per-band peak rate (64) — Davis & Mermelstein 1980 / Slaney 1998 / Furui 1986 / Oganian & Chang 2019
- MFCC + Δ + ΔΔ (39) — classical speech features
- Spectral shape descriptors (10) and octave-band spectral contrast (7) — Peeters 2004 / Jiang 2002
- Zero-crossing rate, RMS linear/dB, A-weighted loudness, broadband peak rate — Rabiner & Schafer 1978 / IEC 61672-1
- Autocorrelation pitch / voicing / HNR (4) — Rabiner 1977 / Boersma 1993
- Chroma (12) — Fujishima 1999
- Temporal modulation filterbank at 1/2/4/8 Hz × 8 pooled bands (32) — Chi, Ru & Shamma 2005 / Elliott & Theunissen 2009

## Agent 2

- run: subtype=success is_error=False turns=26; tool calls: 25
- pip installs: 0
- sandbox/network/credential flags: 0
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 1
  - line 4: learned or fit from data. Output is a (T, H) float32 array.

**Agent summary:**

All checks pass:
- `H = 856` dims, well under the 4000 cap
- ~1.1 s per 2-min segment, well under the 30 s cap
- Finite float32 output of shape `(T, 856)`

`/work/feature_space.py` and `/work/FEATURES.md` are both written. The feature space covers 19 textbook feature groups: log-mel (80), cube-root mel (80), ERB cochleogram compressed and log (64 + 64), 1/3-octave bands (26), Bark log-power and specific loudness (24 + 24), MFCCs + Δ + ΔΔ (39), Peeters spectral descriptors (14), log-centroid/rolloffs (5), temporal LLDs (8), chroma (12), Jiang spectral contrast (18), autocorrelation pitch (4), Dau-style temporal modulation on ERB and mel (256 + 120), and Chi/Shamma spectrotemporal modulation (18).

## Agent 3

- run: subtype=success is_error=False turns=12; tool calls: 11
- pip installs: 0
- sandbox/network/credential flags: 0
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

All checks pass. The feature space is complete.

**Summary of deliverables:**

- `/work/feature_space.py` — `FeatureSpace` class with `win = 2048`, returns `(T, 650)` float32 features in ~1 s per 2-minute segment (well under the 30 s limit, and H = 650 ≪ 4000).
- `/work/FEATURES.md` — documents each feature group, its literature reference, parameters and dimensionality.

**Feature groups** (650 total dimensions):
- Log-mel spectrogram, 80 bands, 20–8000 Hz, Slaney-normalized, log-compressed (Davis & Mermelstein 1980; Slaney 1998)
- Delta and delta-delta log-mel (160, Furui 1986)
- MFCC 20 coefficients + deltas + delta-deltas (60)
- Autocorrelation F0, voicing probability, HNR, voiced-masked F0 (6, Rabiner 1977; de Cheveigné & Kawahara 2002)
- Spectral shape descriptors: centroid, spread, skew, kurtosis, flatness, entropy, three rolloffs, slope (10, Peeters 2004)
- Spectral flux (1, Masri 1996)
- Energy/loudness: RMS, log-RMS, A-weighted RMS, log A-RMS, peak, crest factor (6, IEC 61672-1)
- Zero-crossing rate (1, Rabiner & Schafer 1978)
- Temporal modulation filterbank per mel band at 1/2/4/8 Hz (320, Chi/Ru/Shamma 2005)
- Spectral modulation energy at 0.25–4 cyc/oct (5)
- Broadband onset envelope (1, Bello et al. 2005)

## Agent 4

- run: subtype=success is_error=False turns=17; tool calls: 16
- pip installs: 0
- sandbox/network/credential flags: 0
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

All checks pass: 549 features in ~0.5s per segment (budget: H ≤ 4000, time < 30s).

The feature space at `/work/feature_space.py` composes 11 standard groups from auditory neuroscience and speech processing literature:

- **Log-mel spectrogram** (80 bands) + Δ + ΔΔ
- **MFCC** C0..C19 + Δ + ΔΔ
- **Chroma** (12-bin pitch class profile)
- **Spectral shape**: centroid, bandwidth, rolloff, flatness, slope, flux, crest, entropy
- **Envelope / voicing**: log-RMS + Δ + ΔΔ, log-peak, ZCR, autocorrelation log-F0, pitch confidence, voicing flag
- **Mel subband temporal modulation** at 1/2/4/8/12 Hz (Shamma / Elhilali-style modulation bank)
- **Octave-band spectral contrast** (Jiang et al.)

All closed-form with fixed parameters. `/work/FEATURES.md` documents each group with literature citations, parameters, and dimension counts.

## Agent 5

- run: subtype=success is_error=False turns=13; tool calls: 12
- pip installs: 0
- sandbox/network/credential flags: 0
- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): 0

**Agent summary:**

All checks pass. The feature space is done.

**Summary**

- `/work/feature_space.py`: `FeatureSpace` with `win = 2048`, producing `H = 339` features per frame at ~0.5 s per 2-min segment (well under the 30 s limit).
- `/work/FEATURES.md`: documents each group with parameters, dimensionality, and canonical literature.

**Feature groups** (all closed-form, no learning): 64-band log-mel spectrogram, 32-band ERB-spaced log-gammatone cochleagram, 13 MFCCs + Δ + ΔΔ, 15 spectral shape descriptors (centroid, bandwidth, 5 rolloffs, flatness, slope, skew, excess kurtosis, entropy, crest, flux, ZCR), log-RMS and log-peak energy, 12 log-chroma, autocorrelation log-F0 and voicing, 6-band spectral contrast peaks+valleys (Jiang et al.), 32×5 temporal modulation energies at {0.5,1,2,4,8} Hz (Chi/Shamma, Kingsbury, McDermott & Simoncelli), and a mel-band onset envelope.

