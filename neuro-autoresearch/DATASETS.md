# DATASETS

Context the autoresearch agent reads **every run**. The active dataset is chosen
by `NEURO_DATASET` (default `text`); the harness loads it and you propose a
never-trained model in `evolve/model.py` whose ridge-readout `test_corr` is the
reward. Three presets (`nsd`, `text`, `meg`) live in `harness/config.py`.

For each, the model sees the stimulus `X (N, D_in)` and must emit activations
`(N, D_out)`; the fixed harness fits a ridge readout from those activations to the
neural target `Y` and scores held-out **mean per-target Pearson r = `test_corr`**.

---

## `nsd` — Natural Scenes Dataset, subject 1 (`NsdSubj01`)

- **Modalities:** stimulus = **image** · neural = **fMRI**.
- **Stimulus X:** the shown image, resized to `imsize×imsize` (default 16) and
  flattened to `[0,1]` RGB → **`16·16·3 = 768` dims** (`RawPixels`).
- **Target Y:** the subject's fMRI response over the **`nsdgeneral`** mask
  (~15k voxels; constant-on-train columns dropped). `n_targets` is logged per run.
- **Split:** repeats collapsed to **one sample per unique image** (neural averaged
  over the ~3 presentations), then the **canonical NSD shared-1000** images are the
  held-out test — **image-disjoint** (asserted), **~9000 train / ~1000 test**.
- **Sliceable metadata (USE THESE for `LOG_SLICES`):** NSD voxels carry ROI labels,
  so you can track per-region predictivity. ROI slice menu:
  **`V1, V2, V3, hV4, FFA` (floc-faces), `PPA` (floc-places)**. Pick a stable subset
  once (e.g. `["V1","V2","V3","hV4","FFA","PPA"]`) to see early-visual vs. high-level
  ventral regions separately. Separately, `Dataset.roi` (one config line) can
  restrict the **whole eval** to one region (`V1`/`V2`/`V3`/`hV4`/`FFA`).
- **Reward:** held-out mean per-target Pearson r (`test_corr`) over the kept voxels.

## `text` — Le Petit Prince narrative listening (`Li2022PetitSubj`)

- **Modalities:** stimulus = **audio** · neural = **fMRI**. (Keyed `text`: spoken
  story; each **Word** is the trigger.)
- **Stimulus X:** the **Mel spectrum of the audio at the word**, time-averaged over a
  short stimulus window (`stim_start=0.0`/`stim_duration=0.5`) → a per-word Mel-band
  vector (`MelSpectrum`, time-averaged). The neural window is HRF-offset
  (`start=2.0`/`duration=4.0`); stimulus and neural windows are **decoupled** so the
  BOLD delay never leaks into X.
- **Target Y:** the fMRI volume in the HRF-delayed window (whole-brain voxels;
  constant-on-train columns dropped). `n_targets` is logged per run.
- **Split:** **contiguous temporal hold-out** — the latest `test_frac` (0.2) of words
  by onset are the test set (no random split: time-adjacent words share signal).
- **Sliceable metadata:** **none yet** — leave `LOG_SLICES = []` (only the CORE
  columns are written).
- **Reward:** held-out mean per-target Pearson r (`test_corr`).

## `meg` — Le Petit Prince MEG, listen (`Bel2026PetitListenSample`)

- **Modalities:** stimulus = **audio** · neural = **MEG** (the loop's first
  electrophysiology). Each spoken **Word** triggers a 0.5 s epoch.
- **Stimulus X:** the **Mel spectrum** of the audio in that same `[0, 0.5] s` window,
  time-averaged (`MelSpectrum`). MEG is near-instant, so stimulus and neural windows
  **coincide** (no decoupling).
- **Target Y:** the MEG sensor responses over the epoch (sensors × time frames at
  100 Hz; constant-on-train columns dropped). `n_targets` is logged per run.
- **Split:** **contiguous temporal hold-out** (latest `test_frac` by onset), as for
  `text`.
- **Sliceable metadata:** **none yet** — leave `LOG_SLICES = []`.
- **Reward:** held-out mean per-target Pearson r (`test_corr`).

---

## Hard rules (all datasets)

1. **Edit only `evolve/model.py`.** `harness/` is fixed (the one allowed exception is
   the `DATASET = ...` / `NEURO_DATASET` selection in `harness/config.py`).
2. **Never train the model** — weights are written directly in code: no optimizer, no
   backprop, no gradient descent. Only the ridge readout in `harness/eval.py` is fit,
   and that *is* the evaluation.
3. The model is a plain callable `X (N, D_in) -> activations (N, D_out)` returning
   **raw activations** (the readout does the fitting). Prefer modality-agnostic
   models; image-only models (e.g. spatial pyramids) only apply to `nsd`.
4. The reward is the held-out `test_corr`. `train_corr` is reported too: high train +
   low test = an overfit readout, not progress. Small `test_corr` diffs are noisy.
