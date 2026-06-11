# DATASETS

Context the autoresearch agent reads **every run**. The active dataset is chosen
by `NEURO_DATASET` (default `text`); the harness loads it and you propose a
never-trained model in `model.py` whose ridge-readout `test_corr` is the
reward. Three presets (`nsd`, `text`, `meg`) live in `harness/datasets.py`.

Each run lives in its own isolated folder (`python scripts/setup_run.py <tag>`,
then `cd runs/<tag>`): `harness/` is symlinked there (FIXED), and you edit the
local `model.py`/`experiment.py` while results append to the run-local
`results/<study>.csv` (see `prompts/program.md`).

For each, the model sees the stimulus `X (N, D_in)` and must emit activations
`(N, D_out)`; the fixed harness fits a ridge readout from those activations to the
neural target `Y` and scores held-out **mean per-target Pearson r = `test_corr`**.

---

## `nsd` — Natural Scenes Dataset, subject 1 (`NsdSubj01`)

- **Modalities:** stimulus = **image** · neural = **fMRI**.
- **Stimulus X:** the shown image, resized to `imsize×imsize` (default 16) and
  flattened to `[0,1]` RGB → **`16·16·3 = 768` dims** (`RawPixels`).
- **Target Y:** the subject's **HRF-peak fMRI response** over the **`nsdgeneral`** mask
  (15724 voxels). The BOLD response peaks ~5 s after onset (TR 1.333 s), so the window
  is shifted forward (`neural_offset=5.0`) and the peak TRs in a 4 s window are
  averaged into **one value per voxel** (`neural_aggregation="mean"`) — a denoised,
  betas-like response. (With the old 0–3 s baseline window even a trained AlexNet
  scored ~0.006; this HRF-aligned response lifts that to ~0.32.)
- **Split:** repeats collapsed to **one sample per unique image** (neural averaged
  over the ~3 presentations), then the **canonical NSD shared-1000** images are the
  held-out test — **image-disjoint** (asserted), **~9000 train / ~1000 test**.
- **Reward region (`NEURO_ROI`):** the scalar reward is the mean `test_corr` over ONE
  brain region, chosen per run with `NEURO_ROI` (default `ventral`). The readout still
  fits **all** nsdgeneral voxels and **every** region's predictivity is logged as a
  `corr_<name>` column — only the headline reward narrows. Run one loop per region;
  each writes its own `results/<study>__<roi>.csv`, so per-region hill-climbs never mix.
- **Fixed region menu** (the harness logs all of these automatically — you do NOT
  declare them): famous regions **`V1, V2, V3, hV4, FFA, PPA, EBA, VWFA`** + visual
  streams **`early, ventral`**.
- **Trained-AlexNet reference baseline (the bar to beat):** a frozen
  ImageNet-pretrained AlexNet (fc7, 4096-d) read out with THIS exact ridge scores, per
  region — `V1 0.342 · V2 0.299 · V3 0.321 · hV4 0.289 · FFA 0.396 · PPA 0.403 ·
  EBA 0.436 · VWFA 0.321 · ventral 0.284 · overall 0.324`. This trained network — not the
  identity floor — is the baseline your never-trained model aims to match or beat for its
  reward region.
- **Reward:** held-out mean per-target Pearson r (`test_corr`) over the reward region.

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

1. **Edit only `model.py`** (the model) and **`experiment.py`** (the run-config).
   `harness/` is fixed (the one allowed exception is the `NEURO_DATASET` dataset
   selection backed by `harness/datasets.py`).
2. **Never train the model** — weights are written directly in code: no optimizer, no
   backprop, no gradient descent. Only the ridge readout in `harness/reward.py` is fit,
   and that *is* the evaluation.
3. The model is a plain callable `X (N, D_in) -> activations (N, D_out)` returning
   **raw activations** (the readout does the fitting). Prefer modality-agnostic
   models; image-only models (e.g. spatial pyramids) only apply to `nsd`.
4. The reward is the held-out `test_corr`. `train_corr` is reported too: high train +
   low test = an overfit readout, not progress. Small `test_corr` diffs are noisy.
