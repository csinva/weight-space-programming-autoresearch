# DATASETS

Context the autoresearch agent reads **every run**. The active dataset is chosen
by `NEURO_DATASET` (default `text`); the harness loads it and you propose a
never-trained model in `model.py` whose ridge-readout `test_corr` is the
reward. Presets (`nsd`, `text`, `meg`, `ieeg`, `tvsd`, `nnn`) live in `harness/datasets.py`.

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

## `ieeg` — "Podcast" ECoG, natural story listening (`Zada2025Podcast`)

- **Modalities:** stimulus = **audio** · neural = **iEEG/ECoG** (intracranial; the
  loop's first intracranial electrophysiology). 9 epilepsy patients, 1,330 ECoG
  electrodes total; restricted to ONE subject (default sub-01) because each patient has
  a different electrode set. Each spoken **Word** (from the aligned transcript) triggers
  a 0.5 s epoch.
- **Stimulus X:** the **Mel spectrum** of the podcast audio in that `[0, 0.5] s` window,
  time-averaged (`MelSpectrum`). iEEG is near-instant, so stimulus and neural windows
  **coincide** (no HRF decoupling) — exactly as for `meg`.
- **Target Y:** the subject's ECoG electrode responses over the epoch (ECOG channels at
  100 Hz; ECG/MISC channels dropped, bad channels marked from `channels.tsv`;
  constant-on-train columns dropped). `n_targets` logged per run.
- **Split:** **contiguous temporal hold-out** (latest `test_frac` of words by onset), as
  for `text`/`meg`.
- **Sliceable metadata:** **none yet** — leave `LOG_SLICES = []`.
- **Reward:** held-out mean per-target Pearson r (`test_corr`).

## `tvsd` — THINGS Ventral-stream Spiking Dataset, macaque (`TvsdMonkeyF`/`TvsdMonkeyN`)

- **Modalities:** stimulus = **image** · neural = **macaque spiking (MUA)** — the loop's
  first *electrophysiology in monkey* and first non-human, non-fMRI/MEG target. Each
  THINGS image shown evokes a response; the released data is already a **pre-averaged
  `[electrodes × stimuli]` normalized-MUA table** (no timeline to segment), so it uses the
  `loader="table"` path, not the Segmenter.
- **Stimulus X:** the shown **THINGS image**, resized to `imsize×imsize` (default 16) and
  flattened to `[0,1]` RGB → **`16·16·3 = 768` dims** (`RawPixels`, identical to `nsd`, so
  image models port over directly; raise `imsize` for more high-level detail).
- **Target Y:** each electrode's **time-windowed normalized MUA** (one value per electrode
  per image), for the chosen cortical **area**. 1024 electrodes/monkey across **V1, V4, IT**;
  per-area counts — monkey **F**: V1 512 · IT 320 · V4 192; monkey **N**: V1 512 · V4 256 ·
  IT 256. Constant-on-train electrodes dropped.
- **Split:** the dataset's **own** fixed split — **22,248 train images / 100 test images**
  (the test images are repeated many times; **image-disjoint** by construction). No temporal
  or shared-1000 logic applies.
- **Reward region (`NEURO_ROI`):** the scalar reward is the mean `test_corr` over ONE area,
  chosen per run with `NEURO_ROI` ∈ **`V1` / `V4` / `IT`** (default `IT`). Unlike NSD, the
  readout fits **only** that area's electrodes (per-target ridge ⇒ identical to fit-all);
  there is **no `corr_<area>` panel** — run one loop per area. Pick the **monkey** with
  `NEURO_SUBJECT` (`F`/`N`); each `(monkey, area)` writes its own
  `results/TvsdMonkey<F|N>__<area>.csv`.
- **Noise ceiling:** the release ships an `oracle` (split-half) correlation per electrode —
  the explainable-variance ceiling is high (mean ≈ 0.55–0.63), so strong `test_corr` is
  attainable. (No trained-AlexNet baseline is computed for `tvsd` yet.)
- **Stimuli:** the THINGS images are **gated** (license + ~12 GB) and **not bundled** with
  the spiking data — they must be staged once (see `data._tvsd_images_root`) before running.
- **Reward:** held-out mean per-electrode Pearson r (`test_corr`) over the chosen area.

## `nnn` — Triple-N macaque visual-cortex spiking to the NSD-1000 (`NnnV1`)

- **Modalities:** stimulus = **image** · neural = **macaque spiking** — the spiking
  sibling of `tvsd`, but on the **same NSD shared-1000** natural scenes the `nsd` fMRI
  preset uses, so an untrained model's predictivity is directly comparable across
  species/modalities within this repo. Like `tvsd` the release is a **pre-averaged
  `[units × images]` table** (`response_best`), so it uses the `loader="table"` path
  (`data._load_nnn`), not the Segmenter.
- **Stimulus X:** the shown **NSD-1000 image**, read straight from the bundled
  `img_pool.mat`, resized to `imsize×imsize` (default 16) and flattened to `[0,1]` RGB
  → **`16·16·3 = 768` dims** (`RawPixels`, identical to `nsd`/`tvsd`, so image models
  port over directly; raise `imsize` for more high-level detail). **Self-contained** —
  no NSD staging or id-mapping needed (contrast `tvsd`, whose images are gated).
- **Target Y:** each **reliable unit's** mean firing rate (one value per unit per image),
  **pooled across all 90 sessions** and narrowed by the reward area. Units kept if
  `reliability_best > NNN_MIN_RELIABILITY` (default 0.4): **35,494 of 47,503 pass**
  (EVC 8,740 · IT 21,214). Constant-on-train units dropped.
- **Split:** deterministic **strided hold-out of the 1000 images** — every
  `round(1/test_frac)`-th image → test (**800 train / 200 test** at the default
  `test_frac=0.2`); fixed and **image-disjoint**. No shared-1000/temporal logic applies.
- **Reward region (`NEURO_ROI`):** the scalar reward is the mean `test_corr` over ONE
  area, chosen per run. Menu: `all` · stream `EVC` / `IT` · patch family
  `MB/AB/MF/AF/MO/AO/LPP/PITP/CLC/AMC/PF` · exact label `V1/V2/V4/MB1/…` (default `IT`).
  Area comes from each unit's shank position via `exclude_area.xls`. Like `tvsd`, the
  readout fits **only** that area's units (per-target ridge ⇒ identical to fit-all) and
  there is **no `corr_<area>` panel** — run one loop per area; each writes its own
  `results/NnnV1__<roi>.csv`.
- **Baselines (measured, untrained champion):** EVC 0.280 (8,740 units) · IT 0.233
  (21,214) · LPP 0.232 (407); load ≈ 0.6 s after first cache build. (No trained-AlexNet
  baseline is computed for `nnn` yet.)
- **Reward:** held-out mean per-unit Pearson r (`test_corr`) over the chosen area.

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
