# Triple-N (NNN) — macaque visual-cortex spiking to the NSD-1000

Triple-N (NNN) is a large **macaque extracellular spiking** dataset recorded across
the visual cortex (early visual **EVC** + inferotemporal **IT**, with finer
functional patches) while monkeys viewed the **NSD shared-1000 natural-scene images**.
It is the image/spiking sibling of TVSD — same shape of data (a pre-averaged
`[units × images]` table), but on the **NSD** stimuli the rest of this repo already
uses, which makes it directly comparable to the human NSD fMRI encoding work here.

It is the live **`nnn`** preset in `harness/datasets.py` (`loader="nnn"`,
`harness/loaders/vision/nnn.py`).

## Chosen release

| Property | Value |
| --- | --- |
| Name | Triple-N / NNN (Li Yipeng et al.) |
| Docs | https://liyipeng-moon.github.io/Triple-N-Docs/ · code: https://github.com/liyipeng-moon/Triple-N |
| **Neural modality** | **spikes** (extracellular MUA/SUA, Utah/laminar arrays) |
| **Stimulus modality** | **image** — the **NSD shared-1000** natural scenes (+ 72 face/body/object localizers) |
| Brain areas | **EVC** (V1/V2/V4) + **IT** face/body/object/place/colour patches |
| Source | ScienceDB **DOI 10.57760/sciencedb.33556** (dataset `413ba7ddcb694bb2a534270caa90be36`) |
| Access | **login-gated** — every ScienceDB file/listing API returns `用户未登录` anonymously |
| Full release | ~2.7 TB raw (spikes/LFP/H5/session folders) across 1 Processed + 3 Raw sub-datasets |
| **Staged here** | **Processed only** — 90 `Processed_ses*.mat` + 5 helper files = **845 MB** |
| Local path | `/home/t-ymehta/Research/neuralset/data/nnn_processed/` (gitignored) |

### Why this dataset

- **Spiking on NSD stimuli.** The monkeys saw the *same* NSD shared-1000 images used
  by the human NSD fMRI preset, so an untrained model's predictivity is comparable
  across species/modalities within this repo. (TVSD is spiking too, but on THINGS.)
- **Huge, reliable unit count.** 47,503 recorded units; **35,494 pass the release's
  reliability cut (>0.4)** — 8,740 EVC + 21,214 IT — a rich, low-noise target set.
- **Compact entry point.** The Processed per-session tables are already trial-averaged
  (`response_best`), so we need none of the 2.7 TB raw spikes.

## Exact download steps

ScienceDB is **login-gated**; there is no anonymous wget/API (verified by
reverse-engineering their Nuxt API — `getFileTree`/`listSubset`/`details` all reject
anonymous calls). The one manual step is grabbing the direct-URL manifest from a
logged-in browser; the tokenized URLs then download headlessly:

1. Log into https://www.scidb.cn/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36
   and use **Download → "all URLs"** to obtain the manifest of direct download links.
2. Pull **Processed only** using the curated `manifest_processed.txt` already staged
   under `data/nnn_processed/` (the tokenized direct URLs download headlessly):

```bash
cd ~/Research/neuralset/data/nnn_processed
wget --content-disposition -c -i manifest_processed.txt -P .
```

This stages 90 `Processed_ses*.mat` plus the 5 helpers below.

```
nnn_processed/
  Processed_ses01_240629_M1_2.mat … Processed_ses90_251012_M4_6.mat   (90 sessions, ~6 MB each)
  img_pool.mat        # the 1072 stimulus images in-memory (227×227×3 uint8); cols 1..1000 = NSD
  StimuliNNN.zip      # same images as 0001.bmp..1000.bmp + MFOB001..072.bmp (not needed by loader)
  exclude_area.xls    # per-session shank y-ranges -> patch label (AREALABEL) + stream Area (EVC|IT)
  AreaXYZ.xlsx        # registered template coordinates per area
  ClusInfo.mat        # cross-session cluster info
  manifest_processed.txt   # the curated Processed-only download list
```

## Processed file schema (verified)

Each `Processed_ses*.mat` (scipy `loadmat`, v5/v7) — example session 01, 452 units:

| Field | Shape | Meaning |
| --- | --- | --- |
| `response_best` | **(units, 1072)** | mean firing rate per image; **cols 1..1000 = NSD shared-1000**, 1001..1072 = face/body/object localizers |
| `response_basic` | (units, 1072) | alternative (fixed-window) response matrix |
| `reliability_best` | (units,) | split-half reliability across repeats — **good-unit cut = 0.4** |
| `pos` | (units,) | shank y-position (μm) → area via `exclude_area.xls` |
| `snr`, `snrmax` | (units,) | signal-to-noise per unit |
| `B_SI` / `F_SI` / `O_SI` | (units,) | body / face / object selectivity (d′) |
| `UnitType` | (units,) | unit class (single / MUA / non-somatic) |
| `mean_psth` | (units, 450) | PSTH −50…+400 ms |

Area labels are **not** in the Processed file — they come from `exclude_area.xls`
(`SesIdx, y1, y2, AREALABEL, Area`): a unit's area is the row of its session whose
`[y1, y2)` contains its `pos`. Coarse `Area ∈ {EVC, IT}`; 27 `AREALABEL` patches
(MB/AB body, MF/AF face, MO/AO object, LPP/PITP place, CLC/AMC colour, PF, V1/V2/V4).

## Harness interface (`loader="nnn"`, `load_nnn`)

Like TVSD, NNN is **not a NeuralSet study** — it ships a pre-averaged table, so it
takes the self-contained table path and reads the `.mat`/`.xls`/images directly (the
documented exception to "all data through NeuralSet"). The loader:

- **Y** = every reliable unit (`reliability_best > NNN_MIN_RELIABILITY`, default 0.4)
  **pooled across all 90 sessions**, optionally narrowed by `reward_roi` to one area;
  oriented `[1000 images × units]`.
- **X** = the 1000 NSD images read **straight from `img_pool.mat`** (RawPixels,
  `imsize`), so no NSD staging or id-mapping is needed — fully self-contained.
- **Split** = deterministic strided hold-out of the 1000 images (every
  `round(1/test_frac)`-th → test, i.e. 800/200 at the default `test_frac=0.2`);
  fixed and image-disjoint.
- **`reward_roi`** (set per run via `NEURO_ROI`) picks the target area as a column
  subset — the analog of TVSD's V1/V4/IT and NSD's region menu. Menu:
  `all` · `EVC` · `IT` · patch family `MB/AB/MF/AF/MO/AO/LPP/PITP/CLC/AMC/PF` ·
  exact label `V1/V2/V4/MB1/…`. **Default `IT`.**
- Assembled units are cached by `(reward_roi, reliability)`; X by `imsize`
  (`~/.cache/neuro-autoresearch/splits/nnn_*`). Bypass with `NEURO_NO_CACHE`.

## Run

```bash
NEURO_DATASET=nnn python -m harness.run                 # reward = IT (default)
NEURO_DATASET=nnn NEURO_ROI=EVC python -m harness.run    # early visual cortex
NEURO_DATASET=nnn NEURO_ROI=MF  python -m harness.run    # mid-face patch family
NNN_MIN_RELIABILITY=0.5 NEURO_DATASET=nnn NEURO_ROI=IT python -m harness.run
```

## Load verification (measured)

Current champion model, untrained, ridge readout (load ≈ 6 s after first build):

| `reward_roi` | units (reliable) | train/test images | train_corr | test_corr |
| --- | --- | --- | --- | --- |
| `EVC` | 8,740 | 800 / 200 | 0.579 | **0.280** |
| `IT` (default) | 21,214 | 800 / 200 | 0.505 | **0.233** |
| `LPP` (place) | 407 | 800 / 200 | 0.513 | 0.232 |

Each `python -m harness.run` appends one row to `results/NnnV1__<roi>.csv`.

> Stimulus = **image** (NSD-1000); neural = **macaque spikes** (EVC + IT). Alongside
> human NSD fMRI, macaque TVSD spiking (THINGS), and the text/MEG/iEEG presets, NNN
> adds dense primate spiking on the *same* NSD stimuli — a cross-species encoding bench.
