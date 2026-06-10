# NSD subject-01 visual-cortex download

Selective, **single-subject** download of the Natural Scenes Dataset (NSD,
`Allen2022Massive`) — subject 1 only, visual cortex (`nsdgeneral` ROI in the
`func1pt8mm` preprocessed space the NeuralSet backend masks to). The full study
is 8 subjects / ~3400 timelines / multiple TB; **none of that was downloaded.**

## What was downloaded

| Property | Value |
| --- | --- |
| Study (catalog name) | `Allen2022MassiveSample` |
| Subject | subj01 **only** |
| Sessions / runs | session 01, runs 01–04 (4 fMRI timelines) |
| Space / ROI | `func1pt8mm`, `nsdgeneral` ROI mask applied on load |
| fMRI timeseries | 4 × `timeseries_session01_run0{1..4}.nii.gz` (~484 MB) |
| Design files | 4 × `design_session01_run0{1..4}.tsv` |
| ROI mask | `nsddata/.../subj01/func1pt8mm/roi/nsdgeneral.nii.gz` |
| Stimuli | 225 PNGs (the images shown in subj01 session 01 runs 01–04) |
| **Total size** | **622 MB** |
| Local path | `/home/t-ymehta/Research/neuralset/data/nsd_subj01` |

Data is fetched from the **public** S3 bucket `s3://natural-scenes-dataset/`
with `aws s3 cp --no-sign-request` (no credentials). Files land under
`<path>/Allen2022MassiveSample/` (NeuralSet appends the study name as a subdir).

## NSD terms acceptance (authorized)

The NeuralSet/neuralfetch gate is a **local marker file**, not a credential:

```
~/.config/neuralfetch/.nsd_tcs_accepted   # contents: "NSD Data Access Agreement accepted."
```

The user explicitly authorized accepting the NSD data-use terms; the marker was
created locally to satisfy the gate for the public S3 fetch. No personal data was
submitted to any web form.

## Why `Allen2022MassiveSample` (not the full `Allen2022Massive`)

`Allen2022MassiveSample` **is** the selective, subject-1 visual-cortex config: it
pulls exactly subj01 / func1pt8mm / nsdgeneral (session 1, runs 1–4) plus the 225
stimulus images those runs reference. Pointing the full `Allen2022Massive` at a
path would force the entire **73,000-image** stimulus set (`nsd_stimuli.hdf5`,
multi-GB) and iterate all 8 subjects — i.e. not lean and not single-subject. The
sample shares the identical on-disk layout (`nsddata_timeseries/ppdata/subj01/
func1pt8mm/...`, `nsddata/ppdata/subj01/.../nsdgeneral.nii.gz`), so it is the
correct lean handle for subj01 visual cortex.

## Verified loadable

Loaded through the harness data layer (`harness/data.py::load_split`) with the
config line below — events + fMRI segments materialize:

```
X_train (200, 768)   X_test (50, 768)      # 16×16×3 raw-pixel stimuli
Y_train (200, 31448) Y_test (50, 31448)    # nsdgeneral-masked fMRI
```

250 aligned stimulus↔fMRI segments (200 train / 50 test, `seed=0`).

## Ready-to-use harness config line

Edit `harness/config.py` → `DATASET = Dataset(...)` to point at this download:

```python
DATASET = Dataset(
    study="Allen2022MassiveSample",
    trigger="type=='Image'",
    neural="fmri",
    imsize=16,
    start=0.0,
    duration=3.0,
    download=False,
    path="/home/t-ymehta/Research/neuralset/data/nsd_subj01",
    test_frac=0.2,
    seed=0,
)
```

`download=False` because the data is already present at `path`; the loader reads
`<path>/Allen2022MassiveSample/`. The dataset directory lives outside the repo and
is gitignored — only this doc is tracked.
