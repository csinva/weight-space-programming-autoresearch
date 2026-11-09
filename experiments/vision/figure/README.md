# Reproduce the fMRI figure

This folder contains the paper's five-subject LAION model, the numerical results
for the main fMRI figure, and the code to render or recompute its panels. It uses
the existing `neuro_eval/` and `baselines/` modules. No search agents, coordinator,
dashboard, provider login or API credit are needed.

## Render the published results

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run experiments/vision/figure/make_figure.py
```

The script installs its two pinned plotting dependencies in an isolated script
environment. It does not require `uv sync`, PyTorch, a GPU or neural datasets.
It writes `figure/output/fmri_figure.pdf`, `.svg` and `.png`, individual B–E panels,
and the supplementary attribution heatmaps. Here and below, `figure/` means
`experiments/vision/figure/`. All generated outputs and caches are ignored by Git.
Panel A uses the included dataset illustration; it is not recomputed from neural data.

The included JSON files contain the actual plotted scores and their provenance.
Missing or mock panel data cause an error rather than a partial figure. An output
manifest records input SHA-256 hashes, plotting versions and source provenance.
Historical paths inside the JSON are metadata, not required local files.

Check the included scores and model without installing any scientific packages:

```bash
uv run --no-project python experiments/vision/figure/verification/check_results.py
```

This verifies the running-best trajectory, subject means, image-ID hash, signed
attribution shares and frozen model hash. It does not rerun neural evaluations.

## What each panel measures

| Panel | Computation | Script |
| --- | --- | --- |
| B | 600 scored LAION candidates and the running pooled-best model | `b_trajectory.py` |
| C | Frozen model, eight interpretable baselines and AlexNet conv4 on NSD | `c_nsd.py` |
| D | Exact Shapley contributions of computational families beyond edges | `d_families.py` |
| E | Exact Shapley contributions of five texture operations beyond edges | `e_texture.py` |

The frozen model is `erjpu`, with 9,996 features and SHA-256
`cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633`.
Its feature extractor uses analytic filters and geometric/statistical operations.
The ridge readout is fit to neural responses separately for each subject.

B, D and E reproduce the historical `looped-cross-dataset-v4` protocol: LAION
subjects 01, 03, 05, 06 and 07, with readouts fit on unique images and evaluated
on all shared images. This differs from the current search harness's subject/image
holdouts. C uses NSD subjects 01, 02, 05 and 07, with 998 shared test images after
pixel-matched LAION overlaps are excluded from both training and testing.
AlexNet conv4 was selected on LAION before NSD evaluation.

D and E use signed squared correlations, `sign(r) * r**2`. Negative contributions
are retained. E is a separate attribution within the edges-plus-texture model;
it is not a subdivision of D's texture contribution. The historical results are
for paper reproduction, not feedback for a new search.

## Recompute scores from prepared data

Rendering above is fully self-contained. Recomputing C–E requires the prepared
neural-response files, stimulus images and NSD reliability volumes described
below. These large files and a raw-release preparation pipeline are not included.
Downloading the public raw datasets alone does not produce the required pickles.

Install the repository dependencies and optional baseline implementations:

```bash
uv sync --extra baselines
```

Set the dataset locations explicitly in your shell:

```bash
export LAION_DATA_DIR=/path/to/laion/processed
export LAION_FMRI_DIR=/path/to/laion/release
export NSD_DATA_DIR=/path/to/nsd/processed
export NSD_ROOT=/path/to/nsd/release
export NEURO_ALEXNET_WEIGHTS=/path/to/alexnet-owt-7be5be79.pth
```

| Input | Required content |
| --- | --- |
| `$LAION_DATA_DIR/laion_data.pkl` | Prepared responses, ROI/voxel coordinates, ncsnr, stimulus metadata and shared-image IDs; approximately 1.69 GB |
| `$LAION_FMRI_DIR/stimuli/images/task-images_stimuli.h5` | Encoded LAION stimulus images |
| `$NSD_DATA_DIR/nsd_data.pkl` | Prepared responses with voxel coordinates and shared-image IDs; approximately 3.48 GB |
| `$NSD_ROOT/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5` | NSD images in `imgBrick` |
| `$NSD_ROOT/nsddata_betas/ppdata/subj*/func*/betas_*/ncsnr.nii.gz` | Official voxel reliability volumes matching the response metadata |
| `$NEURO_ALEXNET_WEIGHTS` | ImageNet AlexNet checkpoint `alexnet-owt-7be5be79.pth` |

`NSD_STIMULI_HDF5_PATH` and `NSD_BETAS_DIR` override the two NSD release paths.
`NSD_DATA_FILE` selects a different prepared pickle, but the ordinary
`nsd_data.pkl` is sufficient for panel C. See
[`../neuro_eval/readers.py`](../neuro_eval/readers.py) for the exact prepared-data
schema and source ROI/reliability filters. Weights default to the standard
PyTorch hub cache when `NEURO_ALEXNET_WEIGHTS` is unset; no automatic download runs.

Run these commands from the repository root:

```bash
# NSD transfer: computes features and fits subject-specific readouts.
uv run experiments/vision/figure/c_nsd.py --device cuda

# LAION family attribution: computes features and fits all family subsets.
uv run experiments/vision/figure/d_families.py --device cuda

# Texture attribution: reuses the feature cache created by the previous command.
uv run experiments/vision/figure/e_texture.py --device cuda
```

These are expensive evaluations, especially the exact subset analyses. A CUDA GPU
is recommended; `--device cpu` is available. They cache features and scores under
`figure/output/data/` and resume from those caches. Use a fresh output directory
when changing data, model code or numerical settings. For a smaller NSD check,
`c_nsd.py --only erjpu,gabor --device cuda` evaluates just those two models; a full
panel C requires all ten models.

The scripts preserve the existing evaluator: feature centering, global scaling,
five-fold ridge selection per voxel, voxelwise Pearson correlation and equal
subject averaging. They do not run or modify the search harness.

Panel B records a stochastic search history; rerunning a new search will not
recreate those 600 candidates. Use the bundled `results/b.json`. If you have the
original historical run directory, its `search.db` and `env.json` can regenerate B:

```bash
uv run --no-project python experiments/vision/figure/b_trajectory.py \
  --run /path/to/laion__visual__encoding__2026-11-05
```

After recomputing C–E, combine them with the recorded trajectory and render:

```bash
cp experiments/vision/figure/results/b.json experiments/vision/figure/output/data/b.json
uv run experiments/vision/figure/make_figure.py \
  --data experiments/vision/figure/output/data \
  --out experiments/vision/figure/output/recomputed
```

The checked-in aggregates remain unchanged. Only the plotting dependencies are
pinned independently; score recomputation uses the repository's `uv.lock`.
Floating-point results can vary with hardware and numerical library versions.
