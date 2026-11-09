# Scientific protocol

[README](../README.md) · [Operations](operations.md) · [Scientific protocol](protocol.md) · [Comparisons](comparisons.md)

| Track | Phase 1: search feedback | Phase 2: final evaluation |
| --- | --- | --- |
| fMRI | LAION-fMRI sub-01, 03, 05 on the search image half, RSA or encoding | LAION-fMRI sub-06, 07 on the held-out image half, encoding |
| fMRI | NSD subjects 1–4 on the search image half, RSA or encoding | NSD subjects 5–8 on the held-out image half, encoding |
| Spiking | Triple-N, all five monkeys (ventral) or M1 and M4 (early/visual), RSA or encoding | TVSD, both monkeys, encoding |

LAION and NSD each hold out subjects and images within the dataset. Their shared
images (seen by every subject) are split in half within each image source, by the
SHA-256 hash of the stimulus ID: LAION's 1,492 into 745 search and 747 held-out
images (LAION, THINGSplus, NSD and OOD sources), and NSD's 907 into 453 and 454.
NSD has 907 rather than 1,000 shared images because subjects 3, 4, 6 and 8 did not
complete all sessions; each cohort pairs two subjects who did with two who did not.
Encoding readouts are fit on each subject's own unique images only. Held-out
subjects, held-out images, TVSD responses and final artifacts must never guide
search, coordination, repair or model selection.
A high search score is predictive evidence, not a mechanistic explanation.

## Model and scoring rules

Models receive flattened HWC RGB images in `[0, 1]`, resized to 128×128 by default,
and return a finite `(images, features)` matrix with fewer than 10,000 features.
Parameters must be fixed, analytic or seeded random, with at most 100,000
materialized parameters. Outputs must be independent of batch composition.

**RSA:** compare the upper triangles of model and neural Pearson-correlation
dissimilarity matrices using average-rank Spearman correlation. No readout is
fitted. Constant or nonfinite primary results fail the candidate.

RSA passes the complete stimulus set to the model in one call. Within each
candidate evaluation, subjects with identical ordered image contents reuse the
validated features and model RDM ranks, including source-group ranks. Fixed neural
ranks are cached across candidates with response identity, mutation and subset
checks. Every subject and prescribed region is still scored separately. Models
must bound intermediate memory (internal chunking is allowed); encoding
retains its batched forwards.

- For RSA, LAION and NSD use the search half of the shared images for their search
  subjects. Subject-private training images are excluded.
- For RSA, Triple-N combines its source folds and uses all 1,000 images per
  subject: M1–M5 for ventral, or M1 and M4 for early/visual.
- The full selected stream mask determines each subject's reward; average these
  pooled-stream scores equally across all required subjects. Never average the
  subregion scores to obtain the reward. Missing pooled-stream coverage or a
  missing subject fails evaluation. Search diagnostics include only subregions
  within the selected stream; for `visual`, the early and ventral streams are
  diagnostics as well.
- Fixed source filters are ncsnr >0.2 for LAION/NSD, reliability >0.4 for Triple-N,
  and oracle >0.4 for TVSD.

Default `status`/`wait-for-score` feedback is a short labeled display: pooled score
and parent delta first, selected-stream subregion scores/deltas, then identity and
execution information. `--json` returns compact structured feedback with full
numeric precision. Failures show a specific reason and suggested action first;
bounded candidate traces are available through `--details`. Detailed subject
coverage and LAION RSA source-group diagnostics (`laion`, `thingsplus`, `ood`, `nsd`) are
available through `status <id> --details`. A subregion or source-group cohort score
requires all subjects; unavailable diagnostics do not change the pooled reward.
Repeatedly optimized search scores are not independent held-out confirmation.

**Encoding:** for LAION and NSD, fit each subject's readout on that subject's private
images and test on one half of the shared images: the search half during search,
the held-out half for the final headline score. Triple-N uses source fold 0. For
final evaluation, freeze the selected model before loading any held-out responses.
LAION and NSD final evaluation reports the held-out subjects on held-out images
as the headline and the three other subject × image cells as diagnostics. Center features and normalize
their overall scale using training data; z-score targets with training statistics.
Choose a ridge penalty per neural target using five-fold training CV over the
source's 20-alpha grid. Report test Pearson r for the pooled selected stream,
averaged equally over the full cohort. Search subregion values reuse the same
per-unit predictions; they do not add readout fits. Final evaluation retains the selected
region and always uses encoding, regardless of the search metric.
Search encoding extracts features in bounded 1,024-image batches and validates
feature shape, finiteness and parameter limits after each batch, before continuing.
Isolated models receive CPU-staged inputs; only the image-only worker uploads them
to the GPU. One worker loads the model once and handles all batches and subjects
for that candidate. Encoding search does not run singleton or shuffled-context
verification probes. The worker is closed when that candidate finishes or fails.
Within one candidate's search evaluation, identical ordered test images reuse a
validated raw feature matrix (bounded to 128 MiB). Private training features and
subject-specific readouts are not shared. The cache is discarded on completion or
failure; final evaluation retains its existing batching and no feature cache.
These execution changes leave image policies, cohorts, ridge CV and rewards intact.
Evaluation-code identity still changes, so use a new run after updating the evaluator.

Both RSA and encoding are raw scores, without noise-ceiling normalization.
No comprehensive pixel-level duplicate audit across releases is claimed. The
held-out subject and image policy uses protocol v5; start a new run rather than
resuming a run scored under a previous policy.

### LAION region definitions

`early` uses **laion-EVC**: retinotopic V1–V3 intersected with the visually responsive
laion-general mask. `ventral` uses **laion-ventral**: the broad anatomical ventral
occipitotemporal sector below the inferior occipital–inferior temporal gyrus border.
It is not restricted to FFA/PPA or a pure high-level functional area. See the
[official ROI definitions](https://laion-fmri.hebartlab.com/rois.html).

The prepared response coordinates were checked against the source binary masks
and segmentation labels (EVC=1, ventral=4) for all five subjects. After the fixed
ncsnr >0.2 filter, retained voxel counts are:

| Subject | early | ventral |
| --- | ---: | ---: |
| sub-01 | 2,139 | 2,722 |
| sub-03 | 2,916 | 2,518 |
| sub-05 | 3,077 | 2,374 |
| sub-06 | 2,943 | 2,399 |
| sub-07 | 2,500 | 2,528 |

All coordinates matched their masks, with no early/ventral overlap. These are the
source sector masks, not a reconstruction from separately voxelized named areas.

Early diagnostics report **V1, V2, V3**; ventral diagnostics report **hV4, FFA, PPA**;
`visual` reports both streams and all six areas. The union is the concatenation of
both stream blocks; loading fails if the streams share a unit.
Each diagnostic mask is the intersection of that named area's retained coordinates
with the selected stream's retained coordinates. These diagnostic areas can overlap
and do not necessarily exhaust the stream. The pooled response matrix is stored
once, with subregion masks selecting columns from it; adding diagnostics does not
duplicate voxels or rerun model forwards. Triple-N ventral diagnostics use the
available **IT** units, intersected by their source unit identifiers.

## Implementation provenance

RSA and encoding kernels are implemented directly in `src/neuro_eval/rsa.py` and
`src/neuro_eval/encoding.py`; source-file loading is implemented in `src/neuro_eval/readers.py`.
These implementations originate from `looped-networks` revision
`3ee575baeceb1ad784102162a051b7444182fdaa` (`looped/metrics/{rsa,encoding}.py`
and `looped/data/{nsd,laion,triplen,tvsd,stimuli}.py`), imported October 21, 2026.
Active numerical algorithms, source filters and image decoding are preserved;
unused source helpers are omitted. Local adaptations resolve repository `.env`
and shared-storage paths, cache prepared files by identity, and identify NSD
reliability volumes from response metadata. There is no dependency on the original
checkout or a second copy of the implementations. `SOURCE_REVISION` in
`src/neuro_eval/benchmarks.py` and the evaluation-code hash identify their provenance
in every run and final artifact.
