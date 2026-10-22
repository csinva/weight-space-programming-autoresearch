# Scientific contract and reproducibility

The model receives only flattened HWC RGB images in `[0, 1]`, resized to 128×128
by default. It returns a finite `(images, features)` matrix with fewer than 10,000
features. Parameters must be fixed, analytic or seeded random, with at most
100,000 materialized parameters. Model outputs must be independent of batch
composition. Only `model.py` evolves; the harness, datasets and protocol stay fixed.

**RSA search:** construct a Pearson correlation dissimilarity matrix for model
features and for each neural region, then compare their upper triangles using
average-rank Spearman correlation. No readout is fitted during search. Invalid,
constant or nonfinite results fail the attempt instead of receiving a reward.

LAION uses the shared test images with the 240 NSD-source stimuli excluded:
**1,252 images** in the current prepared release, across subjects sub-01, sub-03,
sub-05, sub-06 and sub-07. Its subject-private training images are not part of
this fixed search reward. Triple-N uses **all 1,000 images**, across M1–M5; its
source folds are combined because search fits no encoding readout. These are
search stimuli, not independent held-out tests after iterative optimization.

The primary anatomical regions match the source evaluator's default: early and
ventral visual streams. Average regions within each subject, then average
subjects equally. Triple-N M1/M4 have early and ventral coverage; M2/M3/M5 have
ventral coverage. Every subject must succeed, and its actual region scores are
retained. Source filters are fixed: ncsnr >0.2 for LAION/NSD, reliability >0.4 for
Triple-N, oracle >0.4 for TVSD. These are **raw RSA and raw Pearson encoding**;
old noise-normalized encoding scores are not comparable.

**Final encoding:** freeze the model selected solely by search RSA, then use the
paired dataset's official train/test image split. Center features with training
statistics and normalize their overall training scale; z-score neural targets
with training statistics. Each target chooses its ridge penalty using five-fold
training CV over the source's 20-alpha grid. Report held-out Pearson r, aggregated
by region then subject. NSD and TVSD never supply a search reward.

This is cross-dataset transfer, with known NSD-source LAION images excluded. It
does not claim a comprehensive pixel-level duplicate audit across releases.

## Runs and reproducibility

A run lives in `runs/<dataset>__<date>/` and contains `search.db`, immutable model
snapshots, proposer workspaces, evaluator logs and `env.json`. The manifest
records the protocol, image policy, subjects, data-file identities, source
revision, harness hash, numerical package versions, input resolution, parameter cap
and provider settings.
Resuming with different science or modified data/code is rejected; use a new
`--date`. Historical encoding runs cannot be resumed as RSA searches.

`test_corr` remains the queue's compatibility column name, but each current row
is explicitly tagged `metric=rsa`, `stage=search`, and the protocol version.
Per-subject feedback is retained. Final evaluation verifies the selected model's
SHA256 against the scored snapshot before it opens final data. File identities
use path, size and modification time; frozen model and harness identities use
SHA256. Provider logs retain their native usage events; unknown monetary cost is
stored as null rather than estimated from obsolete Copilot credits.

The existing source/network guards and agent instructions support reward
integrity; they are not a hostile-code security boundary. With permission skipping
enabled, provider workspace paths do not enforce filesystem isolation. Final-dataset
separation relies on the protocol, scoring interfaces and agent instructions;
there is no separate OS account or data-access service.
