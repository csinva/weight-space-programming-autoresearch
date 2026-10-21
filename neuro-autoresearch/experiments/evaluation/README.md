# Final cross-dataset evaluation

Select the champion by RSA in the completed source run, freeze its bytes, then
score the paired dataset using the copied looped-networks encoding machinery.

```bash
.venv/bin/python -m experiments.evaluation.final --run runs/laion__2026-10-21
.venv/bin/python -m experiments.evaluation.final --run runs/triplen__2026-10-21
```

LAION-fMRI search transfers to all eight NSD subjects; Triple-N search transfers
to both TVSD monkeys. The train/test splits come from each final dataset's
prepared source release. Ridge penalties are selected per neural target by
five-fold training CV. The scalar is held-out Pearson r, with equal region means
within subject, then equal subject means. Final scores do not enter `search.db`.

The output contains a frozen `model.py` and `evaluation.json`, under
`results/final/<protocol>/<model-hash>/`. It records the search selection, source
and model hashes, all subject/region scores, and exact stimulus/file/filter
provenance. Existing output files are not overwritten; choose `--output` to
write a separate artifact outside the source run.

`run.py` delegates to this CLI. `--model PATH --search-dataset laion|triplen`
scores an explicit model and records that it was not selected from an RSA run.

`legacy_run.py`, `legacy_datasets.py`, old cross-dataset matrices and plotting
modules preserve historical schemas and analytical code. Historical
noise-normalized encoding scores must not be compared to current raw RSA or raw
encoding scores. Reproducing a pre-migration experiment requires its original
repository revision and data format.
