# `analyze/` — run report builder

Turns autoresearch run outputs into a single self-contained, interactive,
white-background HTML report. Inspired by the `analyze/` layer in
[csinva/circuit-building-autoresearch](https://github.com/csinva).

## Usage

```bash
python analyze/build_report.py [--runs-dir runs] [--repo-root .] [--out analyze/report.html]
```

Open the resulting `analyze/report.html` in any browser — it has no external
dependencies (CSS, JS and plots are all inlined; plots are base64 PNGs rendered
with the matplotlib `Agg` backend).

## What it reads

- Per-study append-only CSVs at `runs/<tag>/results/<study>.csv` and the repo-root
  `results/<study>.csv`. The schema is the harness ledger:

  ```
  iteration,timestamp,study,model_name,idea,n_params,n_train,n_test,n_targets,
  test_corr,train_corr,gap,frac_above_0.1,best_so_far,seconds,status
  ```

  followed by any optional `corr_<slice>` columns.
- Optional `runs/<tag>/metadata.json` (e.g. `{"dataset","model","effort","note"}`)
  used to label a run; otherwise the report falls back to the tag and the `study`
  column.

## What it produces

- **Top summary** — every run's reward (`test_corr`) vs iteration, one line per run,
  faceted by dataset (`study`).
- **Per-run trajectories** — `test_corr` and `train_corr` vs iteration with a
  running-max line, labelled by tag + dataset/model/effort.
- **Attempts table** — interactive and sortable (click a header); the `idea` text is
  expandable so long ideas stay readable without breaking the layout.
- **Auto-generated writeup** — which ideas were tried and which improved
  `best_so_far`, derived from the CSV.

### Excluded rows

Rows whose `status` is not `ok`, or whose `idea` mentions training, are **flagged and
excluded** from the plots and the writeup — the proposed model is never trained, so
training-based attempts are not valid scores (parity with the reference, which
excludes trained models). Excluded rows are still listed in the table, highlighted,
so nothing is silently dropped.

## Test

```bash
python tests/test_build_report.py      # standalone
pytest tests/test_build_report.py      # or via pytest
```

The test builds a synthetic fixture (a couple of `runs/<tag>/results/<study>.csv`,
one with `metadata.json`, one with an excluded row) and asserts the HTML is produced
with the expected plots, table and labels.
