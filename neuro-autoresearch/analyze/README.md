# `analyze/` — run report builder

Turns autoresearch run outputs into a single self-contained, interactive,
white-background HTML report. Inspired by the `analyze/` layer in
[csinva/circuit-building-autoresearch](https://github.com/csinva).

## Usage

```bash
python analyze/build_report.py [--runs-dir runs] [--repo-root .] [--out analyze/report.html]
python analyze/build_report.py --runs-only --out docs/index.html   # the hosted page
```

Open the resulting HTML in any browser — it has no external dependencies (CSS, JS and
plots are all inlined; plots are base64 PNGs rendered with the matplotlib `Agg`
backend). `--runs-only` ignores the stray repo-root `results/` ledger so the page shows
only the per-run search under `runs/`.

## What it reads

- Per-study append-only CSVs at `runs/<tag>/results/<study>.csv` and the repo-root
  `results/<study>.csv`. The schema is the harness ledger:

  ```
  iteration,timestamp,study,model_name,idea,summary,n_params,n_train,n_test,n_targets,
  test_corr,train_corr,gap,best_so_far,seconds,status
  ```

  followed by any optional `corr_<slice>` columns.
- Optional `runs/<tag>/metadata.json` (e.g. `{"dataset","model","effort","note"}`)
  used to label a run; otherwise the report falls back to the tag and the `study`
  column.

## What it produces

- **What the search found** — a written synthesis: the early-visual front-end the search
  rediscovered in every region (Gabor complex-cell energy + colour opponency + log
  compression + divisive normalization), the region-specific elaborations, and the key
  findings (V1 is most predictable; every region's best model still predicts V1 best;
  parity with frozen AlexNet on early areas).
- **Cross-region predictivity** — a heatmap of each run's best model scored against every
  NSD region (`corr_<region>`), exposing how V1-like the learned features remain.
- **Model mechanics** — one card per run for the model it found: its idea, what it
  computes, its cross-region profile, and the full (untrained) source, ordered by reward.
- **Top summary** — every run's reward (`test_corr`) vs iteration, one line per run,
  faceted by dataset (`study`).
- **Per-run trajectories** — `test_corr` and `train_corr` vs iteration with a
  running-max line, labelled by tag + dataset/model/effort.
- **Attempts table** — interactive and sortable (click a header); the `idea` text is
  expandable so long ideas stay readable without breaking the layout.

## Hosting on GitHub Pages

The page is committed to `docs/index.html` and served by GitHub Pages from the `/docs`
folder of the default branch (Settings → Pages → *Deploy from a branch* → `main` `/docs`,
or `gh api`). The HTML is fully self-contained, so the gitignored `runs/` data is not
needed to view it. Regenerate and commit after new runs:

```bash
python analyze/build_report.py --runs-only --out docs/index.html
git add docs/index.html && git commit -m "docs: refresh report"
```

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
