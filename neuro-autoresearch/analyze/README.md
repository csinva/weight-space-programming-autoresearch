# `analyze/` — search report builder

Turns the proposer-pool search DBs into a single self-contained, interactive,
white-background HTML report.

## Usage

```bash
python analyze/report.py                       # every runs/*/search.db -> analyze/report.html
python analyze/report.py --run runs/nsd__2026-06-24   # just one run
python analyze/report.py --out docs/index.html -k 6 --top-src 5
```

`-k` sets how many champion cards to render per run; `--top-src` sets how many of each proposer's
top models embed expandable source in the explorer (the rest list idea + score only, to keep the
file small).

The output HTML has no external dependencies — the shared CSS (`assets/report.css`) and the
vendored Plotly cartesian bundle (`vendor/plotly-cartesian.min.js`) are inlined, every chart is an
inline Plotly spec, and the only scripting is a few lines of vanilla JS for the explorer tabs and
the log filter. Open it in any browser; the gitignored `runs/` data is not needed to view it.

## What it reads

Each search run is a `runs/<dataset>__<date>/` directory:

```
search.db                      one row per model: id, idea, summary, proposer, dataset,
                               status, test_corr, train_corr, gap, n_params, model_loc,
                               seconds, created_at, path
models/<id>.py                 the immutable, never-trained source for each row
proposers/pNN/direction.txt    that proposer's self-chosen specialization (one line)
proposers/pNN/proposer.log     its full agent trace — reasoning + the commands it ran
```

`status='ok'` rows with a `test_corr` are the scored models (the reward is the dataset's
fixed-region held-out `test_corr`). Rows that are `queued`/`running`/`error` are excluded from the
plots but counted in the header. Two **proposer-owned complexity fields** travel with each model and
are written to the DB at propose time (this report only reads them — it does not recompute):
`n_params` (the model's declared count of hard-coded / random-but-fixed weights, 0 for a genuinely
parameter-free transform) and `model_loc` (lines of code defining the model, a proxy for how
elaborate it is). The live dimension is `proposer`: ten headless agents searching the SAME space in
parallel, each chasing a different front.

## What it produces

For each run:

- **Header** — the run tally (proposed / scored / errors), proposer count, and the champion.
- **The search trajectory** — every scored model in proposal order, coloured by the agent that
  proposed it, with the running-best line. Click a proposer in the legend to isolate it.
- **What each proposer is exploring** — a roster card per agent: its `direction`, its best/mean/
  count, and a running-best sparkline. Click a card to jump into that agent below.
- **When the search happened** — an activity timeline: every scored model as a dot on its agent's
  lane over wall-clock, coloured by reward, so you can watch the parallel pool unfold.
- **Does complexity or overfit pay?** — reward vs `model_loc` (model code size — capacity is rarely
  the lever, but more elaborate models tend to score higher) and `train_corr` vs `test_corr` (the
  generalization gap).
- **Proposer explorer** — the toggle: a tab per agent. The selected panel shows the agent's
  direction, every model it proposed (idea + score, the top few expanding to full source), and its
  **activity log rendered as a filterable timeline** — reasoning prose interleaved with collapsible
  tool calls — so you can read exactly what each agent did and why.
- **Champion models** — the highest-reward models across the run, with full never-trained source.

## Hosting on GitHub Pages

The HTML is fully self-contained, so it can be committed and served from `/docs`:

```bash
python analyze/report.py --out docs/index.html
git add docs/index.html && git commit -m "docs: refresh report"
```
