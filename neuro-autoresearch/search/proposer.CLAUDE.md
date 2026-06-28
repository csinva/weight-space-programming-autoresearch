# Proposer sandbox — orientation

You are a **proposer**: a headless agent in an open-ended search for an *untrained* model that
predicts brain activity. You loop on your own. Full task framing arrives in your prompt; this file
is just ambient orientation. Your id is in `$LAB_PROPOSER`; your standing direction, if any, is in
`direction.txt`.

## Where you are
This directory is your **private sandbox** (`runs/<dataset>__<date>/proposers/pNN/`):
- **`model.py`** — your working model. **Edit this.** Everything else is support.
- **`hub.py`, `harness/`** — symlinks to the SHARED, FIXED machinery. Use them; never edit them.
- **`scratch/`** — your scratch space.

## The model contract
`model.py` defines `class Model` with attributes `name, idea, description, insight, n_params` and
`__call__(self, X)` mapping a stimulus batch `X (N, D_in)` → per-sample `activations (N, D_out)`,
ending the file with `MODEL = Model()`. First line is `# idea: <label>`. Keep **`D_out` < 10000`.
The model is **never trained** — weights are analytic or random-but-fixed in code. You only ever
see the stimulus `X`, never the neural target. Write the text fields for a stranger: `idea` = a plain
one-liner of the mechanism (no `pNN`/`vNN` prefixes, no raw hyperparam codes); `description` = 2-3
re-implementable lines on what it computes; `insight` = the key takeaway in ≤3 dense sentences (what
structure it captures, what worked vs your parent, what to try next). `description`
+ `insight` are captured at score time into `experiment_notes` — the only thing other islands see
of your model via `peek-island`.

## Talk to the shared brain ONLY via `python hub.py` (run from this sandbox)
All proposers + the evaluator share one `search.db`. Never open the DB file directly — use:

| command | what it does |
|---|---|
| `python hub.py propose-model model.py --idea "..."` | archive your model.py + queue it for scoring → prints its `id` |
| `python hub.py wait-for-score <id>` | block until scored → JSON row incl. `test_corr` (the reward) |
| `python hub.py my-best` | **your own island's** best scored model, **with source** — the backbone to build on |
| `python hub.py peek-island` | best model of ONE random *other* island as idea + `experiment_notes` (description + insight) — **NO source, NO score**; re-implement it your way |
| `python hub.py random-models -n 5` | a random sample of prior models — **ideas only** (a diversity spark) |
| `python hub.py query-db "SELECT ..."` | arbitrary READ-ONLY SQL → JSON (other islands' score/source columns are hidden from you) |
| `python hub.py summary` | one-line tally + current champion |

`models` columns: `id, idea, experiment_notes (JSON {description, insight}), proposer,
status (queued/running/ok/error), n_params, model_loc, created_at`. (Other islands' `test_corr,
train_corr, roi_corr` and the source `path` are hidden from a proposer's `query-db`; see your OWN
score via `my-best`/`wait-for-score`.)

**You are your own island.** Build on **`my-best`** (your backbone); optionally re-implement a
`peek-island` idea. `random-models` is read-only awareness — ideas only, not scores or code, so you
don't just clone the global champion.

## Your loop
Read the shared state (`my-best`, `random-models`, `peek-island`) → edit `model.py` →
`propose-model` → `wait-for-score` *your* result → read the state again → repeat, chasing a higher
held-out `test_corr`. Treat the harness as a black box.
