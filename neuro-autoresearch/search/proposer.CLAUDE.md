# Proposer sandbox — orientation

You are a **proposer**: a headless agent in an open-ended search for an *untrained* model that
predicts brain activity. You loop on your own. Full task framing arrives in your prompt; this file
is just ambient orientation. Your id is in `$LAB_PROPOSER`; your standing direction, if any, is in
`direction.txt`.

## Where you are
This directory is your **private sandbox** (`runs/<dataset>__<date>/proposers/pNN/`):
- **`model.py`** — your working model. **Edit this.** Everything else is support.
- **`lab.py`, `harness/`** — symlinks to the SHARED, FIXED machinery. Use them; never edit them.
- **`scratch/`** — your scratch space.

## The model contract
`model.py` defines `class Model` with attributes `name, idea, summary, n_params` and
`__call__(self, X)` mapping a stimulus batch `X (N, D_in)` → per-sample `activations (N, D_out)`,
ending the file with `MODEL = Model()`. First line is `# idea: <label>`. Keep **`D_out` < 10000**.
The model is **never trained** — weights are analytic or random-but-fixed in code. You only ever
see the stimulus `X`, never the neural target.

## Talk to the shared brain ONLY via `python lab.py` (run from this sandbox)
All proposers + the evaluator share one `search.db`. Never open the DB file directly — use:

| command | what it does |
|---|---|
| `python lab.py propose-model model.py --idea "..."` | archive your model.py + queue it for scoring → prints its `id` |
| `python lab.py wait-for-score <id>` | block until scored → JSON row incl. `test_corr` (the reward) |
| `python lab.py my-best` | your own best-scoring model so far |
| `python lab.py top-models -n 5` | the run's current best models (with source) |
| `python lab.py random-models -n 5` | a random sample of prior models (for ideas / diversity) |
| `python lab.py peek-island` | the best model from another island |
| `python lab.py query-db "SELECT ..."` | arbitrary READ-ONLY SQL → JSON |
| `python lab.py summary` | one-line tally + current champion |

`models` columns: `id, idea, summary, proposer, status (queued/running/ok/error), test_corr,
train_corr, gap, n_params, model_loc, created_at`.

## Your loop
Read the shared state (`my-best`, `top-models`, `random-models`, `summary`) → edit `model.py` →
`propose-model` → `wait-for-score` *your* result → read the state again → repeat, chasing a higher
held-out `test_corr`. Treat the harness as a black box.
