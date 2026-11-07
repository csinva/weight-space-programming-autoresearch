# Experiments

Run every command from the repo root unless noted, after `uv sync`. Each search is an agent loop, so it needs [Claude Code](https://claude.com/claude-code) and uses API credit or a subscription.

## Transformers

The agent gets a blank transformer and a task, and must set every weight by hand. `launch_claude.py` makes a run folder per task and seed, then keeps the agent going until it reaches the target number of iterations.

```bash
uv run experiments/evolve/launch_claude.py --model opus55 --tasks all --seeds 1 2 --workers 8
```

Use `--model` to choose `opus55`, `sonnet55` or `haiku55`, and `--tasks` to name tasks from `TASKS` in `evolve/launch_claude.py`. The agent's instructions are in `evolve/program.md`.

The harness restricts the agent's tools, but it cannot stop the agent's own Python from training, so we audit each run afterward:

```bash
uv run experiments/evolve/audit_runs.py runs-opus55         # scan the code for training
uv run experiments/evolve/audit_data_use.py runs-opus55     # rebuild with altered data, check the weights
uv run experiments/evolve/audit_transcripts.py runs-opus55  # check the session logs
```

`evolve/heldout_fresh_seed.py` and `evolve/heldout_trec.py` compute hidden accuracy.

## Text bottleneck

An LLM describes each best model in at most 200 words, and fresh sessions rebuild it from that description alone. The scripts in `nl-bottleneck/` are numbered and run in order from that folder.

## Story-listening fMRI

The agent writes features that predict fMRI responses to narrative stories. First build the data from the public release, [OpenNeuro ds003020](https://openneuro.org/datasets/ds003020). This downloads about 3 GB:

```bash
export FMRI_DIR=~/data/deep-fMRI
uv run experiments/evolve-neuro/prepare_public_data.py --subject UTS03
```

Then point your agent at `experiments/evolve-neuro/program.md`. To score a model yourself:

```bash
uv run experiments/evolve-neuro/setup_run.py demo --skip-baseline
cd experiments/runs-neuro/demo
uv run interpretable_transformer.py --device cpu
```

Our searches scored 3 stories, but `onapproachtopluto` is not in the public release, so a fresh run scores the other 2 and its numbers will differ slightly from the logged ones.

## Visual cortex

The visual cortex searches have their own harness and commands. See `vision/README.md`, and run them from that folder.

## Reading weights

These tests ask whether Claude Opus 5.5 can predict what an edit will do from reading a real model's weights. Over 420 edits, the answer is mostly no. The code samples the edits, gives the model what it needs, and measures the true effects:

```bash
uv run python experiments/systematic-tests/run_round.py build --round 0
uv run python experiments/systematic-tests/run_round.py predict --round 0
uv run python experiments/systematic-tests/run_round.py measure --round 0
```

The exact commands for each experiment are in the Reproducibility section of `analyze/systematic_tests/report.html`. The formatted weight dumps and `dense_effects.npz` files are not committed, and these commands rebuild them.
