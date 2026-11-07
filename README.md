# Interpretable weight-space programming

Neural networks predict well but are hard to understand, which is a serious problem in science, where explanation is the goal. Encoding models in neuroscience can predict brain responses to natural stimuli, yet they struggle to say which features of the stimulus drive those responses.

**Interpretable weight-space programming (iWSP)** takes a different route: a frontier coding agent builds the model by hand, and training is not allowed. With no optimizer to lean on, the agent has to use what it already knows. It writes interpretable features, reuses known ones, scores them on a prediction task and revises them. The result is a model you can read, because every part of it was written on purpose.

This repo holds the code, agent prompts and full logs behind the paper *Interpretable weight-space programming builds transparent transformers and brain encoding models*.

| Domain | What iWSP built | Result |
| --- | --- | --- |
| Algorithmic tasks | Hand-written transformers for tasks such as addition, sorting and GCD | All 26 Opus runs reach 100% on the hidden set |
| Language tasks | Hand-written transformers for classification | 87.3% on TREC, 72.4% on sentiment, 58.1% on inference, all well above baseline |
| Visual cortex | Encoding models of human fMRI and macaque spiking | More accurate than every tested network in early visual cortex, on hidden individuals who played no part in the search |
| Story-listening fMRI | Encoding models of language responses | A readable model exposed a confound at the edges of each scan |
| ECoG responses to audio | Closed-form audio features | Nearly match Whisper (0.081 versus 0.082), and combining the two reaches 0.092 |

The models also hold up to a direct test of readability. An LLM describes each model in at most 200 words, and fresh sessions rebuild it from that description alone. All 26 rebuilt models on algorithmic tasks are perfect.

These findings point to iWSP as a way to align the search for highly predictive scientific models with human-interpretable scientific theories.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and, to run the agent, [Claude Code](https://claude.com/claude-code).

```bash
git clone https://github.com/csinva/weight-space-programming-autoresearch
cd weight-space-programming-autoresearch
uv sync
```

Run the agent on one task. It writes a transformer's weights by hand, scores them and iterates:

```bash
uv run experiments/evolve/launch_claude.py --model opus55 --tasks gcd-three-digits --seeds 1 --workers 1
```

Each run gets its own folder, such as `experiments/runs/opus55/gcd-three-digits-opus55-s1/`, with every model it tried and the agent's transcript. To see the evaluator work without an agent:

```bash
uv run experiments/evolve/setup_run.py demo
cd experiments/runs/manual/demo
uv run interpretable_transformer.py --task addition-five-digits
```

## Reproducing the experiments

| Experiment | Where | Details |
| --- | --- | --- |
| Algorithmic and language transformers | `experiments/evolve/` | [Experiments guide](experiments/README.md#transformers) |
| Text bottleneck | `experiments/nl-bottleneck/` | [Experiments guide](experiments/README.md#text-bottleneck) |
| Story-listening fMRI | `experiments/evolve-neuro/` | [Experiments guide](experiments/README.md#story-listening-fmri) |
| Visual cortex | `experiments/vision/` | [Its README](experiments/vision/README.md) |
| ECoG responses to audio | `experiments/ecog/` | [Experiments guide](experiments/README.md#ecog) |
| Reading a real model's weights | `experiments/systematic-tests/` | [Experiments guide](experiments/README.md#reading-weights) |

## Repository layout

```
experiments/
  evolve/             harness, prompt and audits for the transformer runs
  evolve-neuro/       harness and prompt for the story-listening fMRI runs
  vision/             harness for the visual cortex searches, with its own history
  ecog/               ECoG searches, analyses and new test set results
  nl-bottleneck/      text bottleneck: describe a model, rebuild it, compare
  systematic-tests/   tests of whether an LLM can read a real model's weights
  runs/               every run: opus55, sonnet55, haiku55 and may15 (transformers), neuro (story fMRI)
analyze/              reports and figures built from the runs
```

The [hosted reports](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html) cover the runs in more depth. Runs live in folders and do not use commits, so every model an agent tried is saved as a snapshot next to its logged score.

## Acknowledgements

This work builds on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) by Takehi, Teoh, Zhang and Khattab, in which a general-purpose language model reads and writes a network's parameters directly from a natural-language specification.
