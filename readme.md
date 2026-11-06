# Weight-space programming autoresearch

Mechanistic interpretability usually works backwards: train a network, then try to work out what its weights do.
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang and Khattab) argues for the forward direction:
a general-purpose language model reads and writes a network's parameters directly, guided by a natural-language specification, with no gradient descent.
That post shows a frontier model naming the game behind a policy network from its raw weights, repairing a stuck policy by editing 3.5% of its weights,
writing game-playing networks from scratch, and editing 0.016% of Qwen3.5-0.8B's weights to raise MATH-500 accuracy.

This repo tests the idea in two directions.

**Writing weights from scratch on simple tasks** ([report](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html), `analyze/toy_examples/`).
Claude, running as an autonomous coding agent, is given a blank transformer and must set every weight by hand, with no training, for a task whose algorithm is known.
Across 13 tasks it wrote 959 weight-space programs; all ten algorithmic tasks (addition, multiplication, sorting, parity, GCD and others) reached 100% accuracy within three attempts.
Because each program is written rather than learned, it comes with an account of what every head and neuron does.

**Reading the weights of a real model** (`analyze/systematic_tests/report.html`, code in `systematic-tests/`).
A simpler but more controlled question: if Claude Opus 5.5 reads the weights of Qwen3.5-0.8B, can it predict what a weight edit will do to the output?
Opus runs headless with only a file-reading tool (no code execution), and every prediction is compared with a baseline that sees the same edit description without the numbers.
The result is mostly negative. Over 420 edits (single scalars, MLP neurons, layer skips), reading the raw weights raises the correlation between predicted and measured
log-probability changes by +0.06 ± 0.02, with a clear gain only for single-scalar edits (+0.13 ± 0.04). Predicting a lower-ranked token instead of the top-1 removes the gain,
and naming which tokens rise under an amplification works no better than listing the tokens that were already likely. The model can tell which edits are large,
but the direction of an effect depends on activations that the weights alone do not reveal. Reading all of a model's weights at once, without running the network, is not yet a plausible way to predict what an edit does.

## Reproducing the hand-written weights runs

1. Edit the `evolve/program.md` first line to specify the task name e.g. "(your `task` is `five-digit-addition`)"

Some options are: ['addition-five-digits', 'multiplication-five-digits', 'sort-five-digits', 'digit-counting-10', 'parity-upto10-bits', 'boolean-circuit-5-bits', 'linear-interpolation-two-points']

Slightly harder: ['word-reversal-3x3', 'gcd-three-digits', 'decimal-to-binary-8bit']

Real world: sentiment-sst2, paraphrase-mrpc, nli-snli

1. Prompt your agent with: "Read and follow the instructions in evolve/program.md."

- e.g. `copilot --autopilot --yolo --prompt "Follow the instructions in evolve/program.md."`
- e.g. `copilot-keepgoing -- copilot --autopilot --yolo`

or

copilot --autopilot --yolo

followed by

Follow the instructions in evolve-neuro/program.md

## Reproducing the weight-reading tests

All steps use the repo-root uv environment (`uv sync`). The scripts in `systematic-tests/` sample the edits, build what the predictor sees, call
`claude -p --model claude-opus-5-5 --restricted --tools Read` once per batch of edits, measure the true effects on CPU, and compute the statistics;
`analyze/systematic_tests/build_report.py` renders the report. The exact commands are in the report's "Reproducibility" section.
Large regenerable files (the formatted weight dumps in each `bundle/weights/` and the `dense_effects.npz` files) are not committed and are rebuilt by the build and measure steps.

## Organization

- `evolve`: code for the hand-written weights runs
- `runs` stores each autoresearch run
- `systematic-tests`: code and saved predictions for the weight-reading tests
- `nl-bottleneck`: describe the best Opus 5.5 program per task in 200 words, rebuild it from the description with a fresh Opus 5.5 session, compare accuracy and errors (report in `analyze/nl_bottleneck/report.html`; steps in its Reproducibility section)
- `analyze` contains code and reports that analyze the runs (note that the run is stored in a folder and doesn't use explicit commits)
