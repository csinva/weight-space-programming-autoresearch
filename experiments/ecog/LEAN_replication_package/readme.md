# circuits-evolve

Autonomous AI research on hand-writing transformer weights to solve tasks.

## Running the agent

1. Edit the `evolve/program.md` first line to specify the task name e.g. "(your `task` is `five-digit-addition`)"

Some options are: ['addition-five-digits', 'multiplication-five-digits', 'sort-five-digits', 'digit-counting-10', 'parity-upto10-bits', 'boolean-circuit-5-bits', 'linear-interpolation-two-points']

Slightly harder: ['word-reversal-3x3', 'gcd-three-digits', 'decimal-to-binary-8bit']

Real world: sentiment-sst2, paraphrase-mrpc, nli-snli

1. Prompt your agent with: "Read and follow the instructions in evolve/program.md."

- e.g. `copilot --autopilot --yolo --prompt "Follow the instructions in evolve/program.md."`

or

copilot --autopilot --yolo

followed by

Follow the instructions in evolve-neuro/program.md

## Organization

- `evolve`: all the relevant code
- `runs` stores each autoresearch run
- `analyze` contains code to analyze the runs (note that the run is stored in a folder and doesn't use explicit commits)
  - see the results at <https://csinva.io/circuit-building-autoresearch/analyze/report>
