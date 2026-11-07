# circuits-evolve

Autonomous AI research on hand-writing transformer weights to solve tasks.

## Running the agent

1. Edit the `evolve/program.md` first line to specify the task name e.g. "(your `task` is `five-digit-addition`)" Some options are: ['addition-five-digits', 'multiplication-five-digits', 'sort-five-digits', 'digit-counting-10', 'parity-upto10-bits', 'boolean-circuit-5-bits', 'linear-interpolation-two-points']
2. Prompt your agent with: "Read and follow the instructions in evolve/program.md."

- e.g. `copilot --autopilot --yolo --prompt "Read and follow the instructions in evolve/program.md."`

## Organization

- all the relevant code is in the `evolve` folder and `runs` stores each autoresearch run (note that the run is stored in a folder and doesn't use explicit commits)
