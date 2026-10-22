# Simplify a search champion

Run from the repository root after an RSA search. Simplification starts from its
scored champion, uses a separate workspace, and writes
`RUN/simplify/<mode>/result/simplified_model.py` plus `summary.txt`.
It does not replace the source run's champion. These commands send model requests.

```bash
# Refactor while preserving activations on the generated equivalence inputs.
scripts/simplify/run.sh runs/laion__2026-10-21 --mode exact

# Allow at most a 0.5% relative reduction from the champion's re-scored RSA.
scripts/simplify/run.sh runs/laion__2026-10-21 --mode approximate --tolerance 0.005
```

Exact mode checks generated inputs using the dataset's stimulus shape and range;
it is a numerical check, not a proof of equivalence for every possible image.
Approximate mode evaluates the complete original search cohort and keeps scores
at least `reference_score - abs(reference_score) * tolerance`. Neither mode may
consult NSD or TVSD final encoding scores. Historical encoding-only runs are rejected.

The agent defaults to `codex:astra`, high effort and 15 minutes. Override with
`--model`, `--effort`, or `--minutes`; `NEURO_SIMPLIFY_MODEL` sets the default model.
The time budget applies to the agent session; setup, scoring and final validation
can take additional time. The launcher verifies the original run's scientific
snapshot before starting and restores a passing model if final validation fails.

Inspect `RUN/simplify/<mode>/simplify.log` for agent activity and, in approximate
mode, `eval/evaluator.log` for scoring. Reusing a mode reuses its workspace and
queue. The final evaluation CLI's `--model` option can evaluate the simplified
artifact explicitly; it records that this was not the original RSA champion selection.
