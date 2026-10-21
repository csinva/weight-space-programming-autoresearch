# Experiments

The current experiment is RSA search followed by cross-dataset encoding:

```bash
.venv/bin/python -m experiments.evaluation.final --run runs/laion__2026-10-21
.venv/bin/python -m experiments.evaluation.final --run runs/triplen__2026-10-21
```

See [evaluation/README.md](evaluation/README.md) for frozen-model selection,
subject cohorts and result provenance. Outputs live under `results/final/` and
never affect the search queue.

Historical evaluation schemas and cohorts remain in `evaluation/legacy_*`.
Existing plotting utilities can read historical artifacts. The previous
best-layer pretrained-baseline protocol is archived; its feature extractors
remain available for separate analyses. `complementarity.py` and `imagenet/`
are older follow-up studies, outside the current fixed untrained-model search
contract. Reproducing their original results requires the original repository
revision and evaluation settings.
