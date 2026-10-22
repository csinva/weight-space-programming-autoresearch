# Documentation

## Current workflow

- [Repository quick start](../README.md): preflight, search and final evaluation.
- [Launcher runbook](search-launcher.md): roles, authentication, resume and failures.
- [Scientific contract](protocol.md): fixed cohorts, metrics, provenance and reproducibility.
- [Access and feedback](access-feedback.md): role permissions, bounded feedback and isolation limits.
- [Prepared data](data.md): existing response and stimulus paths.
- [Champion simplification](simplification.md): exact and approximate modes.
- [Final evaluation](../experiments/evaluation/README.md): frozen cross-dataset encoding artifacts.

## Historical and explanatory material

These records are retained for context; use the current workflow above to launch runs.

- [Migration plan](migration-plan.md) and [validation record](migration-validation.md).
- [LAION smoke analysis, 2026-10-21](laion-smoke-2026-10-21.md).
- [HTML overview](index.html) and [concept lessons](concepts/index.html), which describe
  the pre-migration system and may include retired launch instructions.
- [Historical champions](../runs/best/README.md) and the `legacy_*` evaluation modules.

Old baked vision formats under `scripts/prepare_data` are retired. Optional
language loaders (`pip install -e '.[language]'`) and historical language
experiments are outside the current LAION/Triple-N search protocols. Reproducing
an old experiment requires its original revision and data format.
