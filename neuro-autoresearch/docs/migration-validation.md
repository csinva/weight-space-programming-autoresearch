# Migration validation — October 21, 2026

The migration was implemented after the plan in `migration-plan.md`, with
parallel work on data/protocols, scoring, and provider integration.

- Full suite: **65 tests passed** in the repository `.venv`.
- Python compilation, shell syntax and `git diff --check` passed.
- Installed Codex and Antigravity CLI/model preflights passed for both protocols.
- Actual prepared responses and images loaded through the vendored readers for
  all four datasets. Available source cohorts were confirmed: five LAION, five
  Triple-N, eight NSD and two TVSD subjects.
- Complete RSA baseline evaluations succeeded: LAION-fMRI **0.0534619916** over
  five subjects and 1,252 non-NSD-source shared stimuli; Triple-N **0.0426637522**
  over five monkeys and 1,000 stimuli. These are the sparse seeded baseline's
  initial rewards, not results of an autonomous search.
- A real LAION queue round trip passed: scaffold → propose → claim → RSA score →
  record → hash-verified champion selection.
- Encoding diagnostics succeeded on real NSD and TVSD responses using 40 train
  and 12 test images for one subject each. These reduced diagnostics validate the
  pipeline and are **not final scientific evaluations**. Final artifact writing
  and search-database separation were exercised with synthetic tests.

Local verification outputs are under ignored `results/smoke/`: `laion-rsa.json`,
`triplen-rsa.json`, `queue-roundtrip.json`, and `encoding-diagnostics.json`.

No autonomous search, paid model request, full OOD evaluation, Git commit or
publication was performed. CLI arguments and working-directory permissions were
validated without a real model session; OS-level read isolation of held-out data
was not established. The fixed dataset/stage checks and separate final artifacts
provide the implemented experimental boundary.
