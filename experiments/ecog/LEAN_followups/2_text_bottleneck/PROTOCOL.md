# Text-bottleneck protocol for LEAN-v20 (fixed Nov 9, 2026, before any session ran)

Mirrors the paper's Fig. 2g experiment: a frontier LLM (Opus 5.5) describes the final model in at most 200
words; fresh sessions given only that description and the task rebuild it; a control rebuilds with no
description.

- **Model.** LEAN-v20, as `build_embedder(device, cond_mode='none', interact_set='v3e')`. The describer sees
  `shared/model.py`, derived mechanically from `varpart-v20/lean_v20_enc.py` by `make_clean_model.py`: comments,
  docstrings, run-history strings (scores), harness glue and the `__main__` block removed. Verified to produce
  bit-identical features (3,852 dims) to the original.
- **Sessions.** All `claude-opus-5-5`, each in the same bubblewrap sandbox as the textbook baseline
  (`run_session.sh`: only `/work`, an empty home with the credentials file, read-only system and Python; no
  `/gpfs`, no SLURM, network tools masked; tools Bash/Read/Write/Edit only; no MCP servers, no skills).
  1. `describer`: `shared/DESCRIBE_TASK.md` + `model.py` + `INTERFACE.md` + 3 example clips -> `DESCRIPTION.md`
     (at most 200 words, enforced; not truncated: a longer description stops the experiment).
  2. `rebuild_1..5`: `REBUILD_TASK.md` (task + the frozen description) + `check.py` + 3 example clips.
  3. `control_1..5`: `CONTROL_TASK.md` (identical task, no description) + `check.py` + the same 3 clips as
     `rebuild_k` (paired initialization; seeds differ only by the session nonce).
  Rules for rebuild and control: closed-form only, no learning or pretrained models, one shot with no
  performance feedback, any PyPI package, H <= 4000, under 60 s per 2-minute segment.
- **Freeze and audit** (as `lean_textbook_baseline/freeze_audit.py`): sha256 of every deliverable; every tool
  call checked for paths outside the sandbox, network use other than `pip install`, credential access; code
  checked for learning/pretrained components. Paper audit (ii): each rebuilt/control feature space is run on a
  fixed clip with different example data present in its directory; outputs must be identical (no dependence
  on data it saw).
- **Scoring.** Original LEAN-v20 (already scored), the 5 rebuilds and the 5 controls, scored once each with the
  LEAN harness on the held-out sessions of patient1 (all 493 chunks and the 317-chunk primary set), patient2, patient3,
  patient4, patient5 and patient6 (grid electrodes). Measures: test r vs the original (paired chunk-bootstrap CIs);
  closeness to the original, rebuild vs control: correlation of held-out predictions with the original's
  (patient1), and dimensionality ratio H / 3,852.
