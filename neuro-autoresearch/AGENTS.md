# neuro-autoresearch

Autonomous search for fixed, untrained image representations. Read `README.md`
for workflows and scientific contracts. Launch search with `python -m search.run`;
run final encoding evaluation with `final_eval.py`. Scientific code lives in `neuro_eval/`.

- When creating Git commits or otherwise acting on GitHub for the user, use the user's name, Yash Mehta. Before committing, verify that the Git author name is `Yash Mehta`.
- During autonomous model search, only `model.py` and the proposer's notes/scratch are editable. Infrastructure maintenance is a separate task.
- LAION-fMRI and Triple-N supply RSA search feedback. NSD and TVSD are final-only encoding datasets; never use their data or scores to guide search, repair, coordination or model selection.
- Keep all subjects, fixed image policies, source filters, source metric kernels and explicit protocol provenance intact when maintaining `neuro_eval/`.
- Do not commit run databases, data files, caches, credentials or machine-local `.env` settings.
- Proposer instructions live in `search/prompts/proposer.md`, copied as `AGENTS.md`, `GEMINI.md` and `CLAUDE.md` into each proposer workspace.

## Workshop figures

- Keep figure-generation code in `manuscript/neurips_workshop_2026/figures_code/`.
- The `manuscript/neurips_workshop_2026/neuro_autoresearch_latex/` checkout contains manuscript sources and generated assets only; write figures to its `figures/` directory.
- Include a PNG preview alongside publication formats whenever editing a figure.
