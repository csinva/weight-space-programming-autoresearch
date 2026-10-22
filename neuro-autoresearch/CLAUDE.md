# neuro-autoresearch

The current instructions are in [AGENTS.md](AGENTS.md), and the executable
workflow is in [README.md](README.md). `python -m search.run` launches Codex/Astra,
Claude Code/Opus, and Antigravity/Argon through `search/providers.py`.

Search feedback is RSA on all LAION-fMRI subjects or all Triple-N monkeys.
Final evaluation is encoding on NSD or TVSD respectively, through `final_eval.py`.
The fixed scientific code is in `neuro_eval/`.

Per-proposer instructions live in `search/prompts/proposer.md` and are copied as
`AGENTS.md`, `GEMINI.md`, and `CLAUDE.md` inside each proposer workspace.
