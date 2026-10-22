# neuro-autoresearch

The current instructions are in [AGENTS.md](AGENTS.md), and the executable
workflow is in [README.md](README.md). `search/run_fixed.sh` launches Codex/Astra,
Claude Code/Opus, and Antigravity/Argon through `search/providers.py`.

Search feedback is RSA on all LAION-fMRI subjects or all Triple-N monkeys.
Final evaluation is encoding on NSD or TVSD respectively. The historical
Copilot/encoding-search workflow has been replaced; do not restore it from
old champion metadata or HTML concept documents.

Per-proposer instructions live in `search/proposer/prompt.md` and are linked as
`AGENTS.md`, `GEMINI.md`, and `CLAUDE.md` inside each proposer workspace.
