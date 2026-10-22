# Fixed search launcher

Run commands from the repository root. The launcher fixes three proposer slots
and their efforts; it rejects `--fleet` and `--effort` overrides.

| Role | CLI | Exact model | Effort |
| --- | --- | --- | --- |
| p00 | codex | gpt-6-astra | medium |
| p01 | claude | claude-opus-5-5 | medium |
| p02 | agy | gemini-4-argon-medium | medium |
| Conductor | codex | gpt-6-astra | xhigh |
| Repair | codex | gpt-6-astra | medium |

```bash
# Print role settings and exact provider argv; no model requests.
search/run_fixed.sh --show-config

# Check installed providers, dataset files, cohorts, and protocol; no model requests.
search/run_fixed.sh laion --preflight
search/run_fixed.sh triplen --preflight

# Start a real 12-hour search. Run these sequentially on this one-GPU machine.
search/run_fixed.sh laion --hours 12 --gpu 0
search/run_fixed.sh triplen --hours 12 --gpu 0
```

The conductor runs immediately and then hourly by default. Set `--epoch-mins 30`
for half-hourly passes or `--epoch-mins 0` to disable it. The default run directory
is `runs/<dataset>__<current-ET-date>/`; `--date YYYY-MM-DD` selects another run.
Relaunching the same date resumes it subject to the existing protocol and
reproducibility checks. Use `search/run.sh` directly for a configurable roster.

## Authentication and permissions

Roles use existing machine authentication. The fixed launcher pins agy's ADC
route and removes conflicting process-local Claude authentication overrides,
while retaining the remaining shell environment. The selected Python is `NEURO_PYTHON`, legacy `PYTHON`, repository `.venv/bin/python`,
or `python` on `PATH`, in that order. The same interpreter is exported for hub calls.
Executable overrides `NEURO_CODEX_BIN`, `NEURO_CLAUDE_BIN`, and `NEURO_AGY_BIN` remain
available; model alias overrides do not affect the fixed roster's exact model IDs.

The fixed launcher exports `AGY_ADC_AUTH=true`, selecting the existing Google
Cloud Application Default Credentials for `/home/ymehta3/.local/bin/agy` even in
a fresh shell. Existing ADC credentials and their quota project remain unchanged.
Claude Code uses the saved claude.ai first-party Max login verified on this machine;
the wrapper clears process-local API-key, token and backend override variables
that could select another route. No new login, global authentication change or
settings change is performed.

All roles skip permission prompts as requested: Codex uses
`--dangerously-bypass-approvals-and-sandbox`; Claude Code and agy use
`--dangerously-skip-permissions`. Working directories and `--add-dir` do not provide
filesystem isolation by themselves. An outer Bubblewrap sandbox supplies the
filesystem/process boundary, and a role-bound socket broker supplies bounded search
feedback and submissions. Namespace support is mandatory; launch fails closed if
it is unavailable. See the [access contract](access-feedback.md) for role operations
and limits. Proposers edit only `model.py` and notes; only LAION-fMRI or Triple-N RSA
supplies feedback. NSD and TVSD remain final-only encoding datasets.

## Logs and failures

Provider events are retained in `logs/pNN.jsonl` and `conductor.log`
under the run directory. `usage.json` indexes those logs; dollar cost remains
unknown rather than being inferred from token counts or subscription credits.

Preflight failures stop before creating a run. During search, adapter-detected
configuration/authentication failures exit 78. A proposer slot with that status
is disabled instead of repeatedly restarted; if all slots are disabled, the run
stops. A conductor failure with status 78 disables the conductor while the
proposer/evaluator loops continue. Other worker exits remain eligible for restart.
Inspect the named log, correct the provider configuration, and relaunch to restore
a disabled role.

For agy, the adapter checks the terminal stream result as well as the process
exit code: denied actions and incomplete or invalid terminal states cannot count
as successful sessions. Exit zero alone is insufficient because agy can soft-deny
tools in headless mode.

Stop the launcher with Ctrl+C. Shutdown signals the loop process groups, allows
a grace period, kills remaining processes, and requeues orphaned running models.
Keep run databases, logs, data, caches and machine-local credentials out of Git.

## Verified CLI checks

On 2026-10-21, actual Astra medium, Astra xhigh, Opus 5.5 medium, and Argon 4 medium
sessions each executed a shell marker with the permission-skipping flags. Argon
reused the existing ADC configuration unchanged. These checks cover model/effort
selection, authentication and shell execution; they do not establish completion
of a scientific search. Provider unit tests use fake CLIs and make no model requests.

A subsequent five-minute LAION search completed with all three providers submitting
scored models: 55 successful evaluations, best mean RSA 0.190032, and a clean
shutdown. See the [smoke analysis](laion-smoke-2026-10-21.md) for provider counts
and the observed startup bottlenecks.
