# Textbook-baseline protocol (fixed Nov 9, 2026, before any agent ran)

**Question.** Could a conventional, expert-built interpretable feature space match what the autoresearch
search found? A reviewer could argue the search only rediscovered standard features. The baseline must not
be built by anyone who has seen the search results, so it is built by 5 fresh, isolated agents.

**Agents.** 5 independent Claude Code sessions, model `claude-opus-4-7` (the model that ran the June search,
so prior knowledge is matched). Initializations differ by a random session nonce in the prompt and by a
different set of 3 example audio clips (patient1 training chunks, audio only). Same task text for all
(`shared/TASK.md`; prompt: "Read /work/TASK.md and complete the task described there. Work independently.
(session <nonce>)"). The task text names no features, parameters or results: it describes the interface,
the fixed readout and the goal (standard, well-established features as an expert would assemble them; no
learning or pretrained models; one shot).

**Isolation (`run_agent.sh`).** Each agent runs inside bubblewrap. It sees only `/work` (task, smoke test,
example clips), a fresh empty home with the Claude credentials file, read-only system directories, Python
(`/opt/python`, `/opt/venv`: the scoring environment's packages, read-only), `uv` and a `pip` shim, and the
Claude CLI. `/gpfs` (projects, results, logs, search code, notes, memory) is not mounted; SLURM binaries are
absent; ssh, scp, sftp, rsync, wget, curl, ncat and telnet are masked. Claude runs with only the Bash, Read,
Write and Edit tools (no web tools, no subagents, no messaging other sessions, no remote triggers), no MCP
servers (no account connectors), skills disabled. The network is open (the CLI needs the API; packages
install from PyPI); no performance information exists anywhere it can reach.

**Packages.** Any PyPI package via `pip install` into `/work/pkgs` (on the path in the sandbox and at
scoring); numpy, scipy, torch pinned to the scoring versions.

**No feedback.** Agents get no scores. The only check they can run is `check.py` (shape, dtype, finiteness,
H <= 4000, under 30 s per 2-minute segment, on the example audio). Nobody intervenes during a run.

**After the runs.** Deliverables (`feature_space.py`, `FEATURES.md`, `pkgs/`) are frozen with sha256 hashes
before scoring. Audit of every tool call in the logs: any path outside `/work`, `/home/agent`, `/opt`, `/tmp`
or the system directories, any network use other than `pip install`, any learned or pretrained component in
the code. Violations are reported; a feature space that learns or uses pretrained models is excluded.
Crash policy: if a frozen feature space fails during scoring, its agent may receive the traceback (from a
training-audio chunk) and nothing else, once.

**Scoring.** Each frozen feature space is scored exactly once per subject with the same harness as every LEAN
trial (`runs-wave/multisubj`): same 500 training chunks, 4 delays, 100 ms lag, per-chunk z-score,
bootstrapped ridge; held-out sessions of patient1 (all 493 chunks, plus the 317-chunk primary set), patient2,
patient3, patient4, patient5, patient6. Grid electrodes. Compared with the search at trial 52, LEAN-v16, LEAN-v20, the
model the search kept, and Whisper, with paired chunk-bootstrap CIs per subject.
