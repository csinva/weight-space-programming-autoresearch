#!/usr/bin/env python3
"""Verified headless CLI adapters for Codex/Astra and Antigravity/Argon 4.

Agent specs are ``provider:model[:effort]``. Preflight reads CLI help/model lists;
it never submits a prompt. Run uses exec so supervisor signals reach the CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_ROSTER = "codex:astra agy:argon4"
EFFORTS = {"low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class Agent:
    provider: str
    model: str
    effort: str
    executable: str


def resolve(spec: str, effort: str = "high") -> Agent:
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(f"Invalid agent {spec!r}; use provider:model[:effort], e.g. codex:astra or agy:argon4:high")
    provider, model = parts[:2]
    effort = parts[2] if len(parts) == 3 else effort
    if provider not in {"codex", "agy"}:
        raise ValueError(f"Unsupported provider {provider!r}; choose codex or agy")
    if effort not in EFFORTS:
        raise ValueError(f"Unsupported reasoning effort {effort!r}; choose {', '.join(sorted(EFFORTS))}")
    if provider == "codex" and model == "astra":
        model = os.environ.get("NEURO_CODEX_MODEL", "gpt-6-astra")
    if provider == "agy" and model == "argon4":
        model = os.environ.get("NEURO_AGY_MODEL", "gemini-4-argon-high")
    # Codex calls its maximum reasoning effort xhigh; agy also accepts max.
    if provider == "codex" and effort == "max":
        effort = "xhigh"
    executable = os.environ.get(f"NEURO_{provider.upper()}_BIN", provider)
    if not model or not executable:
        raise ValueError("Model ID and CLI executable must be nonempty")
    return Agent(provider, model, effort, executable)


def command(agent: Agent, cwd: str, prompt: str, write_dirs: list[str] | None = None) -> list[str]:
    """Construct argv without a shell; prompt text is never evaluated by the launcher."""
    if agent.provider == "codex":
        argv = [agent.executable, "exec", "--model", agent.model,
                "--config", f'model_reasoning_effort="{agent.effort}"',
                "--config", 'approval_policy="never"',
                "--sandbox", "workspace-write", "--cd", cwd,
                "--skip-git-repo-check", "--json"]
    else:
        argv = [agent.executable, "--model", agent.model, "--effort", agent.effort,
                "--mode", "auto", "--sandbox", "--output-format", "stream-json"]
    for directory in write_dirs or []:
        argv.extend(["--add-dir", directory])
    argv.extend([prompt] if agent.provider == "codex" else ["--print", prompt])
    return argv


def _read_cli(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Provider preflight failed for {argv[0]}: {exc}") from exc
    if result.returncode:
        raise ValueError(f"Provider preflight failed: {' '.join(argv)}: {(result.stderr or result.stdout).strip()}")
    return result.stdout + result.stderr


def preflight(agents: list[Agent]) -> list[dict]:
    """Validate actual CLI capabilities and agy model availability without running an agent."""
    checked: dict[str, str] = {}
    agy_models: dict[str, set[str]] = {}
    for agent in agents:
        if not shutil.which(agent.executable):
            raise ValueError(f"Missing {agent.provider} executable: {agent.executable}")
        if agent.executable not in checked:
            args = [agent.executable, "exec", "--help"] if agent.provider == "codex" else [agent.executable, "--help"]
            checked[agent.executable] = _read_cli(args)
        help_text = checked[agent.executable]
        required = ("--model", "--sandbox", "--add-dir")
        required += ("--cd", "--json", "--config", "--skip-git-repo-check") if agent.provider == "codex" else ("--print", "--effort", "--mode", "--output-format")
        missing = [flag for flag in required if flag not in help_text]
        if missing:
            raise ValueError(f"{agent.provider} lacks required headless options: {', '.join(missing)}")
        if agent.provider == "agy":
            if agent.executable not in agy_models:
                listing = _read_cli([agent.executable, "models"])
                agy_models[agent.executable] = {line.split()[0] for line in listing.splitlines() if line.strip()}
            if agent.model not in agy_models[agent.executable]:
                raise ValueError(f"Antigravity model {agent.model!r} is not listed by agy models; set NEURO_AGY_MODEL to an available Argon 4 ID")
    return [asdict(agent) for agent in agents]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("preflight", help="Check installed CLIs without sending prompts")
    check.add_argument("--agents", nargs="+", default=DEFAULT_ROSTER.split())
    check.add_argument("--effort", default="high")
    run = sub.add_parser("run", help="Execute one headless agent session")
    run.add_argument("--agent", default="codex:astra")
    run.add_argument("--effort", default="high")
    run.add_argument("--cwd", required=True)
    run.add_argument("--write-dir", action="append", default=[])
    run.add_argument("--prompt-file", default="-", help="Prompt file, or - for stdin")
    run.add_argument("--dry-run", action="store_true", help="Print resolved command without executing")
    args = parser.parse_args(argv)
    try:
        if args.action == "preflight":
            print(json.dumps(preflight([resolve(spec, args.effort) for spec in args.agents]), indent=2))
            return 0
        agent = resolve(args.agent, args.effort)
        cwd = str(Path(args.cwd).resolve(strict=True))
        if not Path(cwd).is_dir():
            raise ValueError(f"Working directory is not a directory: {cwd}")
        prompt = sys.stdin.read() if args.prompt_file == "-" else Path(args.prompt_file).read_text()
        if not prompt.strip():
            raise ValueError("Agent prompt must not be empty")
        argv = command(agent, cwd, prompt, args.write_dir)
        if args.dry_run:
            print(json.dumps({"agent": asdict(agent), "cwd": cwd, "argv": argv}, indent=2))
            return 0
        preflight([agent])
        os.chdir(cwd)
        os.execvp(agent.executable, argv)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"[provider] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
