#!/usr/bin/env python3
"""Headless CLI adapters for Codex, Claude Code, and Antigravity.

Agent specs are ``provider:model[:effort]``. Preflight reads CLI help/model lists;
it never submits a prompt. Native agy events are preserved while terminal results
are checked; nonrecoverable provider failures exit 78 so supervisors stop retrying.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys

DEFAULT_ROSTER = "claude:opus5.5:medium agy:argon4:medium agy:argon4:medium"
PROVIDER_UNAVAILABLE = 78
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
    if provider not in {"codex", "claude", "agy"}:
        raise ValueError(f"Unsupported provider {provider!r}; choose codex, claude or agy")
    if effort not in EFFORTS:
        raise ValueError(f"Unsupported reasoning effort {effort!r}; choose {', '.join(sorted(EFFORTS))}")
    if provider == "codex" and model == "astra":
        model = os.environ.get("NEURO_CODEX_MODEL", "gpt-6-astra")
    if provider == "claude" and model == "opus5.5":
        model = os.environ.get("NEURO_CLAUDE_MODEL", "claude-opus-5-5")
    if provider == "agy" and model == "argon4":
        model = os.environ.get("NEURO_AGY_MODEL") or f"gemini-4-argon-{effort}"
    if provider == "agy" and model.startswith("gemini-4-argon-"):
        model_effort = model.removeprefix("gemini-4-argon-")
        if model_effort not in {"low", "medium", "high"} or model_effort != effort:
            raise ValueError(f"Argon model {model!r} conflicts with effort {effort!r}; "
                             "use agy:argon4 with low, medium or high effort, or a matching exact ID")
    # Codex calls its maximum reasoning effort xhigh; agy also accepts max.
    if provider == "codex" and effort == "max":
        effort = "xhigh"
    executable = os.environ.get(f"NEURO_{provider.upper()}_BIN", provider)
    if not model or not executable:
        raise ValueError("Model ID and CLI executable must be nonempty")
    return Agent(provider, model, effort, executable)


def _options(agent: Agent, cwd: str) -> dict[str, str | None]:
    """Headless options shared by command construction and capability checks."""
    if agent.provider == "codex":
        return {"--model": agent.model,
                "--config": f'model_reasoning_effort="{agent.effort}"',
                "--dangerously-bypass-approvals-and-sandbox": None,
                "--cd": cwd, "--skip-git-repo-check": None, "--json": None}
    options = {"--model": agent.model, "--effort": agent.effort,
               "--dangerously-skip-permissions": None, "--output-format": "stream-json"}
    if agent.provider == "claude":
        options["--verbose"] = None
        options["--include-partial-messages"] = None
    return options


def _restrictions(agent: Agent) -> list[str]:
    """Keep native coding tools; omit unrelated account plugins and integrations."""
    if agent.provider == "codex":
        flags = ["--ignore-user-config", "--config", 'web_search="disabled"']
        for feature in ("apps", "plugins", "remote_plugin", "browser_use",
                        "computer_use", "image_generation", "multi_agent"):
            flags.extend(["--disable", feature])
        return flags
    if agent.provider == "claude":
        # Unlike --bare, safe mode supports the subscription OAuth token.
        return ["--safe-mode", "--tools", "Bash,Read,Edit,Write,Glob,Grep",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--disable-slash-commands"]
    # This agy version exposes no built-in tool allowlist. Its private HOME
    # excludes host plugins/config, but its built-in web tools remain available.
    return ["--disable-slash-commands"]


def command(agent: Agent, cwd: str, prompt: str, write_dirs: list[str] | None = None) -> list[str]:
    """Construct argv without a shell; prompt text is never evaluated by the launcher."""
    argv = [agent.executable]
    if agent.provider == "codex":
        argv.append("exec")
    for flag, value in _options(agent, cwd).items():
        argv.append(flag)
        if value is not None:
            argv.append(value)
    argv.extend(_restrictions(agent))
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
        if agent.provider == "claude":
            from search.auth import claude_token
            claude_token()
        if not shutil.which(agent.executable):
            raise ValueError(f"Missing {agent.provider} executable: {agent.executable}")
        if agent.executable not in checked:
            args = [agent.executable, "exec", "--help"] if agent.provider == "codex" else [agent.executable, "--help"]
            checked[agent.executable] = _read_cli(args)
        help_text = checked[agent.executable]
        required = [*_options(agent, ""), "--add-dir",
                    *(value for value in _restrictions(agent) if value.startswith("--"))]
        if agent.provider != "codex":
            required.append("--print")
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


def _claude_auth_error(event: dict) -> bool:
    """Inspect native error fields only, never user/tool content or normal replies."""
    if event.get("type") == "assistant":
        return event.get("error") in ("authentication_failed", "authentication_error")
    if event.get("type") != "result" or not _claude_result_failed(event):
        return False
    values = [event.get("error", ""), event.get("result", "")]
    errors = event.get("errors", [])
    if isinstance(errors, list):
        values.extend(errors)
    # Terminal errors can be strings or native structured API errors. Inspect
    # only their explicit error fields rather than arbitrary nested payloads.
    def error_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return " ".join(error_text(value.get(key, ""))
                            for key in ("type", "code", "message", "error"))
        return ""
    text = " ".join(error_text(value) for value in values).lower()
    return bool(re.search(
        r"authentication[_ ](?:failed|error)|invalid[ _-](?:api[ _-]key|auth(?:entication)?[ _-]token)|"
        r"(?:oauth|access|authentication|auth) token (?:has )?(?:expired|is invalid)|"
        r"not (?:logged in|authenticated)|please (?:run /login|log in|login)|"
        r"\b401\b.*(?:unauthorized|authentication|invalid|expired)", text))


def _claude_result_failed(event: dict) -> bool:
    return event.get("is_error") is True or str(event.get("subtype", "")).startswith("error")


def run_provider(argv: list[str], cwd: str, usage=None, *, validate_agy=True,
                 provider: str | None = None) -> int:
    """Relay native logs/signals and classify failures for the explicit provider.

    Legacy callers retain agy validation by default or raw exit codes with
    validate_agy=False. Explicit provider identity takes precedence.
    """
    provider = provider or ("agy" if validate_agy else "codex")
    terminal = None
    claude_auth_failed = False
    claude_failed = False
    with subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, text=True) as child:
        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda signum, _frame: child.send_signal(signum))
        try:
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and usage is not None:
                    usage.observe(event)
                if isinstance(event, dict):
                    if provider == "claude":
                        claude_auth_failed |= _claude_auth_error(event)
                        if event.get("type") == "assistant" and event.get("error"):
                            claude_failed = True
                        if event.get("type") == "result":
                            terminal = event
                    elif event.get("event") == "result":
                        terminal = event.get("result")
            code = child.wait()
            code = code if code >= 0 else 128 - code
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if provider == "claude":
        if claude_auth_failed:
            print("[provider] Claude authentication failed; stopping this slot. "
                  "Run claude setup-token, then python -m search.auth to save it, or export a fresh CLAUDE_CODE_OAUTH_TOKEN before restarting.",
                  file=sys.stderr)
            return PROVIDER_UNAVAILABLE
        if claude_failed or (isinstance(terminal, dict) and _claude_result_failed(terminal)):
            return code or 1
        return code
    if provider != "agy":
        return code
    if not isinstance(terminal, dict):
        if code:
            return code
        print("[provider] Antigravity exited without a terminal result; stopping this slot.", file=sys.stderr)
        return PROVIDER_UNAVAILABLE
    if terminal.get("denied_actions"):
        print("[provider] Antigravity denied required tool actions; stopping this slot. "
              "Review local CLI permissions before restarting.", file=sys.stderr)
        return PROVIDER_UNAVAILABLE
    if terminal.get("status") == "SUCCESS":
        return code
    if terminal.get("status") not in {"ERROR", "CANCELED", "INTERRUPTED"}:
        print("[provider] Antigravity returned an invalid/incomplete terminal status; stopping this slot.", file=sys.stderr)
        return PROVIDER_UNAVAILABLE
    error = str(terminal.get("error", "")).lower()
    configuration_errors = ("invalid model", "conflicts with", "permission", "not available for this account",
                            "not authenticated", "unauthenticated", "login", "log in")
    if any(marker in error for marker in configuration_errors):
        print("[provider] Antigravity configuration/authentication failure; stopping this slot.", file=sys.stderr)
        return PROVIDER_UNAVAILABLE
    return code or 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("preflight", help="Check installed CLIs without sending prompts")
    check.add_argument("--agents", nargs="+", default=DEFAULT_ROSTER.split())
    check.add_argument("--effort", default="high")
    run = sub.add_parser("run", help="Execute one headless agent session")
    run.add_argument("--agent", default="codex:gpt-6-astra")
    run.add_argument("--effort", default="high")
    run.add_argument("--cwd", required=True)
    run.add_argument("--write-dir", action="append", default=[])
    run.add_argument("--prompt-file", default="-", help="Prompt file, or - for stdin")
    run.add_argument("--role", choices=("proposer", "coordinator", "repair", "standalone"), default="standalone")
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
        # The launcher runs outside the agent namespace and owns its bounded broker.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from search.agents import agent_command
        usage = None
        if os.environ.get('LAB_RUN'):
            from search.costs import UsageSession
            usage = UsageSession(Path(os.environ['LAB_RUN']), agent, args.role, cwd)
        code = None
        try:
            with agent_command(agent, argv, cwd, args.role, usage=usage) as isolated:
                code = run_provider(isolated, cwd, usage, provider=agent.provider)
            return code
        finally:
            if usage is not None:
                usage.finish(code)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(PROVIDER_UNAVAILABLE, f"[provider] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
