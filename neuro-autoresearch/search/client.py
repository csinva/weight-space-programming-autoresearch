#!/usr/bin/env python3
"""Small sandbox-side client; no database or host filesystem access."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time

MAX_RESPONSE = 16_000_000


def request(payload):
    path = os.environ.get("NEURO_BROKER_SOCKET")
    if not path:
        raise RuntimeError("No search feedback broker is configured")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(900 if payload.get("command") == "check-equivalence" else 60)
        connection.connect(path)
        connection.sendall((json.dumps(payload) + "\n").encode())
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_RESPONSE + 1)
    if len(line) > MAX_RESPONSE or not line.endswith(b"\n"):
        raise RuntimeError("Invalid broker response")
    result = json.loads(line)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Search operation failed"))
    return result["result"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("propose-model")
    submit.add_argument("file")
    submit.add_argument("--idea", default="")
    submit.add_argument("--parent", dest="parent_id")
    submit = commands.add_parser("submit")
    submit.add_argument("--file", required=True)
    submit.add_argument("--idea", default="")
    submit.add_argument("--parent", dest="parent_id")
    equivalence = commands.add_parser("check-equivalence")
    equivalence.add_argument("file")
    equivalence.add_argument("--reference")
    equivalence.add_argument("--spec")
    best = commands.add_parser("my-best")
    best.add_argument("--no-code", action="store_true")
    for name in ("random-models", "top-models", "history"):
        command = commands.add_parser(name)
        command.add_argument("-n", "--limit", type=int, default=10)
    for name in ("status", "source", "wait-for-score"):
        command = commands.add_parser(name)
        command.add_argument("id")
        if name in {"status", "wait-for-score"}:
            command.add_argument("--details", action="store_true",
                                 help="include full subject-by-region-by-source-group diagnostics")
    for name in ("peek-island", "summary", "direction"):
        commands.add_parser(name)
    for name in ("set-direction", "reset-island"):
        command = commands.add_parser(name)
        command.add_argument("proposer")
        command.add_argument("--role", "--direction", dest="direction", required=True)
    args = vars(parser.parse_args(argv))
    command = args["command"]
    if command in {"propose-model", "submit", "check-equivalence"}:
        path = Path(args.pop("file"))
        if path.stat().st_size > 1_000_000:
            parser.error("Source must be at most 1 MB")
        args.update(command="check-equivalence" if command == "check-equivalence" else "propose-model",
                    source=path.read_text())
        args.pop("reference", None)
        args.pop("spec", None)
    if command == "my-best":
        args["with_code"] = not args.pop("no_code")
    if command == "wait-for-score":
        args["command"] = "status"
        while True:
            result = request(args)
            if result["status"] in {"ok", "error"}:
                break
            time.sleep(5)
    else:
        result = request(args)
    print(result if isinstance(result, str) else json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
