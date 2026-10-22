"""Executable fixtures keep shell-launcher tests independent of local Python names."""
import os
import sys

import pytest


@pytest.fixture
def selected_python_env(tmp_path):
    """A named python3 wrapper plus a broken bare python expose PATH fallbacks."""
    bindir = tmp_path / "selected python"
    bindir.mkdir()
    selected = bindir / "python3"
    selected.write_text(f'''#!{sys.executable}
import json, os, sys
with open(os.environ["TEST_PYTHON_CALLS"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
''')
    selected.chmod(0o755)
    poison = bindir / "python"
    poison.write_text('#!/bin/sh\necho "bare python must not run" >&2\nexit 99\n')
    poison.chmod(0o755)
    env = dict(os.environ)
    env.pop("PYTHON", None)
    env.update(NEURO_PYTHON=str(selected), TEST_PYTHON_CALLS=str(tmp_path / "python-calls.jsonl"),
               PATH=f"{bindir}:{env['PATH']}")
    return env
