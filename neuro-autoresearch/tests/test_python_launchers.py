"""Selected Python survives relative paths, cwd changes, and venv symlinks."""
import json
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_selected_python_stays_absolute_in_children(tmp_path, selected_python_env):
    expected = selected_python_env["NEURO_PYTHON"]
    env = dict(selected_python_env)
    env["NEURO_PYTHON"] = os.path.relpath(expected, tmp_path)
    env["PYTHON"] = "/missing/legacy-python"
    script = '''
set -eu
REPO="$1"
source "$REPO/search/lib.sh"
use_project_python
cd "$REPO"
# The child resolves the exported selection after changing directories.
bash -c 'source "$1/search/lib.sh"; REPO="$1"; use_project_python; "$PY" -c "$2"' test "$REPO" 'import os; print(os.environ["NEURO_PYTHON"])'
'''
    result = subprocess.run(["bash", "-c", script, "test", str(REPO)], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
    assert len(Path(env["TEST_PYTHON_CALLS"]).read_text().splitlines()) == 1


def test_project_venv_executable_symlink_is_not_dereferenced():
    selected = REPO / ".venv/bin/python"
    if not selected.exists():
        pytest.skip("Checkout has no project venv")
    env = {**os.environ, "NEURO_PYTHON": str(selected)}
    script = '''REPO="$1"; source "$REPO/search/lib.sh"; use_project_python &&
"$PY" -c 'import json,sys; print(json.dumps([sys.executable,sys.prefix]))'
'''
    result = subprocess.run(["bash", "-c", script, "test", str(REPO)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(selected), str(REPO / ".venv")]
