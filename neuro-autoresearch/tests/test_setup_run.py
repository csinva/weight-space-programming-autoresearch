"""Tests for scripts/setup_run.py -- the filesystem-native run-folder setup.

Verifies that a run folder is created with the right tree (symlinks to the FIXED
``harness``/``prompts`` + local copies of ``model.py``/``experiment.py`` + empty
``results``/``models_lib``), that from *inside* a run folder ``harness.run``,
``model``, and ``experiment`` all import (via the symlink + local copies), and
that ``harness.results.save_results`` appends to the run-local ``results/`` dir.
No real dataset is loaded.

Run standalone (`python tests/test_setup_run.py`) or via pytest.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from setup_run import setup_run  # noqa: E402


def _make_run(tmp_path, tag="demo-run1"):
    run_dir = setup_run(tag, repo_root=str(tmp_path))
    assert run_dir == os.path.join(str(tmp_path), "runs", tag)
    return run_dir


def _seed_repo(tmp_path):
    """A minimal fake repo root: real harness symlink target + stub editables."""
    os.symlink(os.path.join(REPO_ROOT, "harness"), os.path.join(tmp_path, "harness"))
    os.makedirs(os.path.join(tmp_path, "prompts"))
    open(os.path.join(tmp_path, "prompts", "program.md"), "w").write("# program\n")
    open(os.path.join(tmp_path, "model.py"), "w").write("MODEL = object()\n")
    open(os.path.join(tmp_path, "experiment.py"), "w").write(
        "LOG_SLICES = []\n"
    )


def test_tree_layout(tmp_path):
    _seed_repo(tmp_path)
    run_dir = _make_run(tmp_path)

    # symlinks to the FIXED code, stored as relative paths
    for name in ("harness", "prompts"):
        link = os.path.join(run_dir, name)
        assert os.path.islink(link), f"{name} should be a symlink"
        assert not os.path.isabs(os.readlink(link)), f"{name} symlink should be relative"
        assert os.path.realpath(link) == os.path.realpath(os.path.join(tmp_path, name))

    # local copies of the EDITABLE artifacts (real files, not links)
    for name in ("model.py", "experiment.py"):
        path = os.path.join(run_dir, name)
        assert os.path.isfile(path) and not os.path.islink(path)

    # empty working dirs
    for name in ("results", "models_lib"):
        d = os.path.join(run_dir, name)
        assert os.path.isdir(d) and os.listdir(d) == []


def test_copies_are_independent(tmp_path):
    _seed_repo(tmp_path)
    run_dir = _make_run(tmp_path)
    local_model = os.path.join(run_dir, "model.py")
    open(local_model, "w").write("MODEL = 'edited'\n")
    # editing the local copy must not touch the repo seed
    assert "edited" not in open(os.path.join(tmp_path, "model.py")).read()


def test_existing_folder_errors(tmp_path):
    _seed_repo(tmp_path)
    _make_run(tmp_path)
    try:
        _make_run(tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("re-creating an existing run folder should raise")


def test_imports_resolve_from_run_folder(tmp_path):
    """From inside a run folder, harness.run/model/experiment import via the tree."""
    _seed_repo(tmp_path)
    run_dir = _make_run(tmp_path)
    # use the real editable files so `import model`/`experiment` exercise real code
    for name in ("model.py", "experiment.py"):
        open(os.path.join(run_dir, name), "w").write(
            open(os.path.join(REPO_ROOT, name)).read()
        )
    code = "import harness.run, model, experiment; print('ok')"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=run_dir, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_save_results_writes_run_local(tmp_path):
    """harness.results.save_results lands in the run-local results/ dir."""
    _seed_repo(tmp_path)
    run_dir = _make_run(tmp_path)
    row = {
        "study": "DemoStudy", "model_name": "m", "idea": "i", "summary": "s",
        "n_params": 0, "n_train": 1, "n_test": 1, "n_targets": 1, "test_corr": 0.5,
        "train_corr": 0.6, "gap": 0.1, "seconds": 0.0,
        "status": "ok",
    }
    code = (
        "from harness.results import save_results; "
        f"save_results({row!r})"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=run_dir, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    csv = os.path.join(run_dir, "results", "DemoStudy.csv")
    assert os.path.isfile(csv), "results CSV should land in the run-local results/"
    assert "DemoStudy" in open(csv).read()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for fn in (
        test_tree_layout, test_copies_are_independent, test_existing_folder_errors,
        test_imports_resolve_from_run_folder, test_save_results_writes_run_local,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        print(f"OK: {fn.__name__}")
