"""Session restarts preserve working notes and archived experiments."""
import pytest

from search.notes import seed_notes


def test_seed_notes_preserves_long_resume_history(tmp_path):
    seed_notes(tmp_path)
    notes = tmp_path / 'scratch/notes.md'
    assert notes.stat().st_size < 4096
    old_notes = b'legacy experiment evidence\n' * 1000
    notes.write_bytes(old_notes)
    archive = tmp_path / 'scratch/archive'
    archive.mkdir()
    old_experiment = archive / 'candidate1.md'
    old_experiment.write_bytes(b'informative failure\n')
    seed_notes(tmp_path)
    assert notes.read_bytes() == old_notes
    assert old_experiment.read_bytes() == b'informative failure\n'


def test_seed_notes_does_not_follow_agent_symlinks(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    scratch = workspace / 'scratch'
    scratch.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        seed_notes(workspace)
    assert not (outside / 'notes.md').exists()
    scratch.unlink()
    scratch.mkdir()
    (scratch / 'notes.md').symlink_to(outside / 'notes.md')
    seed_notes(workspace)
    assert not (outside / 'notes.md').exists()
