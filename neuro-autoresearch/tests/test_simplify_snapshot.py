import os

import pytest

from scripts.simplify.snapshot import snapshot


def test_snapshot_rejects_agent_symlink_and_restores_without_following_it(tmp_path):
    secret = tmp_path / 'private'
    secret.write_text('private bytes')
    candidate = tmp_path / 'model.py'
    candidate.symlink_to(secret)
    frozen = tmp_path / 'reference.py'
    frozen.write_text('# frozen model\n')
    with pytest.raises(OSError):
        snapshot(str(candidate), str(tmp_path / 'result.py'))
    snapshot(str(frozen), str(candidate))
    assert not candidate.is_symlink()
    assert candidate.read_text() == '# frozen model\n'
    assert secret.read_text() == 'private bytes'


def test_snapshot_rejects_fifo_and_oversized_source(tmp_path):
    source = tmp_path / 'source'
    os.mkfifo(source)
    with pytest.raises(ValueError, match='regular'):
        snapshot(str(source), str(tmp_path / 'result.py'))
    source.unlink()
    source.write_bytes(b'x' * 1_000_001)
    with pytest.raises(ValueError, match='1 MB'):
        snapshot(str(source), str(tmp_path / 'result.py'))


def test_snapshot_freezes_bytes_before_later_edits(tmp_path):
    source, frozen = tmp_path / 'model.py', tmp_path / 'candidate.py'
    source.write_text('# original\n')
    snapshot(str(source), str(frozen))
    source.write_text('# later edit\n')
    assert frozen.read_text() == '# original\n'
