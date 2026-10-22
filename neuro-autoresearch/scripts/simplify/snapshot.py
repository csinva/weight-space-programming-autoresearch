"""Copy bounded regular source bytes without following agent-controlled symlinks."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile


def snapshot(source: str, destination: str) -> None:
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            raise ValueError('Expected a regular source file under 1 MB')
        data = stream.read(1_000_001)
        if len(data) > 1_000_000:
            raise ValueError('Source exceeds 1 MB')
    target = Path(destination)
    # Atomic replacement replaces a stale destination symlink rather than its target.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = output.name
            output.write(data)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    snapshot(args.source, args.destination)
