"""Seed rolling working notes without replacing history on session restart."""
from pathlib import Path
import os


TEMPLATE = """# Current working notes
Keep this rolling summary around 2–4 KB. Archive older details by candidate ID
in scratch/archive/ before replacing them here; read archives selectively.

## Best active candidate
ID and primary score: no result recorded yet.

## Current direction

## Latest observations and reviewed IDs

## Competing hypotheses

## Next comparison

## Candidates worth revisiting
Keep a small ID / one-line reason index; details belong in archive/<candidate-id>.md.
"""


def seed_notes(workspace: str | Path) -> None:
    """Create missing notes only; never read, migrate or truncate existing notes."""
    scratch = Path(workspace) / 'scratch'
    scratch.mkdir(exist_ok=True)
    # Agent-owned paths must not redirect host writes outside the workspace.
    directory = os.open(scratch, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open('notes.md', os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         0o644, dir_fd=directory)
        except FileExistsError:
            return
        with os.fdopen(fd, 'w') as stream:
            stream.write(TEMPLATE)
    finally:
        os.close(directory)
