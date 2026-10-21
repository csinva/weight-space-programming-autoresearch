"""Read-only runtime/data checks before a search creates persistent state."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def validate_sources(dataset: str) -> dict[str, str]:
    from harness.datasets import get_dataset
    from harness.loaders.vision.looped import dataset_source_paths
    from harness.protocols import validate_search_dataset

    paths = dataset_source_paths(get_dataset(validate_search_dataset(dataset)))
    for kind, raw_path in paths.items():
        path = Path(raw_path).expanduser()
        if not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
            raise ValueError(f"Search {kind} file missing, empty, or unreadable: {path}")
    return paths


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m search.preflight <laion|triplen>")
    if sys.version_info < (3, 12):
        raise SystemExit("Python >=3.12 is required; install the project environment")
    try:
        import torch, numpy, scipy, h5py, PIL, sklearn  # noqa: F401
        paths = validate_sources(args[0])
    except (ImportError, ValueError, OSError) as exc:
        raise SystemExit(f"[search] environment/data preflight failed: {exc}") from exc
    print(f"[search] runtime/data ready: {args[0]} ({', '.join(paths)})")


if __name__ == "__main__":
    main()
