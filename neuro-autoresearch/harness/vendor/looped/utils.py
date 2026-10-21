"""Minimal path/pickle helpers adapted from looped-networks.

Only this repository's .env is consulted; no sibling checkout is imported.
"""
import os
import pickle
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DATA_DIRS = {
    "NSD_DATA_DIR": "/data/shared/datasets/nsd_processed",
    "LAION_DATA_DIR": "/data/shared/datasets/laion_fmri/processed",
    "TRIPLEN_DATA_DIR": "/data/shared/datasets/triplen/processed",
    "TVSD_DATA_DIR": "/data/shared/datasets/papale2025.tvsd/processed",
}


def get_env_var(key: str) -> str:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    value = os.environ.get(key, DEFAULT_DATA_DIRS.get(key))
    if not value:
        raise RuntimeError(f"Missing required env var: {key} (check this repository's .env)")
    return value


def load_pickle(file_path):
    path = Path(file_path).expanduser().resolve()
    stat = path.stat()
    return _load_pickle(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=4)
def _load_pickle(path, size_bytes, mtime_ns):
    """Cache immutable source objects, invalidating when a prepared file changes."""
    with open(path, "rb") as file:
        return pickle.load(file)
