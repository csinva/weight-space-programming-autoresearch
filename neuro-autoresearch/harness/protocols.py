"""Immutable cross-dataset evaluation contracts (standard-library only).

Subject names are public dataset identifiers, never zero-based pickle indices.
Search rewards use RSA; final encoding evaluation is outside the search loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


PROTOCOL_VERSION = "looped-cross-dataset-v1"
SOURCE_REVISION = "3ee575baeceb1ad784102162a051b7444182fdaa"


@dataclass(frozen=True)
class Protocol:
    name: str
    search_dataset: str
    final_dataset: str
    search_subjects: tuple[str, ...]
    final_subjects: tuple[str, ...]
    image_policy: str
    search_metric: str = "rsa"
    final_metric: str = "encoding"
    aggregation: str = "equal_region_mean_then_equal_subject_mean"
    version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


PROTOCOLS = {
    "laion": Protocol(
        name="laion-rsa-to-nsd-encoding",
        search_dataset="laion", final_dataset="nsd",
        search_subjects=("sub-01", "sub-03", "sub-05", "sub-06", "sub-07"),
        final_subjects=tuple(f"{i:02d}" for i in range(1, 9)),
        image_policy="shared_test_excluding_nsd_source",
    ),
    "triplen": Protocol(
        name="triplen-rsa-to-tvsd-encoding",
        search_dataset="triplen", final_dataset="tvsd",
        search_subjects=("M1", "M2", "M3", "M4", "M5"),
        final_subjects=("F", "N"),
        image_policy="all_1000_shared_images",
    ),
}

VISION_DATASETS = frozenset({"laion", "triplen", "nnn", "nsd", "tvsd"})

_ALIASES = {
    "laion-fmri": "laion", "laion_fmri": "laion", "laion": "laion",
    "triple-n": "triplen", "triple_n": "triplen", "nnn": "triplen", "triplen": "triplen",
    "nsd": "nsd", "tvsd": "tvsd", "huth": "huth", "ieeg": "ieeg",
}


def canonical_dataset(name: str) -> str:
    """Normalize a supported dataset name, rejecting silent misspellings."""
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(_ALIASES)}")
    return _ALIASES[key]


def validate_search_dataset(dataset: str) -> str:
    """Return the canonical search dataset, rejecting held-out and legacy datasets."""
    key = canonical_dataset(dataset)
    if key not in PROTOCOLS:
        raise ValueError(f"{dataset!r} is not a search dataset; choose laion or triplen. "
                         "NSD and TVSD are reserved for final encoding evaluation.")
    return key


def protocol_for(dataset: str) -> Protocol:
    """The fixed protocol for a search or final dataset (aliases accepted)."""
    key = canonical_dataset(dataset)
    if key in PROTOCOLS:
        return PROTOCOLS[key]
    for protocol in PROTOCOLS.values():
        if key == protocol.final_dataset:
            return protocol
    raise ValueError(f"No cross-dataset protocol for {dataset!r}")


def search_subjects_csv(dataset: str) -> str:
    return ",".join(protocol_for(validate_search_dataset(dataset)).search_subjects)
