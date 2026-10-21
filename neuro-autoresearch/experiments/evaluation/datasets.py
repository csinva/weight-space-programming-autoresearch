"""Current cross-dataset protocols; legacy cohorts are in legacy_datasets.py."""
from harness.protocols import PROTOCOLS, canonical_dataset, protocol_for, search_subjects_csv

VISION_DATASETS = ("laion", "triplen", "nsd", "tvsd")
SEARCH_DATASETS = tuple(PROTOCOLS)
FINAL_DATASETS = tuple(protocol.final_dataset for protocol in PROTOCOLS.values())
DATASETS = {
    dataset: {
        "domain": "vision", "metric": "rsa" if dataset in SEARCH_DATASETS else "encoding",
        "stage": "search" if dataset in SEARCH_DATASETS else "final",
        "protocol": protocol_for(dataset).to_dict(),
    }
    for dataset in VISION_DATASETS
}


def evaluation_for(dataset: str) -> dict:
    return DATASETS[canonical_dataset(dataset)]
