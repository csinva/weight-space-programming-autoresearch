"""Immutable evaluation contracts (standard-library only).

Subject names are public dataset identifiers, never zero-based pickle indices.
Search rewards use a selected region (pooled early + ventral by default) and
encoding (default) or RSA. LAION and NSD hold out subjects and images within
the dataset: search scores its search subjects on one half of the shared
images, and final evaluation scores held-out subjects on the other half.
Every subject's ridge readout is fit on that subject's unique images only.
Triple-N search is paired with final TVSD evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib


PROTOCOL_VERSION = "held-out-subjects-images-v5"
DIAGNOSTIC_POLICY_VERSION = "selected-stream-intersection-v1"
_HUMAN_SUBREGIONS = {"early": ("V1", "V2", "V3"), "ventral": ("hV4", "FFA", "PPA"),
                     "visual": ("early visual stream", "ventral visual stream",
                                "V1", "V2", "V3", "hV4", "FFA", "PPA")}
SEARCH_SUBREGIONS = {
    "laion": _HUMAN_SUBREGIONS, "nsd": _HUMAN_SUBREGIONS,
    # Triple-N ventral = IT; its early stream pools V1, V2, the V1/V2 border and V4.
    "triplen": {"early": ("V1", "V2", "V4"), "ventral": ("IT",),
                "visual": ("early visual stream", "ventral visual stream", "V1", "V2", "V4")},
}


def search_subregions(dataset: str, region: str) -> tuple[str, ...]:
    """Public anatomical labels only; does not open neural data."""
    return SEARCH_SUBREGIONS.get(canonical_dataset(dataset), {}).get(canonical_region(region), ())


SOURCE_REVISION = "3ee575baeceb1ad784102162a051b7444182fdaa"


@dataclass(frozen=True)
class Protocol:
    name: str
    search_dataset: str
    final_dataset: str
    search_subjects: tuple[str, ...]
    final_subjects: tuple[str, ...]
    image_policy: str
    search_region: str = "ventral"
    search_metric: str = "rsa"
    final_metric: str = "encoding"
    aggregation: str = "selected_region_then_equal_subject_mean"
    version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


HALVES_POLICY = "unique_train_shared_halves_by_source"
PROTOCOLS = {
    "laion": Protocol(
        name="laion-rsa-to-laion-encoding",
        search_dataset="laion", final_dataset="laion",
        search_subjects=("sub-01", "sub-03", "sub-05"), final_subjects=("sub-06", "sub-07"),
        image_policy=HALVES_POLICY,
    ),
    # Each NSD cohort pairs two subjects who completed all 40 sessions with two who did not.
    "nsd": Protocol(
        name="nsd-rsa-to-nsd-encoding",
        search_dataset="nsd", final_dataset="nsd",
        search_subjects=("01", "02", "03", "04"), final_subjects=("05", "06", "07", "08"),
        image_policy=HALVES_POLICY,
    ),
    "triplen": Protocol(
        name="triplen-rsa-to-tvsd-encoding",
        search_dataset="triplen", final_dataset="tvsd",
        search_subjects=("M1", "M2", "M3", "M4", "M5"),
        final_subjects=("F", "N"),
        image_policy="all_1000_shared_images",
    ),
}

REGIONS = {name: ("early visual stream", "ventral visual stream")
           for name in ("nsd", "laion", "triplen", "tvsd")}
FILTERS = {
    "nsd": {"ncsnr_threshold": 0.2},
    "laion": {"ncsnr_threshold": 0.2},
    "triplen": {"reliability_threshold": 0.4},
    "tvsd": {"oracle_threshold": 0.4},
}

VISION_DATASETS = frozenset({"laion", "triplen", "nsd", "tvsd"})

_ALIASES = {
    "laion-fmri": "laion", "laion_fmri": "laion", "laion": "laion",
    "triple-n": "triplen", "triple_n": "triplen", "triplen": "triplen",
    "nsd": "nsd", "tvsd": "tvsd",
}


def canonical_dataset(name: str) -> str:
    """Normalize a supported dataset name, rejecting silent misspellings."""
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(_ALIASES)}")
    return _ALIASES[key]


def validate_search_dataset(dataset: str) -> str:
    """Return the canonical search dataset, rejecting held-out datasets."""
    key = canonical_dataset(dataset)
    if key not in PROTOCOLS:
        raise ValueError(f"{dataset!r} is not a search dataset; choose laion, nsd or triplen. "
                         "TVSD is reserved for final encoding evaluation.")
    return key


SEARCH_REGIONS = {"early": "early visual stream", "ventral": "ventral visual stream",
                  "visual": "early and ventral visual streams"}
# Anatomical stream blocks loaded for each pooled search ROI. The union pools
# every retained unit of both (disjoint) streams; streams are not reweighted.
STREAM_COMPONENTS = {"early": ("early visual stream",), "ventral": ("ventral visual stream",),
                     "visual": ("early visual stream", "ventral visual stream")}
UNION_ROIS = {SEARCH_REGIONS[k]: v for k, v in STREAM_COMPONENTS.items() if len(v) > 1}
DEFAULT_SEARCH_REGION = "visual"  # Triple-N lacks full-cohort early data: ventral
# Triple-N early responses exist only for these monkeys; early and visual search use this cohort.
TRIPLEN_EARLY_SUBJECTS = ("M1", "M4")
DEFAULT_SEARCH_METRIC = "encoding"


def canonical_region(region: str) -> str:
    key = region.strip().lower()
    aliases = {**{r: r for r in SEARCH_REGIONS}, **{v: k for k, v in SEARCH_REGIONS.items()}}
    if key not in aliases:
        raise ValueError(f"Unknown search region {region!r}; choose visual, early or ventral")
    return aliases[key]


def default_search_region(dataset: str) -> str:
    """Pooled early + ventral where the full cohort has both streams, else ventral (all five monkeys)."""
    key = canonical_dataset(dataset)
    base = PROTOCOLS.get(key) or next((p for p in PROTOCOLS.values() if key == p.final_dataset), None)
    return "ventral" if base is not None and base.search_dataset == "triplen" else DEFAULT_SEARCH_REGION


def validate_search_metric(metric: str) -> str:
    key = metric.strip().lower()
    if key not in {"rsa", "encoding"}:
        raise ValueError(f"Unknown search metric {metric!r}; choose rsa or encoding")
    return key


def protocol_for(dataset: str, region: str = "ventral", metric: str = "rsa") -> Protocol:
    """Resolve a versioned region/metric contract without opening neural data."""
    key = canonical_dataset(dataset)
    region, metric = canonical_region(region), validate_search_metric(metric)
    base = PROTOCOLS.get(key)
    if base is None:
        base = next((p for p in PROTOCOLS.values() if key == p.final_dataset), None)
    if base is None:
        raise ValueError(f"No cross-dataset protocol for {dataset!r}")
    if base.search_dataset == "triplen" and region in ("early", "visual"):
        # M2, M3 and M5 have no early units: the cohort is the monkeys with both streams.
        base = replace(base, search_subjects=TRIPLEN_EARLY_SUBJECTS)
    policy = base.image_policy
    if metric == "encoding" and base.search_dataset == "triplen":
        policy = "source_fold_0_train_test"
    return replace(base, search_region=region, search_metric=metric, image_policy=policy,
                   name=f"{base.search_dataset}-{region}-{metric}-to-{base.final_dataset}-encoding")


def search_subjects_csv(dataset: str) -> str:
    return ",".join(protocol_for(validate_search_dataset(dataset)).search_subjects)


def image_halves(sources: dict[str, str]) -> tuple[list[str], list[str]]:
    """Split shared image IDs into (search, held-out) halves within each source.

    Each source's IDs are ordered by their SHA-256 hash, so the split is fixed
    across runs and machines without a random seed.
    """
    search, held_out = [], []
    for source in sorted(set(sources.values())):
        ids = sorted((i for i, s in sources.items() if s == source),
                     key=lambda i: hashlib.sha256(i.encode()).hexdigest())
        search += ids[:len(ids) // 2]
        held_out += ids[len(ids) // 2:]
    return sorted(search, key=int), sorted(held_out, key=int)
