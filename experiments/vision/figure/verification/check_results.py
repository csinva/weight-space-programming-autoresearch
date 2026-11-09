"""Check included figure aggregates and frozen source without rerunning evaluation.

Run with Python's standard library only. These checks establish internal
consistency and source identity; they do not establish numerical reproducibility
from neural responses or images.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633"
LAION = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
NSD = ["01", "02", "05", "07"]
REGIONS = ["early and ventral visual streams", "early visual stream", "ventral visual stream"]
ROIS = ["V1", "V2", "V3", "hV4", "FFA", "PPA"]
MODELS = {"erjpu", "pixels", "gabor", "gabor_norm", "gist", "hmax",
          "scattering", "texture", "curvature", "alexnet_conv4"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f"{context}: {actual!r} != {expected!r}")


def check_trajectory(data):
    candidates, running = data["candidates"], data["running_best"]
    provenance = data["provenance"]
    require(data["champion"] == "erjpu", "B: unexpected champion")
    require(provenance["subjects"] == LAION, "B: LAION subject list differs")
    require(provenance["dataset"] == "laion" and provenance["metric"] == "encoding"
            and provenance["region"] == "visual", "B: unexpected search setting")
    require(provenance["protocol_version"] == "looped-cross-dataset-v4",
            "B: unexpected historical search protocol")
    require(len(candidates) == len(running) == provenance["n_scored"] == 600,
            "B: inconsistent trajectory length")
    require(len({row["id"] for row in candidates}) == len(candidates),
            "B: duplicate candidate IDs")
    require(candidates == sorted(candidates, key=lambda row: (row["created_at"], row["id"])),
            "B: candidates are out of order")
    require(provenance["db_rows"] == len(candidates) + data["n_excluded"]
            == sum(provenance["db_status_counts"].values()), "B: inconsistent row counts")
    best = None
    for index, (candidate, saved) in enumerate(zip(candidates, running), 1):
        require(candidate["index"] == saved["index"] == index, "B: incorrect index")
        for key in ("pooled", "early", "ventral"):
            require(math.isfinite(candidate[key]), f"B: non-finite {key}")
        if best is None or candidate["pooled"] > best["pooled"]:
            best = candidate
        require(saved["id"] == best["id"], f"B: incorrect best model at {index}")
        for key in ("pooled", "early", "ventral"):
            close(saved[key], best[key], f"B: running {key} at {index}")
    require(best["id"] == data["champion"], "B: champion is not pooled winner")


def check_nsd(data):
    provenance = data["provenance"]
    require(data["subjects"] == provenance["subjects"] == NSD, "C: NSD subjects differ")
    require(provenance["dataset"] == "nsd", "C: unexpected dataset")
    require(set(data["regions"]) == set(REGIONS), "C: regions differ")
    require(set(data["models"]) == MODELS, "C: required model set differs")
    ids = provenance["images"]["test_image_ids"]
    require(data["n_test_images"] == len(ids) == len(set(ids)) == 998,
            "C: test image count or uniqueness differs")
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    require(digest == data["test_image_ids_sha256"], "C: test image hash differs")
    excluded = provenance["laion_exclusion"]["shared_images_in_laion"]
    require(len(excluded) == 2 and not set(ids).intersection(excluded),
            "C: LAION overlap exclusion differs")
    require(provenance["models"]["erjpu"]["sha256"] == MODEL_SHA256,
            "C: frozen model provenance hash differs")
    require(set(provenance["per_subject"]) == set(NSD), "C: subject provenance differs")
    for name, model in data["models"].items():
        require(set(model["scores"]) == set(NSD), f"C/{name}: missing subjects")
        require(set(model["mean"]) == set(REGIONS), f"C/{name}: mean regions differ")
        for subject in NSD:
            require(set(model["scores"][subject]) == set(REGIONS),
                    f"C/{name}/{subject}: score regions differ")
        for region in REGIONS:
            scores = [model["scores"][subject][region] for subject in NSD]
            require(all(math.isfinite(score) and -1 <= score <= 1 for score in scores),
                    f"C/{name}/{region}: invalid Pearson r")
            close(model["mean"][region], statistics.mean(scores), f"C/{name}/{region}: mean")


def check_shares(data, panel, families):
    require(data["subjects"] == LAION, f"{panel}: LAION subjects differ")
    require(data["rois"] == ROIS, f"{panel}: ROIs differ")
    provenance = data["provenance"]
    require(provenance["dataset"] == "laion" and provenance["stage"] == "search"
            and provenance["metric"] == "encoding", f"{panel}: setting differs")
    require(MODEL_SHA256 in provenance["model"], f"{panel}: model provenance differs")
    means = data["share_beyond_edges"]
    subjects = data["share_beyond_edges_subjects"]
    require(set(means) == set(subjects) == set(families), f"{panel}: families differ")
    for family in families:
        require(set(means[family]) == set(subjects[family]) == set(ROIS),
                f"{panel}/{family}: ROIs differ")
        for roi in ROIS:
            require(set(subjects[family][roi]) == set(LAION),
                    f"{panel}/{family}/{roi}: subjects differ")
            values = list(subjects[family][roi].values())
            require(all(math.isfinite(value) for value in values),
                    f"{panel}/{family}/{roi}: non-finite share")
            close(means[family][roi], statistics.mean(values), f"{panel}/{family}/{roi}: mean")
    # Exact Shapley shares can be negative or exceed one. Check their signed sum.
    for roi in ROIS:
        close(sum(means[family][roi] for family in families), 1, f"{panel}/{roi}: sum")
        for subject in LAION:
            close(sum(subjects[family][roi][subject] for family in families), 1,
                  f"{panel}/{roi}/{subject}: sum")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--model", type=Path, default=ROOT / "model.py")
    args = parser.parse_args()
    try:
        data = {panel: json.loads((args.results / f"{panel}.json").read_text())
                for panel in "bcde"}
        check_trajectory(data["b"])
        check_nsd(data["c"])
        check_shares(data["d"], "D", "KSCTG")
        check_shares(data["e"], "E", "MVJOR")
        require(hashlib.sha256(args.model.read_bytes()).hexdigest() == MODEL_SHA256,
                "Frozen model.py SHA-256 differs from the canonical source")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print("PASS: saved trajectory, NSD means/provenance, signed Shapley shares, and frozen source hash.")
    print("These are artifact consistency checks; neural evaluations were not rerun.")


if __name__ == "__main__":
    main()
