"""Shared fixed-cohort orchestration; phase entry points supply their own scorer."""

from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import torch

from neuro_eval.benchmarks import SOURCE_REVISION, VISION_DATASETS, protocol_for
from neuro_eval.datasets import Dataset
from neuro_eval.model import _forward, robust_n_params


def matrix(value, name: str, *, rows: int | None = None) -> torch.Tensor:
    """Reject invalid representations/responses before they can produce reward."""
    result = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty-width 2D matrix; got {tuple(result.shape)}")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{name} has {result.shape[0]} rows; expected {rows}")
    if result.is_complex() or not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values")
    result = result.float()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} overflows the source metric's float32 representation")
    return result


def region_masks(masks: dict, n_targets: int) -> dict[str, np.ndarray]:
    if not masks:
        raise ValueError("Vision scoring requires named anatomical region masks")
    result = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_targets,) or not mask.any():
            raise ValueError(f"Region {name!r} has an empty or misaligned target mask")
        result[name] = mask
    return result


def _experiment_notes(model) -> str:
    """Capture concise proposer notes for cross-island feedback."""
    return json.dumps({
        "description": (getattr(model, "description", "") or "")[:600],
        "insight": (getattr(model, "insight", "") or "")[:600],
    })


def _row_metadata(dataset: Dataset) -> dict:
    from neuro_eval.datasets import dataset_key

    result = {"dataset": dataset_key(dataset), "metric": dataset.metric,
              "stage": dataset.stage, "subject": dataset.subject}
    if dataset.loader in VISION_DATASETS:
        protocol = protocol_for(dataset.loader)
        result.update(protocol=protocol.name, protocol_version=protocol.version,
                      aggregation=protocol.aggregation, source_revision=SOURCE_REVISION,
                      search_image_policy=protocol.image_policy)
    return result


def _validate_contract(dataset: Dataset, stage: str | None = None) -> None:
    """The dataset fixes its stage and metric; environment flags cannot repurpose OOD data."""
    from neuro_eval.datasets import dataset_key

    if dataset.loader not in VISION_DATASETS:
        raise ValueError(f"Unknown image benchmark {dataset.loader!r}")
    protocol = protocol_for(dataset.loader)
    expected_stage = "search" if dataset_key(dataset) == protocol.search_dataset else "final"
    expected_metric = protocol.search_metric if expected_stage == "search" else protocol.final_metric
    if dataset.stage != expected_stage or dataset.metric != expected_metric:
        raise ValueError(f"{dataset.loader} is fixed to {expected_stage}/{expected_metric}")
    for label, requested, expected in (
        ("stage", stage, expected_stage),
        ("NEURO_STAGE", os.environ.get("NEURO_STAGE"), expected_stage),
        ("NEURO_METRIC", os.environ.get("NEURO_METRIC"), expected_metric),
    ):
        if requested and requested != expected:
            raise ValueError(f"{label}={requested!r} conflicts with {dataset.loader}'s fixed {expected!r}")


def _error_status(exc: BaseException) -> str:
    """A compact, single-line ``status`` value for a failed evaluation."""
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())
    return ("error: " + msg)[:300]


def _error_row(model, split, dataset: Dataset, exc: BaseException,
               seconds: float) -> dict:
    """A durable ``status='error'`` row for a model/readout fault (NaN scores)."""
    Y_train, Y_test = (split[2], split[3]) if split is not None else (np.empty((0, 0)), np.empty((0, 0)))
    nan = float("nan")
    return {
        "study": dataset.study,
        "model_name": getattr(model, "name", "?"),
        "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
        "experiment_notes": _experiment_notes(model),
        "n_params": robust_n_params(model),
        "n_train": int(Y_train.shape[0]),
        "n_test": int(Y_test.shape[0]),
        "n_targets": int(Y_test.shape[1]) if getattr(Y_test, "ndim", 1) > 1 else 0,
        "test_corr": nan,
        "train_corr": nan,
        "roi_corr": {},
        "seconds": round(seconds, 2),
        "status": _error_status(exc),
        "reward_roi": dataset.reward_roi,
        "score": nan,
        "reward": None,
        **_row_metadata(dataset),
    }


def _free_gpu(exc: BaseException | None = None) -> None:
    """Release GPU memory after a failed eval so the next iteration starts clean."""
    import gc
    if exc is not None:
        exc.__traceback__ = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mean_rows(rows: list[dict]) -> dict:
    """Equal subject mean, retaining every subject and its actual anatomical coverage."""
    if not rows or any(row.get("status") != "ok" or not np.isfinite(row["test_corr"]) for row in rows):
        raise ValueError("All fixed-cohort subjects must have finite successful scores")
    f = rows[0]
    common = set.intersection(*(set(row.get("roi_corr") or {}) for row in rows))
    roi_avg = {r: {k: float(np.mean([row["roi_corr"][r][k] for row in rows]))
                   for k in (f["roi_corr"][r])}
               for r in (f.get("roi_corr") or {}) if r in common}
    score = float(np.mean([row["test_corr"] for row in rows]))
    result = {**f, "test_corr": score, "score": score,
            "reward": score if f.get("stage") == "search" else None,
            "train_corr": float(np.mean([row["train_corr"] for row in rows])),
            "roi_corr": roi_avg, "seconds": round(sum(row.get("seconds", 0) for row in rows), 2),
            "subject": None, "subjects": [row["subject"] for row in rows],
            "subject_scores": {row["subject"]: row["test_corr"] for row in rows},
            "subject_roi_scores": {row["subject"]: row["roi_corr"] for row in rows},
            "subject_results": {row["subject"]: row for row in rows},
            "n_params": max(row["n_params"] for row in rows),
            "n_targets": sum(row["n_targets"] for row in rows)}
    if f.get("metric") == "rsa":
        result["rsa_diagnostics"] = _cohort_rsa_diagnostics(rows, score)
    return result


def _cohort_rsa_diagnostics(rows: list[dict], score: float) -> dict:
    """Report available-region means with explicit coverage, never reweight reward."""
    subjects = [row["subject"] for row in rows]
    diagnostics = {row["subject"]: row.get("rsa_diagnostics", {}) for row in rows}
    regions = {}
    for region in dict.fromkeys(r for row in rows for r in row["roi_corr"]):
        covered = [row for row in rows if region in row["roi_corr"]]
        regions[region] = {
            "rsa": float(np.mean([row["roi_corr"][region]["test"] for row in covered])),
            "subjects": [row["subject"] for row in covered], "n_subjects": len(covered),
            "n_targets_by_subject": {row["subject"]: diagnostics[row["subject"]].get(
                "regions", {}).get(region, {}).get("n_targets") for row in covered},
        }
    groups = {}
    for name in dict.fromkeys(name for d in diagnostics.values() for name in d.get("source_groups", {})):
        entries = {subject: diagnostics[subject].get("source_groups", {}).get(name)
                   for subject in subjects}
        covered = {subject: entry for subject, entry in entries.items()
                   if entry and entry["status"] == "ok"}
        # A cohort subgroup mean is available only when every subject succeeded.
        complete = len(covered) == len(subjects)
        group_regions = {}
        for region in dict.fromkeys(r for entry in covered.values() for r in entry["regions"]):
            region_scores = {subject: entry["regions"][region] for subject, entry in covered.items()
                             if region in entry["regions"]}
            region_complete = len(region_scores) == len(subjects)
            group_regions[region] = {
                "status": "ok" if region_complete else "unavailable",
                "rsa": float(np.mean(list(region_scores.values()))) if region_complete else None,
                "subjects": list(region_scores), "n_subjects": len(region_scores),
            }
        groups[name] = {"status": "ok" if complete else "unavailable",
                        "rsa": float(np.mean([entry["rsa"] for entry in covered.values()])) if complete else None,
                        "subjects": list(covered), "n_subjects": len(covered),
                        "n_stimuli_by_subject": {subject: entry["n_stimuli"] for subject, entry in entries.items()
                                                 if entry is not None},
                        "regions": group_regions, "subject_results": entries}
    return {"schema_version": 1, "metric": "rsa", "stage": "search", "primary_rsa": score,
            "aggregation": "equal_regions_within_subject_then_equal_subjects",
            "subjects": subjects, "n_subjects": len(subjects),
            "subject_results": diagnostics, "regions": regions, "source_groups": groups}


def evaluate_on_split_safe(model, split, dataset, scorer):
    started = time.time()
    try:
        return scorer(model, split, dataset)
    except Exception as exc:
        traceback.print_exc()
        row = _error_row(model, split, dataset, exc, time.time() - started)
        _free_gpu(exc)
        return row


def result_row(model, split, dataset, scored, n_params, started):
    Y_train, Y_test = split[2:4]
    row = {"study": dataset.study, "model_name": getattr(model, "name", type(model).__name__),
           "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
           "experiment_notes": _experiment_notes(model), "n_params": n_params,
           "n_train": len(Y_train), "n_test": len(Y_test), "n_targets": int(Y_test.shape[1]),
           "seconds": round(time.time() - started, 2), "status": "ok", "reward_roi": dataset.reward_roi,
           **scored, **_row_metadata(dataset)}
    row["score"] = row["test_corr"]
    row["reward"] = row["score"] if dataset.stage == "search" else None
    return row


def forward(model, inputs, device, label):
    copied = inputs.to(device).clone() if torch.is_tensor(inputs) else np.array(inputs, copy=True)
    if isinstance(model, torch.nn.Module):
        model.eval()
    with torch.no_grad():
        output = _forward(model, copied)
    return matrix(output, label, rows=len(inputs))


def evaluate_subjects(model, dataset: Dataset, scorer, *, stage: str) -> dict:
    """Score the complete fixed protocol cohort; a missing or failed subject invalidates it.

    An explicit NEURO_SUBJECT must list the same full cohort. Per-subject scoring
    for final artifacts is available through evaluate_on_split instead.
    """
    from neuro_eval.datasets import with_subject
    from neuro_eval.datasets import load_split

    t0 = time.time()
    tokens = [x.strip() for x in os.environ.get("NEURO_SUBJECT", "").split(",") if x.strip()]
    try:
        _validate_contract(dataset, stage)
        protocol = protocol_for(dataset.loader)
        required = protocol.search_subjects if dataset.stage == "search" else protocol.final_subjects
        if tokens:
            selected = [with_subject(dataset, token).subject for token in tokens]
            if len(selected) != len(required) or set(selected) != set(required):
                raise ValueError(f"{protocol.name} {dataset.stage} requires the complete cohort {required}; "
                                 f"NEURO_SUBJECT selected {selected}")
        tokens = list(required)
    except Exception as exc:
        return _error_row(model, None, dataset, exc, time.time() - t0)
    rows = []
    for s in tokens:
        d = with_subject(dataset, s)
        try:
            split = load_split(d)
        except Exception as exc:
            row = _error_row(model, None, d, exc, time.time() - t0)
        else:
            row = evaluate_on_split_safe(model, split, d, scorer)
        row["subject"] = d.subject
        if row.get("status") != "ok":
            row.update(subjects=tokens, failed_subject=d.subject,
                       subject_results={r["subject"]: r for r in rows}, reward=None)
            return row
        rows.append(row)
    result = _mean_rows(rows)
    result["seconds"] = round(time.time() - t0, 2)
    return result
