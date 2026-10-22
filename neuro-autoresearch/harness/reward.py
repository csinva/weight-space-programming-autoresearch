"""Fixed brain-alignment evaluation; candidate-generating agents must not edit it.

Vision search rewards are source-compatible RSA on LAION-fMRI or Triple-N.
Final OOD evaluation uses source-compatible ridge encoding on NSD or TVSD.
The legacy language readout is retained separately. The caller persists rows.
"""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import torch

from harness.datasets import Dataset
# Retain ridge/ROI and legacy-score exports used by historical analysis scripts.
from harness.ridge import ridge_predict, _readout_device, _to_np
from harness.roi import roi_breakdown, select_targets, reward_region_mask
from harness.legacy_reward import _corr_per_column, _noise_normalize, _cv_column_r, predictivity
from harness.protocols import SOURCE_REVISION, VISION_DATASETS, protocol_for

MAX_DOUT = 10_000             # hard cap on a model's representation width D_out (ridge feasibility)
FORWARD_BATCH = 512           # chunk the forward pass over samples (per-sample invariant => identical output)
# The ONE parameter cap, robustly enforced below (after the forward pass, on every fixed tensor the
# model materialises -- see robust_n_params). Set per run via the NEURO_MAX_PARAMS env var; this is
# the single source of truth for the limit (the proposer prompt is told the same number).
MAX_PARAMS = 100_000


def _parameter_limit() -> int:
    # Final evaluation can restore a saved run environment in the same process.
    # Read its cap at evaluation time, rather than keeping the import-time value.
    limit = int(os.environ.get("NEURO_MAX_PARAMS") or MAX_PARAMS)
    if limit < 0:
        raise ValueError("NEURO_MAX_PARAMS must be nonnegative")
    return limit


def _forward(model, X):
    """Run sample-independent model forwards in bounded batches."""
    if not torch.is_tensor(X) or X.shape[0] <= FORWARD_BATCH:
        return model(X)
    return torch.cat([model(X[i:i + FORWARD_BATCH]) for i in range(0, X.shape[0], FORWARD_BATCH)], 0)


def _experiment_notes(model) -> str:
    """Capture concise proposer notes for cross-island feedback."""
    return json.dumps({
        "description": (getattr(model, "description", "") or "")[:600],
        "insight": (getattr(model, "insight", "") or "")[:600],
    })


def count_params(model) -> int:
    """Count registered and loose tensors, deduplicated by identity.

    Call after forward to include lazy weights. Traverse nested containers and
    modules with cycle/depth guards; uninspectable values are skipped."""
    seen_t: set[int] = set()      # tensor ids already counted
    seen_c: set[int] = set()      # container/module ids already visited (cycle guard)
    total = 0

    def add(t: "torch.Tensor") -> None:
        nonlocal total
        if id(t) not in seen_t:
            seen_t.add(id(t))
            total += int(t.numel())

    def walk(obj, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(obj, torch.Tensor):
            add(obj)
            return
        if id(obj) in seen_c:
            return
        if isinstance(obj, torch.nn.Module):
            seen_c.add(id(obj))
            for t in obj.parameters(recurse=True):
                add(t)
            for t in obj.buffers(recurse=True):
                add(t)
            for v in vars(obj).values():
                walk(v, depth + 1)
        elif isinstance(obj, dict):
            seen_c.add(id(obj))
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            seen_c.add(id(obj))
            for v in obj:
                walk(v, depth + 1)

    try:
        if isinstance(model, torch.nn.Module):
            walk(model, 0)                                  # params/buffers/submodules + loose attrs
        for v in getattr(model, "__dict__", {}).values():  # loose tensors on any model (incl. plain classes)
            walk(v, 0)
    except Exception:  # noqa: BLE001  -- honest best-effort; never fail scoring over a count
        pass
    return total


def robust_n_params(model) -> int:
    """Use the larger of declared parameters and materialized tensor elements."""
    try:
        declared = int(getattr(model, "n_params", 0) or 0)
    except (TypeError, ValueError):
        declared = 0
    return max(declared, count_params(model))


def _row_metadata(dataset: Dataset) -> dict:
    from harness.datasets import dataset_key

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
    from harness.datasets import dataset_key

    if dataset.loader not in VISION_DATASETS:
        return
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


def _validate_model_size(model, d_out: int) -> int:
    """Check materialized weights after forward and before expensive scoring."""
    if d_out >= MAX_DOUT:
        raise ValueError(f"representation D_out={d_out} >= {MAX_DOUT} cap")
    n_params = robust_n_params(model)
    limit = _parameter_limit()
    if n_params > limit:
        raise ValueError(f"n_params {n_params:,} exceeds the {limit:,} cap "
                         "(counts all materialized fixed weights)")
    return n_params


def evaluate_on_split(model, split, dataset: Dataset, slices=()) -> dict:
    """Evaluate one subject with the dataset's fixed metric and named source regions."""
    _validate_contract(dataset)
    if dataset.stage == "legacy":
        from harness.legacy_reward import evaluate_on_split as evaluate_legacy
        return {**evaluate_legacy(model, split, dataset, slices), **_row_metadata(dataset)}

    from harness.metrics import encoding_scores, matrix, rsa_scores

    t0 = time.time()
    X_train, X_test, Y_train, Y_test, roi_masks, _ = split
    device = _readout_device()

    def forward(X, label):
        # Always copy because memoized dataset tensors must survive an in-place model.
        inputs = X.to(device).clone() if torch.is_tensor(X) else np.array(X, copy=True)
        with torch.no_grad():
            output = _forward(model, inputs)
        return matrix(output, label, rows=len(X))

    if isinstance(model, torch.nn.Module):
        model.eval()
    if dataset.metric == "rsa":
        if not len(X_train):
            X, Y = X_test, Y_test
        else:
            X = torch.cat([X_train, X_test]) if torch.is_tensor(X_train) else np.concatenate([X_train, X_test])
            Y = torch.cat([Y_train, Y_test]) if torch.is_tensor(Y_train) else np.concatenate([Y_train, Y_test])
        features = forward(X, "model features")
        n_params = _validate_model_size(model, features.shape[1])
        scored = rsa_scores(features, Y, roi_masks, device, cache_sources=(Y_train, Y_test))
    else:
        A_train = forward(X_train, "training model features")
        A_test = forward(X_test, "test model features")
        d_out = A_train.shape[1]
        if d_out != A_test.shape[1]:
            raise ValueError("Model representation width differs between training and test stimuli")
        n_params = _validate_model_size(model, d_out)
        scored = encoding_scores(A_train, A_test, Y_train, Y_test, roi_masks, device)
    row = {"study": dataset.study, "model_name": getattr(model, "name", type(model).__name__),
           "idea": getattr(model, "idea", None) or getattr(model, "description", ""),
           "experiment_notes": _experiment_notes(model), "n_params": n_params,
           "n_train": len(Y_train), "n_test": len(Y_test), "n_targets": int(Y_test.shape[1]),
           "seconds": round(time.time() - t0, 2), "status": "ok", "reward_roi": dataset.reward_roi,
           **scored, **_row_metadata(dataset)}
    # Retain full precision for aggregation and the experiment artifact. test_corr
    # is the backwards-compatible score field consumed by the search database.
    row["score"] = row["test_corr"]
    row["reward"] = row["score"] if dataset.stage == "search" else None
    return row


def _error_status(exc: BaseException) -> str:
    """A compact, single-line ``status`` value for a failed evaluation."""
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())
    return ("error: " + msg)[:300]


def _error_row(model, split, dataset: Dataset, exc: BaseException,
               seconds: float, slices=()) -> dict:
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


def evaluate_on_split_safe(model, split, dataset: Dataset, slices=()) -> dict:
    """Record model/readout failures as error rows so unattended search can continue."""
    t0 = time.time()
    try:
        return evaluate_on_split(model, split, dataset, slices)
    except Exception as exc:
        traceback.print_exc()
        row = _error_row(model, split, dataset, exc, time.time() - t0, slices)
        _free_gpu(exc)
        return row


def _free_gpu(exc: BaseException | None = None) -> None:
    """Release GPU memory after a failed eval so the next iteration starts clean."""
    import gc
    if exc is not None:
        exc.__traceback__ = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(model, dataset: Dataset, slices=()) -> dict:
    """Load the dataset split and evaluate ``model`` on it.

    ``seconds`` covers the whole from-scratch evaluation (data load + readout); use
    :func:`evaluate_on_split` to reuse a pre-loaded split and time only the readout.
    """
    t0 = time.time()
    from harness.loaders import load_split  # lazy: keeps score-only callers off the data/study stack
    _validate_contract(dataset)
    row = evaluate_on_split(model, load_split(dataset), dataset, slices)
    row["seconds"] = round(time.time() - t0, 2)
    return row


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
    return {**f, "test_corr": score, "score": score,
            "reward": score if f.get("stage") == "search" else None,
            "train_corr": float(np.mean([row["train_corr"] for row in rows])),
            "roi_corr": roi_avg, "seconds": round(sum(row.get("seconds", 0) for row in rows), 2),
            "subject": None, "subjects": [row["subject"] for row in rows],
            "subject_scores": {row["subject"]: row["test_corr"] for row in rows},
            "subject_roi_scores": {row["subject"]: row["roi_corr"] for row in rows},
            "subject_results": {row["subject"]: row for row in rows},
            "n_params": max(row["n_params"] for row in rows),
            "n_targets": sum(row["n_targets"] for row in rows)}


def evaluate_subjects(model, dataset: Dataset, slices=(), *, stage: str | None = None) -> dict:
    """Score the complete fixed protocol cohort; a missing or failed subject invalidates it.

    An explicit NEURO_SUBJECT must list the same full cohort. Per-subject scoring
    for final artifacts is available through evaluate_on_split instead.
    """
    from harness.datasets import with_subject
    from harness.loaders import load_split

    t0 = time.time()
    tokens = [x.strip() for x in os.environ.get("NEURO_SUBJECT", "").split(",") if x.strip()]
    try:
        _validate_contract(dataset, stage)
        if dataset.stage != "legacy":
            protocol = protocol_for(dataset.loader)
            required = protocol.search_subjects if dataset.stage == "search" else protocol.final_subjects
            if tokens:
                selected = [with_subject(dataset, token).subject for token in tokens]
                if len(selected) != len(required) or set(selected) != set(required):
                    raise ValueError(f"{protocol.name} {dataset.stage} requires the complete cohort {required}; "
                                     f"NEURO_SUBJECT selected {selected}")
            tokens = list(required)
        elif not tokens:
            tokens = [dataset.subject]
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
            row = evaluate_on_split_safe(model, split, d, slices)
        row["subject"] = d.subject
        if row.get("status") != "ok":
            row.update(subjects=tokens, failed_subject=d.subject,
                       subject_results={r["subject"]: r for r in rows}, reward=None)
            return row
        rows.append(row)
    result = _mean_rows(rows)
    result["seconds"] = round(time.time() - t0, 2)
    return result
