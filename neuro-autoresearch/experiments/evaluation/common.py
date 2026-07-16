"""Shared representation scoring and equal-weight subject aggregation helpers.

Score a raw representation (train/test activations already in hand) through the harness
reward-region readout -- the body of ``harness.reward.evaluate_on_split`` for callers that hold
features rather than a MODEL. Same reward-region target selection, same fast/himalaya ridge, same
per-target noise ceiling, same ``cv_folds`` branch -- so the number is apples-to-apples with the
search DB. Returns ``test_corr`` under the active dataset's metric plus its per-ROI breakdown.
"""
import hashlib
import json
import os

import numpy as np
import torch


def subject_tokens(cli_subject=None) -> list[str]:
    """Return ordered subjects from a comma-separated CLI value."""
    return [token.strip() for token in str(cli_subject or "").split(",") if token.strip()]


def common_roi_keys(roi_dicts: list[dict]) -> set:
    """Return ROIs present for every subject, matching ``harness.reward._mean_rows``."""
    return set.intersection(*(set(rois or {}) for rois in roi_dicts)) if roi_dicts else set()


def mean_scalar(values) -> float:
    """Equal-weight mean of per-subject scalars, rounded like the harness reward."""
    return round(float(np.mean(list(values))), 4)


def sha256_file(path: str) -> str:
    """SHA256 of an evaluated model file, used to bind results to exact source."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_scores(subject_results: dict, subjects: list[str]) -> dict:
    """Canonical per-subject scores plus their equal-weight cohort mean."""
    values = {str(subject): float(subject_results[str(subject)]) for subject in subjects}
    return {"subjects": values, "mean": mean_scalar(values.values())}


def validate_evaluation(document: dict, expected_models: list[str], model_path: str) -> None:
    """Reject incomplete, internally inconsistent, or stale evaluation artifacts."""
    if document.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if document.get("dataset", {}).get("domain") not in {"vision", "language"}:
        raise ValueError("dataset.domain must be vision or language")
    if not document.get("dataset", {}).get("metric"):
        raise ValueError("dataset.metric is required")
    if set(document.get("models", {})) != set(expected_models):
        raise ValueError(f"expected models {expected_models}, got {list(document.get('models', {}))}")

    def reject_gap(value):
        if isinstance(value, dict):
            if "gap" in value:
                raise ValueError("gap is intentionally not part of the evaluation schema")
            for child in value.values():
                reject_gap(child)
        elif isinstance(value, list):
            for child in value:
                reject_gap(child)

    reject_gap(document)
    cohorts = document.get("cohorts", {})
    for model_id, model in document["models"].items():
        for cohort in ("search", "heldout"):
            expected_subjects = [str(s) for s in cohorts.get(cohort, [])]
            scores = model.get("scores", {}).get(cohort, {})
            actual_subjects = list(scores.get("subjects", {}))
            if actual_subjects != expected_subjects:
                raise ValueError(
                    f"{model_id}.{cohort} subjects must be {expected_subjects}, got {actual_subjects}")
            expected_mean = mean_scalar(scores["subjects"].values())
            if scores.get("mean") != expected_mean:
                raise ValueError(
                    f"{model_id}.{cohort}.mean must be {expected_mean}, got {scores.get('mean')}")

    recorded_hash = document["models"]["champion"].get("model_sha256")
    actual_hash = sha256_file(model_path)
    if recorded_hash != actual_hash:
        raise ValueError(f"champion hash mismatch: recorded {recorded_hash}, actual {actual_hash}")


def write_evaluation(document: dict, path: str) -> None:
    """Write a validated evaluation atomically enough for normal local runs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(document, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def readout(A_train, A_test, Ytr, Yte, roi_masks, nc, DATASET, R, roi_breakdown, device):
    """Ridge from ``A_train`` activations to the reward-region targets, scored on ``A_test``."""
    from harness.ridge import ridge_predict

    mask = R.reward_region_mask(roi_masks, DATASET.reward_roi)
    if mask is not None:
        roi_masks = {k: np.asarray(v, dtype=bool)[mask] for k, v in roi_masks.items()}
        if nc is not None:
            nc = nc[mask]
        if torch.is_tensor(Ytr):
            mask = torch.as_tensor(mask, device=Ytr.device)
    Yv_train, Yv_test = R.select_targets(Ytr, mask), R.select_targets(Yte, mask)

    # cv_folds>1 (nnn/ieeg): reproduce the reward's rotating K-fold CV over the FULL set; else the
    # single train/test split -- the same choice harness.reward makes.
    if getattr(DATASET, "cv_folds", 0) > 1:
        A_full = torch.cat([A_train, A_test], 0)
        Y_full = torch.cat([torch.as_tensor(Yv_train), torch.as_tensor(Yv_test)], 0)
        train_n, test_n = R._cv_column_r(A_full, Y_full, DATASET.cv_folds, device, nc)
    else:
        pred_train, pred_test = ridge_predict(A_train, A_test, Yv_train, device=device)
        test_n = R._noise_normalize(R._corr_per_column(pred_test, R._to_np(Yv_test)), nc)
        train_n = R._noise_normalize(R._corr_per_column(pred_train, R._to_np(Yv_train)), nc)
    return {
        "d_out": int(A_train.shape[1]),
        "test_corr": round(float(test_n.mean()), 4), "train_corr": round(float(train_n.mean()), 4),
        "roi_corr": roi_breakdown(roi_masks, train_n, test_n),
    }
