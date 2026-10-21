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


def cohort_roi_scores(subject_results: dict, subjects: list[str]) -> dict:
    """Preserve every subject's ROI scores and average ROIs present for the full cohort."""
    values = {
        str(subject): {str(roi): float(score) for roi, score in subject_results[str(subject)].items()}
        for subject in subjects
    }
    common = common_roi_keys(list(values.values()))
    first = next(iter(values.values()), {})
    ordered_common = [roi for roi in first if roi in common]
    means = {
        roi: mean_scalar(values[str(subject)][roi] for subject in subjects)
        for roi in ordered_common
    }
    return {"subjects": values, "mean": means}


def validate_evaluation(document: dict, expected_models: list[str], model_path: str) -> None:
    """Reject incomplete, internally inconsistent, or stale evaluation artifacts."""
    schema_version = document.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("schema_version must be 1 or 2")
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
    roi_names = document.get("dataset", {}).get("rois", [])
    required_rois = document.get("dataset", {}).get("required_rois", [])
    if schema_version == 2 and not isinstance(roi_names, list):
        raise ValueError("dataset.rois must be a list")
    if schema_version == 2 and not isinstance(required_rois, list):
        raise ValueError("dataset.required_rois must be a list")
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

            if schema_version == 2 and roi_names:
                roi_scores = model.get("roi_scores", {}).get(cohort, {})
                roi_subjects = roi_scores.get("subjects", {})
                if list(roi_subjects) != expected_subjects:
                    raise ValueError(
                        f"{model_id}.roi_scores.{cohort} subjects must be {expected_subjects}, "
                        f"got {list(roi_subjects)}")
                for subject, subject_rois in roi_subjects.items():
                    extras = set(subject_rois) - set(roi_names)
                    if extras:
                        raise ValueError(
                            f"{model_id}.roi_scores.{cohort}.{subject} has undeclared ROIs "
                            f"{sorted(extras)}")
                    missing = set(required_rois) - set(subject_rois)
                    if missing:
                        raise ValueError(
                            f"{model_id}.roi_scores.{cohort}.{subject} is missing required ROIs "
                            f"{sorted(missing)}")
                common = common_roi_keys(list(roi_subjects.values()))
                first = next(iter(roi_subjects.values()), {})
                expected_roi_means = {
                    roi: mean_scalar(roi_subjects[subject][roi] for subject in expected_subjects)
                    for roi in first if roi in common
                }
                if roi_scores.get("mean") != expected_roi_means:
                    raise ValueError(
                        f"{model_id}.roi_scores.{cohort}.mean must be {expected_roi_means}, "
                        f"got {roi_scores.get('mean')}")

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
    """Use the current dataset metric; retain the old readout only for legacy language."""
    if getattr(DATASET, "stage", "legacy") != "legacy":
        from harness.metrics import encoding_scores, rsa_scores
        R._validate_contract(DATASET)
        if DATASET.metric == "rsa":
            features = torch.cat([torch.as_tensor(A_train), torch.as_tensor(A_test)])
            targets = torch.cat([torch.as_tensor(Ytr), torch.as_tensor(Yte)])
            result = rsa_scores(features, targets, roi_masks, device)
        else:
            result = encoding_scores(A_train, A_test, Ytr, Yte, roi_masks, device)
        return {**result, "d_out": int(A_test.shape[1]), "metric": DATASET.metric, "stage": DATASET.stage}
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
