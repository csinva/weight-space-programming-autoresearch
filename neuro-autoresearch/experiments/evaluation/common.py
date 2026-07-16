"""Shared representation scoring and equal-weight subject aggregation helpers.

Score a raw representation (train/test activations already in hand) through the harness
reward-region readout -- the body of ``harness.reward.evaluate_on_split`` for callers that hold
features rather than a MODEL. Same reward-region target selection, same fast/himalaya ridge, same
per-target noise ceiling, same ``cv_folds`` branch -- so the number is apples-to-apples with the
search DB. Returns ``test_corr`` under the active dataset's metric plus its per-ROI breakdown.
"""
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
