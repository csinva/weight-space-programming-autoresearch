"""Score a raw representation (train/test activations already in hand) through the harness
reward-region readout -- the body of ``harness.reward.evaluate_on_split`` for callers that hold
features rather than a MODEL. Same reward-region target selection, same fast/himalaya ridge, same
per-voxel noise ceiling, same ``cv_folds`` branch -- so the number is apples-to-apples with the
search DB. Returns the noise-normalized ``test_corr`` + per-ROI breakdown.
"""
import numpy as np
import torch


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
