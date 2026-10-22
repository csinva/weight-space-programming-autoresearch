"""ENCODING scoring for the complete fixed final cohort."""

from __future__ import annotations

import time

import numpy as np
import torch

from neuro_eval import _evaluation as common
from neuro_eval._evaluation import matrix, region_masks
from neuro_eval.model import _validate_model_size, _readout_device
from neuro_eval.reference.metrics import encoding

ENCODING_FEATURES = "center"


def encoding_scores(train_features, test_features, train_targets, test_targets, masks, device) -> dict:
    """One source ridge fit; training predictions are diagnostics, never CV data."""
    train_features = matrix(train_features, "training model features")
    test_features = matrix(test_features, "test model features")
    train_targets = matrix(train_targets, "training neural responses", rows=len(train_features))
    test_targets = matrix(test_targets, "test neural responses", rows=len(test_features))
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Training and test feature widths differ")
    if train_targets.shape[1] != test_targets.shape[1]:
        raise ValueError("Training and test neural target widths differ")
    n_train, n_test = len(train_features), len(test_features)
    if n_train < encoding.N_FOLDS or n_test < 2:
        raise ValueError("Encoding requires at least five training stimuli and two test stimuli")
    masks = region_masks(masks, train_targets.shape[1])
    features = torch.cat([train_features.to(device), test_features.to(device)])
    targets = torch.cat([train_targets.to(device), test_targets.to(device)])
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    design = encoding.design(features, tr, device, features=ENCODING_FEATURES)
    # Query both splits in one fitted model. Only tr enters the solver/CV and
    # target scaling; source subsets reuse predictions without a second fit.
    _, alphas, scores = encoding.unit_scores(
        design, targets, tr, np.arange(len(features)), subsets={"train": tr, "test": te})
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Encoding returned nonfinite unit scores")
    regions = {name: {split: float(values[mask].mean()) for split, values in scores.items()}
               for name, mask in masks.items()}
    return {"train_corr": float(np.mean([row["train"] for row in regions.values()])),
            "test_corr": float(np.mean([row["test"] for row in regions.values()])),
            "roi_corr": regions, "metric_detail": "ridge_5fold_per_unit_pearson_r",
            "encoding_features": ENCODING_FEATURES, "encoding_cv_folds": encoding.N_FOLDS,
            "encoding_alpha_min": float(alphas.min()), "encoding_alpha_max": float(alphas.max())}


def evaluate_on_split(model, split, dataset):
    common._validate_contract(dataset, "final")
    started = time.time()
    X_train, X_test, Y_train, Y_test, masks, _ = split
    device = _readout_device()
    A_train = common.forward(model, X_train, device, "training model features")
    A_test = common.forward(model, X_test, device, "test model features")
    if A_train.shape[1] != A_test.shape[1]:
        raise ValueError("Model representation width differs between training and test stimuli")
    n_params = _validate_model_size(model, A_train.shape[1])
    scored = encoding_scores(A_train, A_test, Y_train, Y_test, masks, device)
    return common.result_row(model, split, dataset, scored, n_params, started)


def evaluate_on_split_safe(model, split, dataset):
    return common.evaluate_on_split_safe(model, split, dataset, evaluate_on_split)


def evaluate_subjects(model, dataset, *, stage=None):
    if stage is not None and stage != "final":
        return common._error_row(model, None, dataset, ValueError("This evaluator requires final"), 0)
    return common.evaluate_subjects(model, dataset, evaluate_on_split, stage="final")
