"""Metrics for evaluating imputed well-log values."""

import numpy as np


def compute_imputation_metrics(
    truth: np.ndarray,
    imputation: np.ndarray,
    indicating_mask: np.ndarray,
    *,
    include_mape: bool = True,
) -> dict[str, float | int]:
    """Calculate metrics only where values were artificially removed."""

    truth = np.asarray(truth, dtype=np.float64)
    imputation = np.asarray(imputation, dtype=np.float64)
    indicating_mask = np.asarray(indicating_mask)
    if truth.shape != imputation.shape or truth.shape != indicating_mask.shape:
        raise ValueError("Truth, imputation and indicating_mask must have identical shapes.")
    if not np.isin(indicating_mask, [0, 1]).all():
        raise ValueError("indicating_mask must be binary.")
    valid = indicating_mask.astype(bool) & np.isfinite(truth)
    if not np.isfinite(imputation[valid]).all():
        raise ValueError("Non-finite predictions at scored positions.")
    if not np.any(valid):
        raise ValueError("No valid values are available for evaluation.")

    target = truth[valid]
    prediction = imputation[valid]
    error = prediction - target
    mse = np.mean(error**2)
    denominator = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - np.sum(error**2) / denominator if denominator > 0 else np.nan
    nonzero = target != 0
    mape = (
        np.mean(np.abs(error[nonzero] / target[nonzero])) * 100
        if np.any(nonzero)
        else np.nan
    )

    result = {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "count": int(len(target)),
    }
    if include_mape:
        result["mape"] = float(mape)
    return result
