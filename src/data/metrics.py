"""Metrics for evaluating imputed well-log values."""

import numpy as np


def compute_imputation_metrics(
    truth: np.ndarray,
    imputation: np.ndarray,
    indicating_mask: np.ndarray,
) -> dict[str, float | int]:
    """Calculate metrics only where values were artificially removed."""

    valid = (
        indicating_mask.astype(bool)
        & np.isfinite(truth)
        & np.isfinite(imputation)
    )
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

    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mape": float(mape),
        "r2": float(r2),
        "count": int(len(target)),
    }
