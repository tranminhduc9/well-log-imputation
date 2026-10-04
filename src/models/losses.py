"""Shared masked objectives for artificial-gap supervision."""

import torch


def masked_reconstruction_mae(prediction, truth, mask):
    """Pooled MAE on observations (ORT), distinct from segment-weighted MIT."""
    return ((prediction - truth).abs() * mask).sum() / mask.sum().clamp_min(1)


def masked_imputation_mae(prediction, truth, mask, reduction="segment"):
    """Average each segment's masked MAE before averaging valid segments.

    With uniformly sampled scenarios this gives each scenario equal expected
    weight, independent of gap length. ``point`` retains the original pooled
    objective for ablations. Empty segments contribute neither loss nor count.
    """
    error = torch.where(mask.bool(), (prediction - truth).abs(), 0.0)
    if reduction == "point":
        return error.sum() / mask.sum().clamp_min(1)
    if reduction != "segment":
        raise ValueError("MIT reduction must be 'segment' or 'point'.")
    counts = mask.flatten(1).sum(1)
    losses = error.flatten(1).sum(1) / counts.clamp_min(1)
    return losses.sum() / (counts > 0).sum().clamp_min(1)


def masked_imputation_mse(prediction, truth, mask, reduction="segment"):
    """Squared error with the same gap/segment weighting as MIT MAE.

    Select the residual before squaring so unscored NaNs cannot contaminate
    the gradient. Empty segments are excluded from the segment average.
    """
    residual = torch.where(mask.bool(), prediction - truth, 0.0)
    error = residual.square()
    if reduction == "point":
        return error.sum() / mask.sum().clamp_min(1)
    if reduction != "segment":
        raise ValueError("MIT reduction must be 'segment' or 'point'.")
    counts = mask.flatten(1).sum(1)
    losses = error.flatten(1).sum(1) / counts.clamp_min(1)
    return losses.sum() / (counts > 0).sum().clamp_min(1)
