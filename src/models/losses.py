"""Shared masked objectives for artificial-gap supervision."""

import torch


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
