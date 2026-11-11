"""Masked waveform MSE/SSIM and missing-log statistics supervision."""

import torch
from torch.nn import functional as F

from src.models.losses import masked_imputation_mse
from .network import log_statistics


def masked_ssim_loss(prediction, truth, mask):
    length = prediction.shape[1]
    size = min(11, length if length % 2 else length - 1)
    offsets = torch.arange(size, dtype=prediction.dtype, device=prediction.device) - size // 2
    kernel = torch.exp(-offsets.square() / (2 * 1.5 ** 2))
    kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
    hidden = mask.transpose(1, 2).flatten(0, 1).unsqueeze(1)
    x = torch.where(mask.bool(), prediction, 0.0).transpose(1, 2).flatten(0, 1).unsqueeze(1)
    y = torch.where(mask.bool(), truth, 0.0).transpose(1, 2).flatten(0, 1).unsqueeze(1)
    support = F.conv1d(hidden, kernel, padding=size // 2).clamp_min(1e-8)

    def average(values):
        return F.conv1d(values, kernel, padding=size // 2) / support

    mean_x, mean_y = average(x), average(y)
    var_x = (average(x.square()) - mean_x.square()).clamp_min(0)
    var_y = (average(y.square()) - mean_y.square()).clamp_min(0)
    covariance = average(x * y) - mean_x * mean_y
    similarity = ((2 * mean_x * mean_y + 0.02 ** 2) * (2 * covariance + 0.06 ** 2)
                  / ((mean_x.square() + mean_y.square() + 0.02 ** 2)
                     * (var_x + var_y + 0.06 ** 2)))
    error = (1 - similarity).reshape(prediction.shape[0], prediction.shape[2], length)
    counts = mask.flatten(1).sum(1)
    per_segment = (error.transpose(1, 2) * mask).flatten(1).sum(1) / counts.clamp_min(1)
    return per_segment.sum() / (counts > 0).sum().clamp_min(1)


def flexlognet_loss(output, truth, observed, hidden, config):
    known = (observed.bool() | hidden.bool()) & torch.isfinite(truth)
    truth = torch.where(known, truth, 0.0)
    target_mean, target_std = log_statistics(truth, known, config.epsilon)
    location = torch.where(output["missing_logs"], target_mean, output["observed_mean"])
    scale = torch.where(output["missing_logs"], target_std, output["observed_std"])
    target = (truth - location) / scale
    waveform = masked_imputation_mse(output["normalized"], target, hidden, reduction="segment")
    if config.ssim_weight:
        waveform = waveform + config.ssim_weight * masked_ssim_loss(output["normalized"], target, hidden)
    supervised = output["missing_logs"].squeeze(1) & hidden.bool().any(1)
    error = (config.mean_weight * (output["mean"] - target_mean.squeeze(1)).abs()
             + (output["std"] - target_std.squeeze(1)).abs())
    statistics = torch.where(supervised, error, 0.0).sum() / supervised.sum().clamp_min(1)
    return waveform + config.statistics_weight * statistics
