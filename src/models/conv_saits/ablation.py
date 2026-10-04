"""Optional Conv-SAITS variants for architectural ablations."""

import math

import numpy as np
import torch
from torch import nn


class _ChannelLayerNorm(nn.Module):
    """Normalize latent channels at each sample, never across depth/batch."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


def gap_features(observed):
    """Mask-only features: index distances, availability, coverage, entire gap.

    Distances are sample-index distances / T, NOT physical depth. Missing-side
    distances use 1 plus an explicit availability flag, including Entire-Log.
    """
    _, steps, _ = observed.shape
    index = torch.arange(steps, device=observed.device).view(1, steps, 1)
    valid = observed.bool()
    left = torch.where(valid, index, -1).cummax(dim=1).values
    right = torch.where(valid, index, steps).flip(1).cummin(dim=1).values.flip(1)
    has_left, has_right = left >= 0, right < steps
    left_distance = torch.where(has_left, (index - left) / steps, 1.)
    right_distance = torch.where(has_right, (right - index) / steps, 1.)
    coverage = observed.mean(1, keepdim=True).expand_as(observed)
    return torch.cat((left_distance, right_distance, has_left.to(observed.dtype),
                      has_right.to(observed.dtype), coverage, (coverage == 0).to(observed.dtype)), -1)


class _PerLogDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(config.d_model, config.decoder_width), nn.ReLU(),
                          nn.Linear(config.decoder_width, 1))
            for _ in range(config.n_features)
        )

    def forward(self, hidden):
        return torch.cat([head(hidden) for head in self.heads], -1)


class CrossLogAttention(nn.Module):
    """Attention over observed LOGS at each depth, never over hidden targets."""
    def __init__(self, config):
        super().__init__()
        width = config.cross_log_width
        self.input = nn.Linear(2, width)
        self.variable = nn.Parameter(torch.randn(config.n_features, width) * .02)
        self.fallback = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(width, config.n_heads,
                                               dropout=config.attn_dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(config.n_features * width, config.d_model)

    def forward(self, values, observed):
        batch, steps, features = values.shape
        clean = torch.where(observed.bool(), values, 0.)
        tokens = self.input(torch.stack((clean, observed), -1)) + self.variable
        tokens = tokens.reshape(batch * steps, features, -1)
        valid = observed.reshape(batch * steps, features).bool()
        keys = torch.cat((tokens, self.fallback.expand(batch * steps, -1, -1)), 1)
        # Sentinel prevents all-masked attention NaNs; active only when no log is observed.
        padding = torch.cat((~valid, valid.any(-1, keepdim=True)), -1)
        attended, _ = self.attention(tokens, keys, keys, key_padding_mask=padding, need_weights=False)
        return self.output(self.norm(tokens + attended).reshape(batch, steps, -1))


class ConditionalLogDecoder(nn.Module):
    """A target-specific head with direct access to other observed logs/masks."""
    def __init__(self, config):
        super().__init__()
        self.other_logs = [tuple(j for j in range(config.n_features) if j != i)
                           for i in range(config.n_features)]
        self.heads = nn.ModuleList(nn.Sequential(
            nn.Linear(config.d_model + 2 * (config.n_features - 1), config.decoder_width),
            nn.GELU(), nn.Linear(config.decoder_width, 1)) for _ in self.other_logs)

    def forward(self, hidden, values, observed):
        clean = torch.where(observed.bool(), values, 0.)
        return torch.cat([head(torch.cat((hidden, clean[..., list(indices)],
                                         observed[..., list(indices)]), -1))
                          for head, indices in zip(self.heads, self.other_logs)], -1)


def validation_objectives(config, predict, dataset):
    """Equal scenario RMSE; equal log then scenario physical MAPE (exact-zero exclusion)."""
    from src.data.metrics import compute_imputation_metrics
    scenarios = dataset.get("validation_scenarios", {"single": dataset})
    scores = []
    for data in scenarios.values():
        model_input = {"X": data["X"]}
        if "depth" in data:
            model_input["depth"] = data["depth"]
        prediction = predict(model_input)["imputation"]
        truth, mask = data["X_intact"], data["indicating_mask"]
        rmse = compute_imputation_metrics(truth, prediction, mask, include_mape=False)["rmse"]
        mapes = []
        for index, (mean, std) in enumerate(zip(config.physical_means, config.physical_stds)):
            if not mask[..., index].any():
                raise ValueError("Joint validation requires scored points in every log/scenario")
            target = truth[..., index] * std + mean
            estimate = prediction[..., index] * std + mean
            if config.validation_mape_floor:
                selected = mask[..., index].astype(bool)
                physical_target = target[selected].astype(np.float64)
                value = 100 * np.mean(np.abs(estimate[selected].astype(np.float64)-physical_target) /
                    np.maximum(np.abs(physical_target), config.relative_floors[index]))
            else:
                value = compute_imputation_metrics(target, estimate, mask[..., index])["mape"]
            if not math.isfinite(value):
                raise ValueError("Undefined physical MAPE in validation")
            mapes.append(value)
        scores.append((rmse, float(np.mean(mapes))))
    if not scores:
        raise ValueError("Empty validation scenarios")
    return tuple(np.mean(scores, axis=0))
