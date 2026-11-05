"""Optional normalization, decoders, cross-log attention and validation scoring.

The A0–A5 encoder removals are defined in notebook/ablation.ipynb. Components
here are selected through ConvSAITSConfig for additional experiments.
"""

import numpy as np
import torch
from torch import nn

from src.data.metrics import compute_imputation_metrics


class _ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each depth sample."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


def gap_features(observed):
    """Six mask features per log; distances are in samples divided by length.

    Return left/right distances, left/right availability, observed fraction
    and an entirely-missing flag. A missing neighbour has distance 1.
    """
    steps = observed.shape[1]
    position = torch.arange(steps, device=observed.device).view(1, steps, 1)
    valid = observed.bool()
    left = torch.where(valid, position, -1).cummax(dim=1).values
    right = torch.where(valid, position, steps).flip(1).cummin(dim=1).values.flip(1)
    has_left, has_right = left >= 0, right < steps
    coverage = observed.mean(dim=1, keepdim=True).expand_as(observed)
    return torch.cat([
        torch.where(has_left, (position - left) / steps, 1.),
        torch.where(has_right, (right - position) / steps, 1.),
        has_left.to(observed.dtype), has_right.to(observed.dtype),
        coverage, (coverage == 0).to(observed.dtype),
    ], dim=-1)


class _PerLogDecoder(nn.Module):
    """Decode the shared representation with a separate MLP for each log."""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(config.d_model, config.decoder_width), nn.ReLU(),
                          nn.Linear(config.decoder_width, 1))
            for _ in range(config.n_features)
        )

    def forward(self, hidden):
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


class CrossLogAttention(nn.Module):
    """Attend to observed logs at each depth; missing values never become keys."""

    def __init__(self, config):
        super().__init__()
        width = config.cross_log_width
        self.input = nn.Linear(2, width)
        self.variable = nn.Parameter(torch.randn(config.n_features, width) * .02)
        self.fallback = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(
            width, config.n_heads, dropout=config.attn_dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(config.n_features * width, config.d_model)

    def forward(self, values, observed):
        batch, steps, features = values.shape
        clean = torch.where(observed.bool(), values, 0.)
        tokens = self.input(torch.stack((clean, observed), dim=-1)) + self.variable
        tokens = tokens.reshape(batch * steps, features, -1)
        valid = observed.reshape(batch * steps, features).bool()
        keys = torch.cat((tokens, self.fallback.expand(batch * steps, -1, -1)), dim=1)
        # Use the fallback only when every log at this depth is missing.
        padding = torch.cat((~valid, valid.any(dim=-1, keepdim=True)), dim=-1)
        attended, _ = self.attention(tokens, keys, keys, key_padding_mask=padding, need_weights=False)
        hidden = self.norm(tokens + attended).reshape(batch, steps, -1)
        return self.output(hidden)


class ConditionalLogDecoder(nn.Module):
    """Decode each target using hidden features and the other observed logs."""

    def __init__(self, config):
        super().__init__()
        self.other_logs = [
            [j for j in range(config.n_features) if j != i]
            for i in range(config.n_features)
        ]
        width = config.d_model + 2 * (config.n_features - 1)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(width, config.decoder_width), nn.GELU(),
                          nn.Linear(config.decoder_width, 1))
            for _ in self.other_logs
        )

    def forward(self, hidden, values, observed):
        clean = torch.where(observed.bool(), values, 0.)
        estimates = []
        for head, indices in zip(self.heads, self.other_logs):
            inputs = torch.cat((hidden, clean[..., indices], observed[..., indices]), dim=-1)
            estimates.append(head(inputs))
        return torch.cat(estimates, dim=-1)


def _physical_mape(config, truth, prediction, mask):
    """Compute physical-unit MAPE per log, then average logs equally."""
    scores = []
    for index, (mean, std) in enumerate(zip(config.physical_means, config.physical_stds)):
        selected = mask[..., index].astype(bool)
        if not selected.any():
            raise ValueError('Joint validation requires scored points in every log/scenario')
        target = truth[..., index] * std + mean
        estimate = prediction[..., index] * std + mean
        if config.validation_mape_floor:
            target, estimate = target[selected].astype(np.float64), estimate[selected].astype(np.float64)
            denominator = np.maximum(np.abs(target), config.relative_floors[index])
            score = 100 * np.mean(np.abs(estimate - target) / denominator)
        else:
            score = compute_imputation_metrics(target, estimate, selected)['mape']
        if not np.isfinite(score):
            raise ValueError('Undefined physical MAPE in validation')
        scores.append(score)
    return float(np.mean(scores))


def validation_objectives(config, predict, dataset):
    """Return normalized RMSE and physical MAPE, averaged equally by scenario."""
    scenarios = dataset.get('validation_scenarios', {'single': dataset})
    if not scenarios:
        raise ValueError('Empty validation scenarios')
    scores = []
    for data in scenarios.values():
        inputs = {key: data[key] for key in ('X', 'depth') if key in data}
        prediction = predict(inputs)['imputation']
        truth, mask = data['X_intact'], data['indicating_mask']
        rmse = compute_imputation_metrics(truth, prediction, mask, include_mape=False)['rmse']
        mape = _physical_mape(config, truth, prediction, mask)
        scores.append((rmse, mape))
    return tuple(np.mean(scores, axis=0))
