"""FlexLogNet HGNN/U-Net and FCN heads, using dense four-log graphs.

Reference: Dai et al., Computers & Geosciences 191 (2024), 105666.
Graph attention follows the time-axis projections in the authors' model_AIgeo.py.
See README.md for the adaptations to this repository's missingness protocol.
"""

import math

import numpy as np
from scipy.interpolate import interp1d
import torch
from torch import nn
from torch.nn import functional as F

from .config import FlexLogNetConfig


def log_statistics(values, observed, epsilon):
    values = torch.where(observed.bool(), values, 0.0)
    count = observed.sum(1, keepdim=True)
    mean = values.sum(1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(observed.bool(), values - mean, 0.0)
    variance = centered.square().sum(1, keepdim=True) / count.clamp_min(1)
    std = variance.clamp_min(epsilon ** 2).sqrt()
    std = torch.where((count > 1) & (variance > epsilon ** 2), std, 1.0)
    return mean, std


def conv_block(in_channels, out_channels, kernel_size, repeats=2):
    layers = []
    for _ in range(repeats):
        layers.extend((nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(out_channels), nn.LeakyReLU(0.2)))
        in_channels = out_channels
    return nn.Sequential(*layers)


class MultiscaleInput(nn.Module):
    def __init__(self, seq_len, factors):
        super().__init__()
        self.factors = factors
        for factor in factors:
            count = math.ceil(seq_len / factor)
            if count == 1:
                matrix = np.ones((seq_len, 1))
            else:
                matrix = interp1d(np.linspace(0, 1, count), np.eye(count), axis=0,
                                  kind="quadratic" if count >= 3 else "linear")(
                                      np.linspace(0, 1, seq_len))
            self.register_buffer(f"interpolation_{factor}", torch.tensor(matrix, dtype=torch.float32))

    def forward(self, values):
        scales = [values]
        for factor in self.factors:
            padded = F.pad(values, (0, (-values.shape[-1]) % factor))
            blocks = padded.unfold(-1, factor, factor).sort(-1).values
            median = (blocks[..., (factor - 1) // 2] + blocks[..., factor // 2]) / 2
            scales.append(median @ getattr(self, f"interpolation_{factor}").T)
        return torch.stack(scales, dim=2)


class HeterogeneousAggregation(nn.Module):
    """Directed non-self edges with per-channel attention and per-edge depth maps."""

    def __init__(self, n_features, length, dropout):
        super().__init__()
        self.n_features = n_features
        self.query = nn.Linear(length, length)
        self.key = nn.Linear(length, length)
        edges = [(target, source) for target in range(n_features)
                 for source in range(n_features) if target != source]
        self.register_buffer("targets", torch.tensor([edge[0] for edge in edges]))
        self.register_buffer("sources", torch.tensor([edge[1] for edge in edges]))
        self.depth_maps = nn.Parameter(torch.empty(len(edges), length, length))
        nn.init.uniform_(self.depth_maps, -math.sqrt(3 / length), math.sqrt(3 / length))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, available):
        batch, features, channels, length = values.shape
        query, key = self.query(values), self.key(values)
        scores = (query[:, self.targets] * key[:, self.sources]).sum(-1) / math.sqrt(length)
        scores = scores.reshape(batch, features, features - 1, channels)
        valid = available[:, self.sources].reshape(batch, features, features - 1, 1)
        weights = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(dim=2)
        weights = weights * valid
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
        weights = self.dropout(weights).reshape(batch, -1, channels, 1)
        messages = torch.einsum("becl,elm->becm", values[:, self.sources] * weights, self.depth_maps)
        return messages.reshape(batch, features, features - 1, channels, length).sum(2)


class LogEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool1d(5, stride=2, padding=2)

    def forward(self, values):
        skip = self.conv(values)
        return self.pool(skip), skip


class LogDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = conv_block(in_channels, 2 * out_channels, kernel_size)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(2 * out_channels, out_channels, kernel_size, stride=2,
                               padding=kernel_size // 2, output_padding=1),
            nn.BatchNorm1d(out_channels), nn.LeakyReLU(0.2),
        )

    def forward(self, values):
        return self.up(self.conv(values))


class FlexLogNetNetwork(nn.Module):
    def __init__(self, config: FlexLogNetConfig):
        super().__init__()
        self.config = config
        features = config.n_features
        channels = (1 + len(config.resample_factors), *config.encoder_channels)
        lengths = [math.ceil(config.seq_len / 2 ** level) for level in range(len(channels))]
        self.multiscale = MultiscaleInput(config.seq_len, config.resample_factors)
        self.encoders = nn.ModuleList(nn.ModuleList(
            LogEncoder(before, after, config.kernel_size) for _ in range(features)
        ) for before, after in zip(channels[:-1], channels[1:]))
        self.decoders, self.fusions, self.aggregations = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        previous = channels[-1]
        for level in reversed(range(len(config.encoder_channels))):
            width = channels[level + 1]
            self.decoders.append(nn.ModuleList(
                LogDecoder(previous, width, config.kernel_size) for _ in range(features)))
            self.fusions.append(conv_block(2 * width, width, config.kernel_size, repeats=1))
            self.aggregations.append(HeterogeneousAggregation(features, lengths[level], config.dropout))
            previous = width
        width = config.encoder_channels[0]
        self.output = nn.Sequential(conv_block(width, width, config.kernel_size),
                                    nn.Conv1d(width, 1, config.kernel_size, padding=config.kernel_size // 2))
        hidden = config.stats_hidden
        sizes = (features * config.seq_len, features * hidden, features * hidden,
                 features * max(hidden // 4, 1), features * max(hidden // 16, 1),
                 features * max(hidden // 16, 1), 2 * features)
        layers = []
        for index, (before, after) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(before, after))
            if index < len(sizes) - 2:
                layers.append(nn.LeakyReLU(0.2))
        self.statistics = nn.Sequential(*layers)

    def forward(self, values, observed):
        observed = observed.bool() & torch.isfinite(values)
        values = torch.where(observed, values, 0.0)
        observed_mean, observed_std = log_statistics(values, observed, self.config.epsilon)
        normalized = torch.where(observed, (values - observed_mean) / observed_std, 0.0)
        encoded = self.multiscale(normalized.transpose(1, 2))
        skips = []
        for encoders in self.encoders:
            outputs = [encoder(encoded[:, log]) for log, encoder in enumerate(encoders)]
            encoded = torch.stack([output[0] for output in outputs], dim=1)
            skips.append(torch.stack([output[1] for output in outputs], dim=1))
        available = observed.any(dim=1)
        for decoders, fusion, aggregation, skip in zip(
            self.decoders, self.fusions, self.aggregations, reversed(skips)
        ):
            decoded = torch.stack([decoder(encoded[:, log]) for log, decoder in enumerate(decoders)], dim=1)
            decoded = decoded[..., :skip.shape[-1]]
            batch, features, _, length = decoded.shape
            fused = fusion(torch.cat((skip, decoded), dim=2).flatten(0, 1))
            encoded = aggregation(fused.reshape(batch, features, -1, length), available)
            available = torch.ones_like(available)
        normalized_prediction = self.output(encoded.flatten(0, 1)).reshape(
            values.shape[0], self.config.n_features, self.config.seq_len).transpose(1, 2)
        mean, raw_std = self.statistics(values.transpose(1, 2).flatten(1)).chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.config.epsilon
        missing_logs = ~observed.any(dim=1, keepdim=True)
        location = torch.where(missing_logs, mean.unsqueeze(1), observed_mean)
        scale = torch.where(missing_logs, std.unsqueeze(1), observed_std)
        reconstruction = normalized_prediction * scale + location
        return {
            "imputation": torch.where(observed, values, reconstruction),
            "normalized": normalized_prediction, "mean": mean, "std": std,
            "observed_mean": observed_mean, "observed_std": observed_std,
            "missing_logs": missing_logs,
        }
