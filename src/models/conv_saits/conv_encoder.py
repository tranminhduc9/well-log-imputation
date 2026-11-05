"""Convolutional encoder: depthwise conv -> channel mixing -> variable mixing."""

import torch
from torch import nn
from torch.nn import functional as F
from .config import ConvSAITSConfig
from .ablation import _ChannelLayerNorm


class DecoupledFeatureEncoder(nn.Module):
    """ModernTCN-style encoder from Equations (3)-(5).

    Internally the tensor has shape ``(batch, variables, channels, time)``.
    Temporal depthwise convolution operates independently on every
    variable/channel pair.  ConvFFN1 groups by variable and mixes latent
    channels, while ConvFFN2 groups by latent channel and mixes variables.
    """

    def __init__(self, config: ConvSAITSConfig) -> None:
        super().__init__()
        self.n_features = config.n_features
        self.channels = config.encoder_channels
        self.total_channels = self.n_features * self.channels
        self.temporal_residual = config.temporal_residual
        total_channels = self.total_channels

        # A shared per-variable projection keeps variables decoupled at H0.
        self.input_projection = nn.Linear(2, self.channels)
        self.temporal = None
        self.temporal_branches = None
        self.temporal_fusion = None
        if config.multi_scale_conv:
            kernels = tuple(config.multi_scale_kernels)
            self.temporal_branches = nn.ModuleList(
                nn.Conv1d(total_channels, total_channels, kernel_size=kernel,
                          padding=kernel // 2, groups=total_channels)
                for kernel in kernels
            )
            # Stack as (B, channel, scale, time), so each grouped 1x1 filter
            # combines scales for one latent channel without mixing variables.
            self.temporal_fusion = nn.Conv1d(
                total_channels * len(kernels), total_channels, kernel_size=1,
                groups=total_channels,
            )
        else:
            self.temporal = nn.Conv1d(
                total_channels, total_channels, config.kernel_size,
                padding=config.kernel_size // 2, groups=total_channels,
            )
        self.temporal_norm = (
            nn.BatchNorm1d(total_channels) if config.encoder_norm == "batch"
            else _ChannelLayerNorm(total_channels)
        )

        def grouped_ffn(groups):
            width = total_channels * config.conv_expansion
            return nn.Sequential(
                nn.Conv1d(total_channels, width, 1, groups=groups),
                nn.GELU(), nn.Dropout(config.dropout),
                nn.Conv1d(width, total_channels, 1, groups=groups),
            )

        self.channel_ffn = grouped_ffn(self.n_features)
        self.variable_ffn = grouped_ffn(self.channels)
        self.output_projection = nn.Linear(total_channels, config.d_model)

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        batch, steps, variables = values.shape
        if variables != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, received {variables}."
            )

        # H0: (B, T, M, C), then flatten independent M/C pairs for Conv1d.
        hidden = self.input_projection(torch.stack((values, observed), dim=-1))
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, -1, steps)
        if self.temporal_branches is not None:
            temporal = torch.stack(
                [branch(hidden) for branch in self.temporal_branches], dim=2
            ).reshape(batch, self.total_channels * len(self.temporal_branches), steps)
            temporal = self.temporal_fusion(temporal)
        else:
            temporal = self.temporal(hidden)
        temporal = self.temporal_norm(temporal)
        hidden = hidden + temporal if self.temporal_residual else temporal
        hidden = hidden + F.gelu(self.channel_ffn(hidden))

        # ConvFFN2 needs channels ordered as (latent channel, variable) so each
        # group contains all variables for exactly one latent channel.
        hidden = hidden.reshape(batch, self.n_features, self.channels, steps)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, -1, steps)
        hidden = hidden + F.gelu(self.variable_ffn(hidden))
        hidden = hidden.reshape(batch, self.channels, self.n_features, steps)
        hidden = hidden.permute(0, 3, 2, 1).reshape(batch, steps, -1)
        return self.output_projection(hidden)
