"""Conv-SAITS figure: encoder -> DMSA 1 -> DMSA 2 -> weighted combination."""

import torch
from torch import nn
from src.models._attention import (
    AttentionLayer as DiagonallyMaskedAttentionLayer,
    sinusoidal_position as _sinusoidal_position,
)
from .config import ConvSAITSConfig
from .conv_encoder import DecoupledFeatureEncoder
from .ablation import _PerLogDecoder, CrossLogAttention, ConditionalLogDecoder, gap_features


class ConvSAITSNetwork(nn.Module):
    """Two DMSA blocks receiving the enhanced encoder representation Z_enc."""

    def __init__(self, config: ConvSAITSConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DecoupledFeatureEncoder(config)
        self.cross_log_encoder = CrossLogAttention(config) if config.cross_log_attention else None
        self.first_embedding = nn.Linear(config.d_model, config.d_model)
        self.second_embedding = nn.Linear(config.d_model + config.n_features, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.depth_projection = (
            nn.Sequential(nn.Linear(1, config.d_model), nn.Tanh())
            if config.depth_encoding else None
        )
        self.register_buffer("position", _sinusoidal_position(config.seq_len, config.d_model))
        self.first_block = nn.ModuleList(DiagonallyMaskedAttentionLayer(config) for _ in range(config.n_layers))
        self.second_block = nn.ModuleList(DiagonallyMaskedAttentionLayer(config) for _ in range(config.n_layers))
        self.first_output = nn.Linear(config.d_model, config.n_features)
        self.second_output = nn.Sequential(
            nn.Linear(config.d_model, config.n_features),
            nn.ReLU(),
            nn.Linear(config.n_features, config.n_features),
        )
        if config.decoder == "per_log":
            self.second_output = _PerLogDecoder(config)
        elif config.decoder == "conditional":
            self.second_output = ConditionalLogDecoder(config)
        # At each time step: observed mask (M) + attention row (T) -> eta (M).
        self.gate = nn.Linear(
            config.n_features + config.seq_len + (6 * config.n_features if config.gap_aware_gate else 0),
            config.n_features
        )

    def _attention_block(self, hidden, layers):
        attention: torch.Tensor | None = None
        for layer in layers:
            hidden, attention = layer(hidden, self.config.diagonal_attention_mask)
        return hidden, attention

    def forward(self, values, observed, depth=None):
        # 1. Values + mask -> multi-scale convolution encoder -> Z_enc (B, T, D).
        encoded = self.encoder(values, observed)
        if self.cross_log_encoder is not None:
            encoded = encoded + self.cross_log_encoder(values, observed)
        position = self.position
        if self.depth_projection is not None:
            if depth is None:
                raise ValueError("depth is required when depth_encoding=True.")
            if depth.shape != values.shape[:2] or not torch.isfinite(depth).all():
                raise ValueError("depth must be finite with shape (batch, steps).")
            normalized_depth = (depth - self.config.depth_mean) / self.config.depth_std
            position = position + self.depth_projection(normalized_depth.unsqueeze(-1))

        # 2. DMSA 1 receives Z_enc, then produces the first estimate (B, T, M).
        first_hidden = self.embedding_dropout(self.first_embedding(encoded) + position)
        first_hidden, _ = self._attention_block(first_hidden, self.first_block)
        estimate_first = self.first_output(first_hidden)

        filled = observed * values + (1 - observed) * estimate_first
        # 3. DMSA 2 receives [filled values, Z_enc], exactly as in the diagram.
        second_hidden = self.embedding_dropout(
            self.second_embedding(torch.cat((filled, encoded), dim=-1))
            + position
        )
        second_hidden, attention = self._attention_block(second_hidden, self.second_block)
        estimate_second = (self.second_output(second_hidden, values, observed)
                           if self.config.decoder == "conditional" else self.second_output(second_hidden))

        # 4. Combine the two estimates using the mask and mean attention map.
        mean_attention = attention.mean(dim=1)
        gate_input = [observed, mean_attention]
        if self.config.gap_aware_gate:
            gate_input.append(gap_features(observed))
        eta = torch.sigmoid(self.gate(torch.cat(gate_input, -1)))
        estimate = (1 - eta) * estimate_first + eta * estimate_second
        imputation = observed * values + (1 - observed) * estimate
        return imputation, estimate_first, estimate_second, estimate, attention
