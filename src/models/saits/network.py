"""Two DMSA blocks followed by attention-weighted combination (SAITS figure)."""

import torch
from torch import nn
from torch.nn import functional as F
from src.models._attention import AttentionLayer as _AttentionLayer, sinusoidal_position
from .config import SAITSConfig


class SAITSNetwork(nn.Module):
    """Two DMSA blocks with attention-weighted combination of their estimates."""

    def __init__(self, config: SAITSConfig) -> None:
        super().__init__()
        self.config = config
        self.input_first = nn.Linear(2 * config.n_features, config.d_model)
        self.input_second = nn.Linear(2 * config.n_features, config.d_model)
        self.register_buffer("position", sinusoidal_position(config.seq_len, config.d_model))
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.first_block = nn.ModuleList(_AttentionLayer(config) for _ in range(config.n_layers))
        self.second_block = nn.ModuleList(_AttentionLayer(config) for _ in range(config.n_layers))
        self.first_output = nn.Linear(config.d_model, config.n_features)
        self.second_hidden = nn.Linear(config.d_model, config.n_features)
        self.second_output = nn.Linear(config.n_features, config.n_features)
        self.combine = nn.Linear(config.n_features + config.seq_len, config.n_features)

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # 1. DMSA: concatenate values/mask -> Linear -> position -> N layers -> Linear.
        first = self.embedding_dropout(self.input_first(torch.cat((values, observed), -1)) + self.position)
        for layer in self.first_block:
            first, _ = layer(first, self.config.diagonal_attention_mask)
        estimate_first = self.first_output(first)

        completed_first = observed * values + (1 - observed) * estimate_first
        # 2. Replace gaps with estimate 1, concatenate the mask, then run DMSA 2.
        second = self.embedding_dropout(
            self.input_second(torch.cat((completed_first, observed), -1)) + self.position
        )
        for layer in self.second_block:
            second, attention = layer(second, self.config.diagonal_attention_mask)
        estimate_second = self.second_output(F.relu(self.second_hidden(second)))

        # 3. Mean attention over heads + mask -> Linear -> Sigmoid -> weighted sum.
        mean_attention = attention.mean(dim=1)
        weights = torch.sigmoid(self.combine(torch.cat((observed, mean_attention), dim=-1)))
        estimate = weights * estimate_first + (1 - weights) * estimate_second
        # The final Replace operation preserves every observed sample.
        imputation = observed * values + (1 - observed) * estimate
        return imputation, estimate_first, estimate_second, estimate
