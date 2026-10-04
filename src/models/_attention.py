"""Self-attention and positional encoding shared by SAITS and Conv-SAITS."""

import math

import torch
from torch import nn


class AttentionLayer(nn.Module):
    """Multi-head attention with optional diagonal masking and a feed-forward layer."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.attn_dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.norm_feedforward = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.d_inner), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(config.d_inner, config.d_model),
        )

    def forward(self, hidden, diagonal_mask=True):
        batch, steps, _ = hidden.shape

        def split_heads(projected):
            return projected.reshape(batch, steps, self.n_heads, self.head_size).transpose(1, 2)

        query, key, value = [split_heads(layer(hidden)) for layer in (self.query, self.key, self.value)]
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        if diagonal_mask:
            diagonal = torch.eye(steps, device=hidden.device, dtype=torch.bool)
            scores = scores.masked_fill(diagonal[None, None], torch.finfo(scores.dtype).min)
        attention = scores.softmax(-1)
        context = (self.attention_dropout(attention) @ value).transpose(1, 2).reshape(batch, steps, -1)
        hidden = self.norm_attention(hidden + self.dropout(self.output(context)))
        hidden = self.norm_feedforward(hidden + self.dropout(self.feedforward(hidden)))
        return hidden, attention


def sinusoidal_position(seq_len, d_model):
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(1, seq_len, d_model)
    encoding[0, :, 0::2] = torch.sin(positions * frequencies)
    encoding[0, :, 1::2] = torch.cos(positions * frequencies[:d_model // 2])
    return encoding
