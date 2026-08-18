"""Standard Attentive Neural Process baseline without a depth penalty."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn

from .attention_neural_process import AttentionNeuralProcess
from .base import AbstractModel


class StandardCrossAttention(nn.Module):
    """Scaled dot-product multi-head attention using content scores only.

    The projections and output shape match ``DepthAwareCrossAttention``.  The
    depth arguments remain in the interface so this module can be exchanged
    directly with the depth-aware version; they are intentionally ignored.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = tensor.shape
        return tensor.reshape(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_depth: torch.Tensor,
        key_depth: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del query_depth, key_depth
        q = self._heads(self.query(query))
        k = self._heads(self.key(key))
        v = self._heads(self.value(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_valid is not None:
            invalid = ~key_valid.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(invalid, torch.finfo(scores.dtype).min)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous()
        attended = attended.reshape(query.shape[0], query.shape[1], self.hidden_dim)
        return self.output(attended), weights


class StandardAttentiveNeuralProcess(AttentionNeuralProcess):
    """ANP whose attention has no explicit depth-distance penalty."""

    model_label = "ANP"

    def __init__(
        self,
        *args: Any,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        attention: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        standard_attention = attention or StandardCrossAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        super().__init__(
            *args,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            attention=standard_attention,
            **kwargs,
        )


class StandardANPModel(AbstractModel):
    """Benchmark adapter for standard ANP without depth-aware attention."""

    name = "anp_standard"

    def _build_backend(self) -> StandardAttentiveNeuralProcess:
        cfg = self.config
        return StandardAttentiveNeuralProcess(
            n_steps=cfg.seq_len,
            n_features=cfg.n_features,
            hidden_dim=128,
            latent_dim=32,
            n_heads=4,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            patience=cfg.patience,
            learning_rate=min(cfg.learning_rate, 3e-4),
            kl_weight=3e-3,
            kl_warmup_epochs=40,
            min_scale=3e-2,
            max_scale=3.0,
            prediction_samples=16,
            device=cfg.device,
            saving_path=cfg.output_dir / self.name,
        )
