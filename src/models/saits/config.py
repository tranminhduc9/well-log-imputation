"""SAITS architecture and training settings."""

from dataclasses import dataclass
from src.models.base import ModelConfig


@dataclass(frozen=True)
class SAITSConfig(ModelConfig):
    n_layers: int = 2
    d_model: int = 256
    d_inner: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    diagonal_attention_mask: bool = True
    ort_weight: float = 1.0
    mit_weight: float = 1.0
    mit_reduction: str = "segment"
    independent_shuffle: bool = False
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mit_reduction not in {"segment", "point"}:
            raise ValueError("mit_reduction must be 'segment' or 'point'.")
        if self.n_layers <= 0 or self.d_model <= 0 or self.d_inner <= 0 or self.n_heads <= 0:
            raise ValueError("SAITS layer, model, feed-forward, and head sizes must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.diagonal_attention_mask and self.seq_len < 2:
            raise ValueError("Diagonal attention requires at least two time steps.")
        if not 0 <= self.dropout < 1 or not 0 <= self.attn_dropout < 1:
            raise ValueError("SAITS dropout rates must be in [0, 1).")
        if self.ort_weight < 0 or self.mit_weight < 0 or self.min_delta < 0:
            raise ValueError("SAITS loss weights and min_delta must be non-negative.")
        if self.ort_weight == self.mit_weight == 0:
            raise ValueError("At least one SAITS loss weight must be positive.")
