"""Basic Neural Process baseline for well-log imputation.

This baseline deliberately reuses the latent path, decoder, objective,
training loop, and predictive-interval code of the depth-aware ANP.  Its only
architectural change is the original NP deterministic aggregation: context
representations are averaged into one global vector instead of being queried
with cross-attention.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .attention_neural_process import AttentionNeuralProcess, TensorDict
from .base import AbstractModel


class NeuralProcess(AttentionNeuralProcess):
    """Neural Process with permutation-invariant mean context aggregation."""

    model_label = "NP"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The base implementation owns all common NP/ANP machinery.  Plain NP
        # does not need a depth-query encoder or an attention module.
        kwargs["attention"] = nn.Identity()
        super().__init__(*args, **kwargs)
        self.depth_encoder = nn.Identity()

    def _apply_attention(
        self,
        context: TensorDict,
        target: TensorDict,
        representation: TensorDict,
    ) -> TensorDict:
        """Replace cross-attention with the canonical NP mean aggregator."""
        point_valid = (context["context_mask"].sum(dim=-1, keepdim=True) > 0).to(
            representation["points"].dtype
        )
        pooled = (representation["points"] * point_valid).sum(dim=1)
        pooled = pooled / point_valid.sum(dim=1).clamp_min(1.0)
        deterministic = pooled.unsqueeze(1).expand(
            -1, target["target_depth"].shape[1], -1
        )
        return {
            **representation,
            "deterministic": deterministic,
            "attention_weights": None,
        }


class NPModel(AbstractModel):
    """Benchmark adapter for the basic Neural Process baseline."""

    name = "np"

    def _build_backend(self) -> NeuralProcess:
        cfg = self.config
        return NeuralProcess(
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
