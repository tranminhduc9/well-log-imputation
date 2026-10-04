"""SAITS imputation adapted to the project's ``AbstractModel`` interface.

The architecture follows the two attention blocks and ORT/MIT objectives used
by the well-log benchmark's PyPOTS SAITS model. This implementation only needs
PyTorch, which is already a project dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.model import AbstractModel, ModelConfig
from src.models.losses import masked_imputation_mae, masked_reconstruction_mae as _masked_mae
from src.models._attention import AttentionLayer as _AttentionLayer, sinusoidal_position
from src.models._training import (
    EarlyStopping, training_loader, validation_metrics, predict_batches, synchronized_time,
)
from src.preprocessing.pipeline import create_missing_mask


LOGGER = logging.getLogger(__name__)


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
        first = self.embedding_dropout(self.input_first(torch.cat((values, observed), -1)) + self.position)
        for layer in self.first_block:
            first, _ = layer(first, self.config.diagonal_attention_mask)
        estimate_first = self.first_output(first)

        completed_first = observed * values + (1 - observed) * estimate_first
        second = self.embedding_dropout(
            self.input_second(torch.cat((completed_first, observed), -1)) + self.position
        )
        for layer in self.second_block:
            second, attention = layer(second, self.config.diagonal_attention_mask)
        estimate_second = self.second_output(F.relu(self.second_hidden(second)))

        mean_attention = attention.mean(dim=1)
        weights = torch.sigmoid(self.combine(torch.cat((observed, mean_attention), dim=-1)))
        estimate = weights * estimate_first + (1 - weights) * estimate_second
        imputation = observed * values + (1 - observed) * estimate
        return imputation, estimate_first, estimate_second, estimate


class _SAITSBackend:
    def __init__(self, config: SAITSConfig) -> None:
        self.config = config
        wants_gpu = config.device.lower() in {"gpu", "cuda"}
        self.device = torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self.network = SAITSNetwork(config).to(self.device)
        self.training_history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None

    def fit(self, train_set: dict, val_set: dict | None = None) -> None:
        truth = np.asarray(train_set.get("X_intact", train_set["X"]), dtype=np.float32)
        original_observed = np.isfinite(np.asarray(train_set["X"]))
        predefined_mask = train_set.get("indicating_mask")
        if predefined_mask is not None:
            predefined_mask = np.asarray(predefined_mask, dtype=bool)
        if not np.any(np.isfinite(truth) & original_observed):
            raise ValueError("SAITS training data has no observed values.")

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        stopping = EarlyStopping(self.config.patience, self.config.min_delta)
        self.training_history = []
        self.best_epoch = None

        for epoch in range(self.config.epochs):
            epoch_started = synchronized_time(self.device)
            # Use fresh pseudo gaps when the caller provides intact training data.
            if predefined_mask is None:
                hidden = create_missing_mask(truth, random_state=self.config.seed + epoch)
                hidden &= original_observed & np.isfinite(truth)
            else:
                hidden = predefined_mask & np.isfinite(truth)
            observed = original_observed & np.isfinite(truth) & ~hidden
            if not np.any(hidden):
                raise ValueError("SAITS requires at least one known value to mask for MIT training.")
            loader = training_loader(truth, observed, hidden, self.config.batch_size)
            self.network.train()
            loss_total = 0.0
            for batch_inputs, batch_truth, batch_observed, batch_hidden in loader:
                batch_inputs, batch_truth, batch_observed, batch_hidden = (
                    tensor.to(self.device) for tensor in (batch_inputs, batch_truth, batch_observed, batch_hidden)
                )
                _, first, second, combined = self.network(batch_inputs, batch_observed)
                ort = sum(
                    _masked_mae(part, batch_truth, batch_observed)
                    for part in (first, second, combined)
                ) / 3
                mit = masked_imputation_mae(
                    combined, batch_truth, batch_hidden, self.config.mit_reduction
                )
                loss = self.config.ort_weight * ort + self.config.mit_weight * mit
                if not torch.isfinite(loss):
                    raise RuntimeError("SAITS training loss is non-finite.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += loss.item()

            train_finished = synchronized_time(self.device)
            train_loss = loss_total / len(loader)
            score = self._validation_rmse(val_set) if val_set is not None else train_loss
            if not math.isfinite(score):
                raise RuntimeError(f"SAITS produced a non-finite selection score at epoch {epoch + 1}.")
            validation_finished = synchronized_time(self.device)
            self.training_history.append({
                "train_seconds": train_finished - epoch_started,
                "validation_seconds": validation_finished - train_finished,
                "epoch_seconds": validation_finished - epoch_started,
                "epoch": epoch + 1,
                "loss": train_loss,
                "validation_rmse": score if val_set is not None else float("nan"),
            })
            LOGGER.info(
                "SAITS epoch %d/%d | loss=%.6f | selection_score=%.6f",
                epoch + 1, self.config.epochs, train_loss, score,
            )
            stopped = stopping.update(score, self.network, epoch + 1)
            self.best_epoch = stopping.best_epoch
            if stopped:
                LOGGER.info("SAITS early stopping at epoch %d; best epoch=%d", epoch + 1, self.best_epoch)
                break
        stopping.restore(self.network)

    def _validation_rmse(self, dataset):
        return validation_metrics(dataset, self.predict, self.config.batch_size, "SAITS")["rmse"]

    def _predict_batch(self, values: np.ndarray, depth=None) -> np.ndarray:
        batch = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        observed = torch.isfinite(batch).float()
        imputation, _, _, _ = self.network(
            torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0), observed
        )
        return imputation.cpu().numpy()

    def predict(self, dataset):
        return predict_batches(self.network, self._predict_batch, dataset, self.config.batch_size)


class SAITS(AbstractModel):
    """Self-attention imputation for segmented well logs."""

    name = "saits"

    def __init__(self, config: SAITSConfig | None = None) -> None:
        super().__init__(config or SAITSConfig())

    def _build_backend(self) -> _SAITSBackend:
        return _SAITSBackend(self.config)
