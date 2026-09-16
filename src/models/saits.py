"""SAITS imputation adapted to the project's ``AbstractModel`` interface.

The architecture follows the two attention blocks and ORT/MIT objectives used
by the well-log benchmark's PyPOTS SAITS model. This implementation only needs
PyTorch, which is already a project dependency.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import AbstractModel, ModelConfig
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
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
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


class _AttentionLayer(nn.Module):
    """Transformer encoder layer that also exposes attention weights."""

    def __init__(self, config: SAITSConfig) -> None:
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
            nn.Linear(config.d_model, config.d_inner),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_inner, config.d_model),
        )

    def forward(self, hidden: torch.Tensor, diagonal_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = hidden.shape

        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.reshape(batch, steps, self.n_heads, self.head_size).transpose(1, 2)

        query = split_heads(self.query(hidden))
        key = split_heads(self.key(hidden))
        value = split_heads(self.value(hidden))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        if diagonal_mask:
            diagonal = torch.eye(steps, device=hidden.device, dtype=torch.bool)
            scores = scores.masked_fill(diagonal[None, None], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        context = self.attention_dropout(attention) @ value
        context = context.transpose(1, 2).reshape(batch, steps, -1)
        hidden = self.norm_attention(hidden + self.dropout(self.output(context)))
        hidden = self.norm_feedforward(hidden + self.dropout(self.feedforward(hidden)))
        return hidden, attention


class SAITSNetwork(nn.Module):
    """Two DMSA blocks with attention-weighted combination of their estimates."""

    def __init__(self, config: SAITSConfig) -> None:
        super().__init__()
        self.config = config
        self.input_first = nn.Linear(2 * config.n_features, config.d_model)
        self.input_second = nn.Linear(2 * config.n_features, config.d_model)
        positions = torch.arange(config.seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, config.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.d_model)
        )
        position = torch.zeros(1, config.seq_len, config.d_model)
        position[0, :, 0::2] = torch.sin(positions * frequencies)
        position[0, :, 1::2] = torch.cos(positions * frequencies[: config.d_model // 2])
        self.register_buffer("position", position)
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


def _masked_mae(prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - truth).abs() * mask).sum() / mask.sum().clamp_min(1)


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
        best_score = float("inf")
        patience_score = float("inf")
        best_state = None
        stale_epochs = 0
        self.training_history = []
        self.best_epoch = None

        for epoch in range(self.config.epochs):
            # Use fresh pseudo gaps when the caller provides intact training data.
            if predefined_mask is None:
                hidden = create_missing_mask(truth, random_state=self.config.seed + epoch)
                hidden &= original_observed & np.isfinite(truth)
            else:
                hidden = predefined_mask & np.isfinite(truth)
            observed = original_observed & np.isfinite(truth) & ~hidden
            if not np.any(hidden):
                raise ValueError("SAITS requires at least one known value to mask for MIT training.")
            inputs = np.where(observed, truth, 0).astype(np.float32)
            targets = np.where(np.isfinite(truth), truth, 0).astype(np.float32)
            loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(inputs),
                    torch.from_numpy(targets),
                    torch.from_numpy(observed.astype(np.float32)),
                    torch.from_numpy(hidden.astype(np.float32)),
                ),
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            self.network.train()
            loss_total = 0.0
            for batch_inputs, batch_truth, batch_observed, batch_hidden in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_truth = batch_truth.to(self.device)
                batch_observed = batch_observed.to(self.device)
                batch_hidden = batch_hidden.to(self.device)
                _, first, second, combined = self.network(batch_inputs, batch_observed)
                ort = sum(
                    _masked_mae(part, batch_truth, batch_observed)
                    for part in (first, second, combined)
                ) / 3
                mit = _masked_mae(combined, batch_truth, batch_hidden)
                loss = self.config.ort_weight * ort + self.config.mit_weight * mit
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += loss.item()

            train_loss = loss_total / len(loader)
            score = self._validation_rmse(val_set) if val_set is not None else train_loss
            if not math.isfinite(score):
                raise RuntimeError(f"SAITS produced a non-finite selection score at epoch {epoch + 1}.")
            self.training_history.append({
                "epoch": epoch + 1,
                "loss": train_loss,
                "validation_rmse": score if val_set is not None else float("nan"),
            })
            LOGGER.info(
                "SAITS epoch %d/%d | loss=%.6f | selection_score=%.6f",
                epoch + 1, self.config.epochs, train_loss, score,
            )
            if score < best_score:
                best_score = score
                best_state = deepcopy(self.network.state_dict())
                self.best_epoch = epoch + 1
            if score < patience_score - self.config.min_delta:
                patience_score = score
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    LOGGER.info("SAITS early stopping at epoch %d; best epoch=%d", epoch + 1, self.best_epoch)
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _validation_rmse(self, dataset: dict) -> float:
        scenarios = dataset.get("validation_scenarios")
        if scenarios is not None:
            if not scenarios:
                raise ValueError("SAITS validation_scenarios must not be empty.")
            return float(np.mean([self._validation_rmse(data) for data in scenarios.values()]))
        if "X_intact" not in dataset or "indicating_mask" not in dataset:
            raise ValueError("SAITS validation requires X_intact and indicating_mask.")
        truth = np.asarray(dataset["X_intact"], dtype=np.float32)
        inputs = np.asarray(dataset["X"], dtype=np.float32)
        mask = np.asarray(dataset["indicating_mask"], dtype=bool)
        error_sum = 0.0
        count = 0
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(truth), self.config.batch_size):
                end = start + self.config.batch_size
                prediction = self._predict_batch(inputs[start:end])
                valid = mask[start:end] & np.isfinite(truth[start:end])
                difference = prediction[valid] - truth[start:end][valid]
                error_sum += float(np.square(difference.astype(np.float64)).sum())
                count += int(valid.sum())
        if count == 0:
            raise ValueError("SAITS validation has no masked values to score.")
        return math.sqrt(error_sum / count)

    def _predict_batch(self, values: np.ndarray) -> np.ndarray:
        batch = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        observed = torch.isfinite(batch).float()
        imputation, _, _, _ = self.network(
            torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0), observed
        )
        return imputation.cpu().numpy()

    def predict(self, dataset: dict) -> dict[str, np.ndarray]:
        values = np.asarray(dataset["X"], dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            imputation = np.concatenate(
                [
                    self._predict_batch(values[start : start + self.config.batch_size])
                    for start in range(0, len(values), self.config.batch_size)
                ]
            )
        return {"imputation": imputation}


class SAITS(AbstractModel):
    """Self-attention imputation for segmented well logs."""

    name = "saits"

    def __init__(self, config: SAITSConfig | None = None) -> None:
        super().__init__(config or SAITSConfig())

    def _build_backend(self) -> _SAITSBackend:
        return _SAITSBackend(self.config)
