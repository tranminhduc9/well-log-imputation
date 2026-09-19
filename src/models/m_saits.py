"""M-SAITS for multivariate well-log imputation.

This module implements the architecture described in ``M-SAITS.pdf``:

* a ModernTCN-inspired encoder which separately models temporal, latent-channel,
  and cross-variable relationships;
* two diagonally-masked self-attention (DMSA) stages; and
* an attention- and mask-conditioned gate which combines the coarse and refined
  estimates.

The paper specifies the data flow and objective, but not every implementation
hyperparameter.  ``encoder_channels`` and ``conv_expansion`` therefore expose
the latent width and ConvFFN expansion used by this implementation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import AbstractModel, ModelConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MSAITSConfig(ModelConfig):
    """Configuration for M-SAITS.

    ``kernel_size=51`` follows the example in Equation (3) of the paper.  The
    kernel is required to be odd so that temporal convolution preserves the
    sequence length exactly.
    """

    n_layers: int = 2
    d_model: int = 256
    d_inner: int = 128
    n_heads: int = 4
    encoder_channels: int = 16
    kernel_size: int = 51
    conv_expansion: int = 2
    dropout: float = 0.1
    attn_dropout: float = 0.1
    masking_rate: float = 0.2
    diagonal_attention_mask: bool = True
    ort_weight: float = 1.0
    mit_weight: float = 1.0
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        positive = {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "d_inner": self.d_inner,
            "n_heads": self.n_heads,
            "encoder_channels": self.encoder_channels,
            "kernel_size": self.kernel_size,
            "conv_expansion": self.conv_expansion,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                "M-SAITS architecture settings must be positive: "
                + ", ".join(invalid)
            )
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.d_model % 2:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        if self.diagonal_attention_mask and self.seq_len < 2:
            raise ValueError("Diagonal attention requires at least two time steps.")
        if not 0 <= self.dropout < 1 or not 0 <= self.attn_dropout < 1:
            raise ValueError("M-SAITS dropout rates must be in [0, 1).")
        if not 0 < self.masking_rate < 1:
            raise ValueError("masking_rate must be in (0, 1).")
        if self.ort_weight < 0 or self.mit_weight < 0 or self.min_delta < 0:
            raise ValueError("M-SAITS loss weights and min_delta must be non-negative.")
        if self.ort_weight == self.mit_weight == 0:
            raise ValueError("At least one M-SAITS loss weight must be positive.")


class DecoupledFeatureEncoder(nn.Module):
    """ModernTCN-style encoder from Equations (3)-(5).

    Internally the tensor has shape ``(batch, variables, channels, time)``.
    Temporal depthwise convolution operates independently on every
    variable/channel pair.  ConvFFN1 groups by variable and mixes latent
    channels, while ConvFFN2 groups by latent channel and mixes variables.
    """

    def __init__(self, config: MSAITSConfig) -> None:
        super().__init__()
        self.n_features = config.n_features
        self.channels = config.encoder_channels
        total_channels = self.n_features * self.channels

        # A shared per-variable projection keeps variables decoupled at H0.
        self.input_projection = nn.Linear(2, self.channels)
        self.temporal = nn.Conv1d(
            total_channels,
            total_channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
            groups=total_channels,
        )
        self.temporal_norm = nn.BatchNorm1d(total_channels)

        channel_hidden = self.n_features * self.channels * config.conv_expansion
        self.channel_ffn = nn.Sequential(
            nn.Conv1d(
                total_channels,
                channel_hidden,
                kernel_size=1,
                groups=self.n_features,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(
                channel_hidden,
                total_channels,
                kernel_size=1,
                groups=self.n_features,
            ),
        )

        variable_hidden = self.channels * self.n_features * config.conv_expansion
        self.variable_ffn = nn.Sequential(
            nn.Conv1d(
                total_channels,
                variable_hidden,
                kernel_size=1,
                groups=self.channels,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(
                variable_hidden,
                total_channels,
                kernel_size=1,
                groups=self.channels,
            ),
        )
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
        hidden = self.temporal_norm(self.temporal(hidden))
        hidden = hidden + F.gelu(self.channel_ffn(hidden))

        # ConvFFN2 needs channels ordered as (latent channel, variable) so each
        # group contains all variables for exactly one latent channel.
        hidden = hidden.reshape(batch, self.n_features, self.channels, steps)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, -1, steps)
        hidden = hidden + F.gelu(self.variable_ffn(hidden))
        hidden = hidden.reshape(batch, self.channels, self.n_features, steps)
        hidden = hidden.permute(0, 3, 2, 1).reshape(batch, steps, -1)
        return self.output_projection(hidden)


class DiagonallyMaskedAttentionLayer(nn.Module):
    """Transformer encoder layer whose attention cannot view the same step."""

    def __init__(self, config: MSAITSConfig) -> None:
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

    def forward(
        self,
        hidden: torch.Tensor,
        diagonal_mask: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = hidden.shape

        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.reshape(
                batch, steps, self.n_heads, self.head_size
            ).transpose(1, 2)

        query = split_heads(self.query(hidden))
        key = split_heads(self.key(hidden))
        value = split_heads(self.value(hidden))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        if diagonal_mask:
            diagonal = torch.eye(steps, dtype=torch.bool, device=hidden.device)
            scores = scores.masked_fill(
                diagonal[None, None], torch.finfo(scores.dtype).min
            )
        attention = scores.softmax(dim=-1)
        context = self.attention_dropout(attention) @ value
        context = context.transpose(1, 2).reshape(batch, steps, -1)
        hidden = self.norm_attention(hidden + self.dropout(self.output(context)))
        hidden = self.norm_feedforward(
            hidden + self.dropout(self.feedforward(hidden))
        )
        return hidden, attention


def _sinusoidal_position(seq_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(1, seq_len, d_model)
    encoding[0, :, 0::2] = torch.sin(positions * frequencies)
    encoding[0, :, 1::2] = torch.cos(positions * frequencies)
    return encoding


class MSAITSNetwork(nn.Module):
    """Dual-stage M-SAITS network from Equations (7)-(9)."""

    def __init__(self, config: MSAITSConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DecoupledFeatureEncoder(config)
        self.first_embedding = nn.Linear(config.d_model, config.d_model)
        self.second_embedding = nn.Linear(
            config.d_model + config.n_features, config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position", _sinusoidal_position(config.seq_len, config.d_model)
        )
        self.first_block = nn.ModuleList(
            DiagonallyMaskedAttentionLayer(config)
            for _ in range(config.n_layers)
        )
        self.second_block = nn.ModuleList(
            DiagonallyMaskedAttentionLayer(config)
            for _ in range(config.n_layers)
        )
        self.first_output = nn.Linear(config.d_model, config.n_features)
        self.second_output = nn.Sequential(
            nn.Linear(config.d_model, config.n_features),
            nn.ReLU(),
            nn.Linear(config.n_features, config.n_features),
        )
        # At each time step: observed mask (M) + attention row (T) -> eta (M).
        self.gate = nn.Linear(
            config.n_features + config.seq_len, config.n_features
        )

    def _attention_block(
        self,
        hidden: torch.Tensor,
        layers: nn.ModuleList,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention: torch.Tensor | None = None
        for layer in layers:
            hidden, attention = layer(
                hidden, self.config.diagonal_attention_mask
            )
        if attention is None:  # guarded by MSAITSConfig.n_layers > 0
            raise RuntimeError("M-SAITS attention block has no layers.")
        return hidden, attention

    def forward(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values, observed)

        first_hidden = self.embedding_dropout(
            self.first_embedding(encoded) + self.position
        )
        first_hidden, _ = self._attention_block(first_hidden, self.first_block)
        estimate_first = self.first_output(first_hidden)

        filled = observed * values + (1 - observed) * estimate_first
        second_hidden = self.embedding_dropout(
            self.second_embedding(torch.cat((filled, encoded), dim=-1))
            + self.position
        )
        second_hidden, attention = self._attention_block(
            second_hidden, self.second_block
        )
        estimate_second = self.second_output(second_hidden)

        mean_attention = attention.mean(dim=1)
        eta = torch.sigmoid(self.gate(torch.cat((observed, mean_attention), -1)))
        estimate = (1 - eta) * estimate_first + eta * estimate_second
        imputation = observed * values + (1 - observed) * estimate
        return imputation, estimate_first, estimate_second, estimate, attention


def _masked_mae(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error over a binary mask, with a zero-safe denominator."""

    return ((prediction - truth).abs() * mask).sum() / mask.sum().clamp_min(1)


class _MSAITSBackend:
    def __init__(self, config: MSAITSConfig) -> None:
        self.config = config
        wants_gpu = config.device.lower() in {"gpu", "cuda"}
        self.device = torch.device(
            "cuda" if wants_gpu and torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(config.seed)
        self.network = MSAITSNetwork(config).to(self.device)
        self.training_history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None

    def _training_masks(
        self,
        train_set: dict[str, Any],
        truth: np.ndarray,
        epoch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        finite_truth = np.isfinite(truth)
        input_observed = np.isfinite(np.asarray(train_set["X"])) & finite_truth
        predefined = train_set.get("indicating_mask")
        if predefined is not None:
            hidden = np.asarray(predefined, dtype=bool) & finite_truth
        else:
            generator = np.random.default_rng(self.config.seed + epoch)
            hidden = (
                generator.random(truth.shape) < self.config.masking_rate
            ) & input_observed
            if not np.any(hidden) and np.any(input_observed):
                candidates = np.flatnonzero(input_observed)
                hidden.flat[generator.choice(candidates)] = True
        observed = input_observed & ~hidden
        return observed, hidden

    def fit(
        self,
        train_set: dict[str, Any],
        val_set: dict[str, Any] | None = None,
    ) -> None:
        truth = np.asarray(
            train_set.get("X_intact", train_set["X"]), dtype=np.float32
        )
        if not np.any(np.isfinite(truth) & np.isfinite(train_set["X"])):
            raise ValueError("M-SAITS training data has no observed values.")

        optimizer = self._make_optimizer()
        best_score = float("inf")
        patience_score = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        self.training_history = []
        self.best_epoch = None

        for epoch in range(self.config.epochs):
            observed, hidden = self._training_masks(train_set, truth, epoch)
            if not np.any(hidden):
                raise ValueError(
                    "M-SAITS requires at least one known value for MIT training."
                )
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
            ort_total = 0.0
            mit_total = 0.0
            for batch_inputs, batch_truth, batch_observed, batch_hidden in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_truth = batch_truth.to(self.device)
                batch_observed = batch_observed.to(self.device)
                batch_hidden = batch_hidden.to(self.device)
                _, first, second, combined, _ = self.network(
                    batch_inputs, batch_observed
                )
                ort = sum(
                    _masked_mae(part, batch_truth, batch_observed)
                    for part in (first, second, combined)
                ) / 3
                mit = _masked_mae(combined, batch_truth, batch_hidden)
                loss = self.config.ort_weight * ort + self.config.mit_weight * mit
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += float(loss.item())
                ort_total += float(ort.item())
                mit_total += float(mit.item())

            batches = len(loader)
            train_loss = loss_total / batches
            score = (
                self._validation_rmse(val_set)
                if val_set is not None
                else train_loss
            )
            if not math.isfinite(score):
                raise RuntimeError(
                    "M-SAITS produced a non-finite selection score at "
                    f"epoch {epoch + 1}."
                )
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": train_loss,
                    "ort_loss": ort_total / batches,
                    "mit_loss": mit_total / batches,
                    "validation_rmse": (
                        score if val_set is not None else float("nan")
                    ),
                }
            )
            LOGGER.info(
                "M-SAITS epoch %d/%d | loss=%.6f | selection_score=%.6f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                score,
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
                    LOGGER.info(
                        "M-SAITS early stopping at epoch %d; best epoch=%d",
                        epoch + 1,
                        self.best_epoch,
                    )
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        configured = self.config.optimizer
        if configured is None:
            return torch.optim.Adam(
                self.network.parameters(), lr=self.config.learning_rate
            )
        if isinstance(configured, torch.optim.Optimizer):
            return configured
        if callable(configured):
            return configured(
                self.network.parameters(), lr=self.config.learning_rate
            )
        raise TypeError("optimizer must be an optimizer instance, callable, or None.")

    def _validation_rmse(self, dataset: dict[str, Any]) -> float:
        scenarios = dataset.get("validation_scenarios")
        if scenarios is not None:
            if not scenarios:
                raise ValueError("M-SAITS validation_scenarios must not be empty.")
            return float(
                np.mean(
                    [self._validation_rmse(data) for data in scenarios.values()]
                )
            )
        if "X_intact" not in dataset or "indicating_mask" not in dataset:
            raise ValueError(
                "M-SAITS validation requires X_intact and indicating_mask."
            )
        truth = np.asarray(dataset["X_intact"], dtype=np.float32)
        values = np.asarray(dataset["X"], dtype=np.float32)
        mask = np.asarray(dataset["indicating_mask"], dtype=bool)
        error_sum = 0.0
        count = 0
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                end = start + self.config.batch_size
                prediction = self._predict_batch(values[start:end])
                valid = mask[start:end] & np.isfinite(truth[start:end])
                difference = prediction[valid] - truth[start:end][valid]
                error_sum += float(np.square(difference.astype(np.float64)).sum())
                count += int(valid.sum())
        if count == 0:
            raise ValueError("M-SAITS validation has no masked values to score.")
        return math.sqrt(error_sum / count)

    def _predict_batch(self, values: np.ndarray) -> np.ndarray:
        batch = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        observed = torch.isfinite(batch).float()
        clean = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        imputation, _, _, _, _ = self.network(clean, observed)
        return imputation.cpu().numpy()

    def predict(self, dataset: dict[str, Any]) -> dict[str, np.ndarray]:
        values = np.asarray(dataset["X"], dtype=np.float32)
        if len(values) == 0:
            return {"imputation": values.copy()}
        self.network.eval()
        with torch.no_grad():
            imputation = np.concatenate(
                [
                    self._predict_batch(values[start : start + self.config.batch_size])
                    for start in range(0, len(values), self.config.batch_size)
                ]
            )
        return {"imputation": imputation}


class MSAITS(AbstractModel):
    """Dual-stage convolution-and-attention imputer for segmented well logs."""

    name = "m_saits"

    def __init__(self, config: MSAITSConfig | None = None) -> None:
        super().__init__(config or MSAITSConfig())

    def _build_backend(self) -> _MSAITSBackend:
        return _MSAITSBackend(self.config)


# Readable compatibility alias for code which mirrors the paper's hyphenation.
M_SAITS = MSAITS


__all__ = [
    "DecoupledFeatureEncoder",
    "DiagonallyMaskedAttentionLayer",
    "MSAITS",
    "MSAITSConfig",
    "MSAITSNetwork",
    "M_SAITS",
]
