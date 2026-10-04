"""Conv-SAITS: decoupled temporal encoder, two DMSA stages and masked fusion."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.model import AbstractModel
from .config import ConvSAITSConfig
from src.models.losses import (
    masked_imputation_mae, masked_imputation_mse,
    masked_reconstruction_mae as _masked_mae,
)
from src.models._attention import (
    AttentionLayer as DiagonallyMaskedAttentionLayer,
    sinusoidal_position as _sinusoidal_position,
)
from src.models._training import (
    EarlyStopping, training_loader, validation_metrics, predict_batches, synchronized_time,
)
from .ablation import (
    _ChannelLayerNorm, _PerLogDecoder, CrossLogAttention, ConditionalLogDecoder,
    gap_features, validation_objectives,
)
from .training_state import resume_training, save_training
from src.preprocessing.pipeline import create_missing_mask


LOGGER = logging.getLogger(__name__)


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


class ConvSAITSNetwork(nn.Module):
    """Dual-stage Conv-SAITS network from Equations (7)-(9)."""

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

        first_hidden = self.embedding_dropout(self.first_embedding(encoded) + position)
        first_hidden, _ = self._attention_block(first_hidden, self.first_block)
        estimate_first = self.first_output(first_hidden)

        filled = observed * values + (1 - observed) * estimate_first
        second_hidden = self.embedding_dropout(
            self.second_embedding(torch.cat((filled, encoded), dim=-1))
            + position
        )
        second_hidden, attention = self._attention_block(second_hidden, self.second_block)
        estimate_second = (self.second_output(second_hidden, values, observed)
                           if self.config.decoder == "conditional" else self.second_output(second_hidden))

        mean_attention = attention.mean(dim=1)
        gate_input = [observed, mean_attention]
        if self.config.gap_aware_gate:
            gate_input.append(gap_features(observed))
        eta = torch.sigmoid(self.gate(torch.cat(gate_input, -1)))
        estimate = (1 - eta) * estimate_first + eta * estimate_second
        imputation = observed * values + (1 - observed) * estimate
        return imputation, estimate_first, estimate_second, estimate, attention


class _ConvSAITSBackend:
    def __init__(self, config: ConvSAITSConfig) -> None:
        self.config = config
        wants_gpu = config.device.lower() in {"gpu", "cuda"}
        self.device = torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self.network = ConvSAITSNetwork(config).to(self.device)
        self.training_history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None
        # Opt-in study hooks; old callers and checkpoint architecture unchanged.
        self.training_checkpoint: Path | None = None
        self.epoch_callback = None

    def _training_masks(self, train_set, truth, epoch):
        finite_truth = np.isfinite(truth)
        input_observed = np.isfinite(np.asarray(train_set["X"])) & finite_truth
        predefined = train_set.get("indicating_mask")
        if predefined is not None:
            hidden = np.asarray(predefined, dtype=bool) & finite_truth
        elif self.config.masking_strategy == "mixed":
            # Sample Single, Block-20, Block-100 or Entire-Log per segment.
            hidden = create_missing_mask(truth, random_state=self.config.seed + epoch)
            hidden &= input_observed
        else:
            generator = np.random.default_rng(self.config.seed + epoch)
            hidden = (generator.random(truth.shape) < self.config.masking_rate) & input_observed
            if not np.any(hidden) and np.any(input_observed):
                candidates = np.flatnonzero(input_observed)
                hidden.flat[generator.choice(candidates)] = True
        observed = input_observed & ~hidden
        return observed, hidden

    def fit(self, train_set, val_set=None):
        truth = np.asarray(train_set.get("X_intact", train_set["X"]), dtype=np.float32)
        if not np.any(np.isfinite(truth) & np.isfinite(train_set["X"])):
            raise ValueError("Conv-SAITS training data has no observed values.")

        optimizer = self._make_optimizer()
        stopping = EarlyStopping(self.config.patience, self.config.min_delta)
        self.training_history = []
        self.best_epoch = None
        start_epoch, stopped = resume_training(self, optimizer, stopping)

        for epoch in range(start_epoch, self.config.epochs) if not stopped else ():
            epoch_started = synchronized_time(self.device)
            observed, hidden = self._training_masks(train_set, truth, epoch)
            if not np.any(hidden):
                raise ValueError("Conv-SAITS requires at least one known value for MIT training.")
            depth = train_set.get("depth")
            if self.config.depth_encoding:
                if depth is None:
                    raise ValueError("Training data needs depth when depth_encoding=True.")
                depth = np.asarray(depth, dtype=np.float32)
                if depth.shape != truth.shape[:2] or not np.isfinite(depth).all():
                    raise ValueError("Training depth must be finite and aligned with X.")
            else:
                depth = np.zeros(truth.shape[:2], dtype=np.float32)
            loader = training_loader(
                truth, observed, hidden, self.config.batch_size, depth=depth,
                seed=self.config.seed + epoch if self.config.independent_shuffle else None,
            )

            self.network.train()
            totals = dict.fromkeys(
                ("loss", "ort_loss", "mit_loss", "mit_mse_loss", "mit_relative_loss"), 0.0
            )
            for batch_inputs, batch_truth, batch_observed, batch_hidden, batch_depth in loader:
                batch_inputs, batch_truth, batch_observed, batch_hidden, batch_depth = (
                    tensor.to(self.device) for tensor in
                    (batch_inputs, batch_truth, batch_observed, batch_hidden, batch_depth)
                )
                _, first, second, combined, _ = self.network(batch_inputs, batch_observed, batch_depth)
                ort = sum(
                    _masked_mae(part, batch_truth, batch_observed)
                    for part in (first, second, combined)
                ) / 3
                mit = masked_imputation_mae(
                    combined, batch_truth, batch_hidden, self.config.mit_reduction
                )
                loss = self.config.ort_weight * ort + self.config.mit_weight * mit
                mit_mse = relative = combined.new_zeros(())
                if self.config.mit_mse_weight:
                    mit_mse = masked_imputation_mse(
                        combined, batch_truth, batch_hidden, self.config.mit_reduction
                    )
                    loss = loss + self.config.mit_mse_weight * mit_mse
                if self.config.mit_relative_weight:
                    scales = batch_truth.new_tensor(self.config.physical_stds)
                    means = batch_truth.new_tensor(self.config.physical_means)
                    floors = batch_truth.new_tensor(self.config.relative_floors)
                    physical_truth = batch_truth * scales + means
                    denominator = physical_truth.abs().clamp_min(floors)
                    relative_mask = batch_hidden
                    relative_prediction = (combined - batch_truth) * scales / denominator
                    relative = masked_imputation_mae(relative_prediction, torch.zeros_like(combined),
                                                      relative_mask, self.config.mit_reduction)
                    loss = loss + self.config.mit_relative_weight * relative
                if not torch.isfinite(loss):
                    raise RuntimeError("Conv-SAITS training loss is non-finite.")
                optimizer.zero_grad()
                loss.backward()
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.gradient_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                for key, value in zip(totals, (loss, ort, mit, mit_mse, relative)):
                    totals[key] += float(value.item())

            train_finished = synchronized_time(self.device)
            losses = {key: value / len(loader) for key, value in totals.items()}
            train_loss = losses["loss"]
            validation_rmse, validation_mape = float("nan"), float("nan")
            if val_set is None:
                score = train_loss
            elif self.config.validation_objective == "rmse_mape":
                validation_rmse, validation_mape = self._validation_objectives(val_set)
                score = max(validation_rmse / self.config.reference_rmse,
                            validation_mape / self.config.reference_mape)
            else:
                score = validation_rmse = self._validation_rmse(val_set)
            if not math.isfinite(score):
                raise RuntimeError(
                    "Conv-SAITS produced a non-finite selection score at "
                    f"epoch {epoch + 1}."
                )
            validation_finished = synchronized_time(self.device)
            self.training_history.append(
                {
                    "train_seconds": train_finished - epoch_started,
                    "validation_seconds": validation_finished - train_finished,
                    "epoch_seconds": validation_finished - epoch_started,
                    "epoch": epoch + 1,
                    **losses,
                    "validation_rmse": validation_rmse,
                    **({"validation_mape": validation_mape}
                       if self.config.validation_objective == "rmse_mape" else {}),
                    "validation_score": score,
                }
            )
            LOGGER.info(
                "Conv-SAITS epoch %d/%d | loss=%.6f | selection_score=%.6f",
                epoch + 1, self.config.epochs, train_loss, score,
            )
            stopped = stopping.update(score, self.network, epoch + 1)
            self.best_epoch = stopping.best_epoch
            save_training(self, optimizer, stopping, epoch + 1, stopped)
            if stopped:
                LOGGER.info("Conv-SAITS early stopping at epoch %d; best epoch=%d", epoch + 1, self.best_epoch)
                break
        stopping.restore(self.network)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        configured = self.config.optimizer
        if configured is None:
            return torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        if isinstance(configured, torch.optim.Optimizer):
            return configured
        if callable(configured):
            return configured(self.network.parameters(), lr=self.config.learning_rate)
        raise TypeError("optimizer must be an optimizer instance, callable, or None.")

    def _validation_rmse(self, dataset):
        return validation_metrics(dataset, self.predict, self.config.batch_size, "Conv-SAITS")["rmse"]

    def _validation_objectives(self, dataset):
        return validation_objectives(self.config, self.predict, dataset)

    def _predict_batch(self, values: np.ndarray, depth: np.ndarray | None = None) -> np.ndarray:
        batch = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        observed = torch.isfinite(batch).float()
        clean = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        batch_depth = None
        if depth is not None:
            batch_depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        imputation, _, _, _, _ = self.network(clean, observed, batch_depth)
        return imputation.cpu().numpy()

    def predict(self, dataset: dict[str, Any]) -> dict[str, np.ndarray]:
        values = np.asarray(dataset["X"], dtype=np.float32)
        depth = dataset.get("depth")
        if self.config.depth_encoding:
            if depth is None:
                raise ValueError("Prediction data needs depth when depth_encoding=True.")
            depth = np.asarray(depth, dtype=np.float32)
            if depth.shape != values.shape[:2] or not np.isfinite(depth).all():
                raise ValueError("Prediction depth must be finite and aligned with X.")
        return predict_batches(self.network, self._predict_batch, dataset, self.config.batch_size)


class ConvSAITS(AbstractModel):
    """Dual-stage convolution-and-attention imputer for segmented well logs."""

    name = "conv_saits"

    def __init__(self, config: ConvSAITSConfig | None = None) -> None:
        super().__init__(config or ConvSAITSConfig())

    def _build_backend(self) -> _ConvSAITSBackend:
        return _ConvSAITSBackend(self.config)


# Readable compatibility alias for code which mirrors the paper's hyphenation.
CONV_SAITS = ConvSAITS


__all__ = [
    "DecoupledFeatureEncoder",
    "DiagonallyMaskedAttentionLayer",
    "ConvSAITS",
    "ConvSAITSConfig",
    "ConvSAITSNetwork",
    "CONV_SAITS",
]
