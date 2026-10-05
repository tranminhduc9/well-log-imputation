"""Core BRITS model for multivariate well-log imputation."""

from dataclasses import dataclass
import logging
import math
import time

import numpy as np
import torch
from torch import nn

from src.models.model import AbstractModel, ModelConfig
from src.models.losses import masked_imputation_mae
from src.models._training import EarlyStopping, training_loader, validation_metrics, predict_batches, synchronized_time
from src.preprocessing.pipeline import BLOCK_LENGTHS, MISSING_SCENARIOS


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BRITSConfig(ModelConfig):
    hidden_size: int = 64
    consistency_weight: float = 0.1
    mit_weight: float = 1.0
    mit_reduction: str = "segment"
    masking_strategy: str = "mixed"
    masking_rate: float = 0.2
    gradient_clip: float = 1.0
    validation_metric: str = "rmse"
    min_delta: float = 1e-4
    # Keep PyPOTS-compatible optimization by default. Experiments can opt in
    # to regularization (the final GeoLink notebook uses 1e-5).
    weight_decay: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.mit_reduction not in {"segment", "point"}:
            raise ValueError("mit_reduction must be 'segment' or 'point'.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative.")
        if self.mit_weight < 0:
            raise ValueError("mit_weight must be non-negative.")
        if self.masking_strategy not in {"mixed", "random", "none"}:
            raise ValueError(
                "masking_strategy must be 'mixed', 'random', or 'none'."
            )
        if not 0 < self.masking_rate < 1:
            raise ValueError("masking_rate must be in (0, 1).")
        if self.gradient_clip < 0:
            raise ValueError("gradient_clip must be non-negative.")
        if self.validation_metric not in {"mse", "rmse"}:
            raise ValueError("validation_metric must be 'mse' or 'rmse'.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diagonal=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.register_buffer("mask", torch.eye(input_size) if diagonal else None)
        self._reset_parameters()

    def _reset_parameters(self):
        """Match the parameter initialization used by PyPOTS."""

        bound = 1.0 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, delta):
        weight = self.weight if self.mask is None else self.weight * self.mask
        return torch.exp(-torch.relu(nn.functional.linear(delta, weight, self.bias)))


class FeatureRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, n_features))
        self.bias = nn.Parameter(torch.empty(n_features))
        self.register_buffer("mask", 1 - torch.eye(n_features))
        self._reset_parameters()

    def _reset_parameters(self):
        """Match the parameter initialization used by PyPOTS."""

        bound = 1.0 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values):
        return nn.functional.linear(values, self.weight * self.mask, self.bias)


class RITS(nn.Module):
    """One directional recurrent imputation model."""

    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(n_features * 2, hidden_size)
        self.history = nn.Linear(hidden_size, n_features)
        self.feature = FeatureRegression(n_features)
        self.decay_h = TemporalDecay(n_features, hidden_size)
        self.decay_x = TemporalDecay(n_features, n_features, diagonal=True)
        self.combine = nn.Linear(n_features * 2, n_features)

    def forward(self, values, masks, deltas, compute_loss=True):
        batch_size, sequence_length, _ = values.shape
        hidden = values.new_zeros(batch_size, self.hidden_size)
        cell = values.new_zeros(batch_size, self.hidden_size)
        imputations = []
        errors = []
        # Recurrent steps consume [batch, features]. Make these slices
        # contiguous once instead of striding through batch-major tensors.
        values = values.transpose(0, 1).contiguous()
        masks = masks.transpose(0, 1).contiguous()
        deltas = deltas.transpose(0, 1).contiguous()
        # These projections depend only on the masks/deltas, not recurrent
        # state. Compute them for the entire sequence in three large calls.
        decay_h = self.decay_h(deltas)
        missing = 1 - masks
        weights = torch.sigmoid(self.combine(torch.cat((self.decay_x(deltas), masks), -1)))
        feature_weight = self.feature.weight * self.feature.mask

        for step in range(sequence_length):
            x = values[step]
            mask = masks[step]
            hidden = hidden * decay_h[step]
            history = self.history(hidden)

            completed = mask * x + missing[step] * history
            feature = nn.functional.linear(completed, feature_weight, self.feature.bias)

            weight = weights[step]
            estimate = weight * feature + (1 - weight) * history
            if compute_loss:
                errors.append((torch.abs(history - x) + torch.abs(feature - x)
                               + torch.abs(estimate - x)) * mask)

            completed = mask * x + missing[step] * estimate
            hidden, cell = self.rnn(torch.cat((completed, mask), 1), (hidden, cell))
            imputations.append(completed)

        loss = values.new_tensor(0.0)
        if compute_loss:
            # Preserve the original per-timestep normalization (not a pooled
            # loss across the sequence), including empty-mask timesteps.
            numerators = torch.stack(errors).sum(dim=(1, 2))
            denominators = masks.sum(dim=(1, 2)) + 1e-5
            loss = (numerators / denominators).sum() / (sequence_length * 3)
        return torch.stack(imputations, 1), loss


class BRITSNetwork(nn.Module):
    def __init__(self, n_features, hidden_size, consistency_weight):
        super().__init__()
        self.forward_rits = RITS(n_features, hidden_size)
        self.backward_rits = RITS(n_features, hidden_size)
        self.consistency_weight = consistency_weight

    def forward(self, values, masks, return_components=False, compute_loss=True):
        forward, forward_loss = self.forward_rits(values, masks, _deltas(masks), compute_loss)

        reverse_values = torch.flip(values, (1,))
        reverse_masks = torch.flip(masks, (1,))
        backward, backward_loss = self.backward_rits(
            reverse_values,
            reverse_masks,
            _deltas(reverse_masks),
            compute_loss,
        )
        backward = torch.flip(backward, (1,))

        consistency = (torch.mean(torch.abs(forward - backward)) if compute_loss
                       else values.new_tensor(0.0))
        reconstruction = forward_loss + backward_loss
        loss = reconstruction + self.consistency_weight * consistency
        imputation = (forward + backward) / 2
        if return_components:
            return imputation, loss, reconstruction, consistency
        return imputation, loss


def _masked_mae(prediction, target, mask):
    return torch.sum(torch.abs(prediction - target) * mask) / (torch.sum(mask) + 1e-5)


def _deltas(masks):
    """Unit depth steps since the latest observation for every log."""

    deltas = torch.zeros_like(masks)
    # The recurrence is distance from the most recent observation BEFORE the
    # current step. cummax finds those positions in parallel for binary masks.
    positions = torch.arange(masks.shape[1], device=masks.device).view(1, -1, 1)
    observed_positions = torch.where(masks[:, :-1].bool(), positions[:, :-1], 0)
    latest = observed_positions.cummax(dim=1).values
    deltas[:, 1:] = (positions[:, 1:] - latest).to(masks.dtype)
    return deltas


class _BRITSBackend:
    def __init__(self, config):
        self.config = config
        wants_gpu = config.device.lower() in {"gpu", "cuda"}
        self.device = torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self.network = BRITSNetwork(
            config.n_features,
            config.hidden_size,
            config.consistency_weight,
        ).to(self.device)
        LOGGER.info("BRITS actual device: %s", self.device)
        if wants_gpu and self.device.type != "cuda":
            LOGGER.warning("BRITS requested GPU but CUDA is unavailable; using CPU.")
        self.training_history = []
        self.best_epoch = None

    def _mixed_missing_mask(self, values, random_state):
        """Sample one evaluation-style missingness scenario per segment."""

        generator = np.random.default_rng(random_state)
        segment_count, sequence_length, feature_count = values.shape
        available_scenarios = [
            scenario
            for scenario in MISSING_SCENARIOS
            if BLOCK_LENGTHS.get(scenario, 1) <= sequence_length
        ]
        scenarios = generator.choice(available_scenarios, size=segment_count)
        hidden = np.zeros(values.shape, dtype=bool)
        for segment, scenario in enumerate(scenarios):
            feature = generator.integers(feature_count)
            if scenario == "Single":
                step = generator.integers(sequence_length)
                hidden[segment, step, feature] = True
            elif scenario in BLOCK_LENGTHS:
                length = BLOCK_LENGTHS[scenario]
                start = generator.integers(sequence_length - length + 1)
                hidden[segment, start : start + length, feature] = True
            else:
                hidden[segment, :, feature] = True
        return hidden

    def _training_masks(self, train_set, truth, epoch):
        finite_truth = np.isfinite(truth)
        input_observed = np.isfinite(np.asarray(train_set["X"])) & finite_truth
        predefined = train_set.get("indicating_mask")
        if predefined is not None:
            hidden = np.asarray(predefined, dtype=bool) & finite_truth
        elif self.config.masking_strategy == "mixed":
            hidden = self._mixed_missing_mask(
                truth, random_state=self.config.seed + epoch
            )
            hidden &= input_observed
        elif self.config.masking_strategy == "random":
            generator = np.random.default_rng(self.config.seed + epoch)
            hidden = (
                generator.random(truth.shape) < self.config.masking_rate
            ) & input_observed
        else:
            hidden = np.zeros(truth.shape, dtype=bool)

        if (
            self.config.masking_strategy != "none"
            and predefined is None
            and not np.any(hidden)
            and np.any(input_observed)
        ):
            generator = np.random.default_rng(self.config.seed + epoch)
            hidden.flat[generator.choice(np.flatnonzero(input_observed))] = True
        observed = input_observed & ~hidden
        return observed, hidden

    def fit(self, train_set, val_set=None):
        truth = np.asarray(
            train_set.get("X_intact", train_set["X"]), dtype=np.float32
        )
        if not np.any(np.isfinite(truth) & np.isfinite(train_set["X"])):
            raise ValueError("BRITS training data has no observed values.")
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.training_history = []
        self.best_epoch = None
        stopping = EarlyStopping(self.config.patience, self.config.min_delta)
        training_started = time.perf_counter()
        for epoch in range(self.config.epochs):
            epoch_started = synchronized_time(self.device)
            observed, hidden = self._training_masks(train_set, truth, epoch)
            loader = training_loader(truth, observed, hidden, self.config.batch_size,
                                     pin_memory=self.device.type == "cuda")
            self.network.train()
            totals = torch.zeros(4, device=self.device)
            for batch, batch_truth, masks, batch_hidden in loader:
                batch, batch_truth, masks, batch_hidden = (
                    tensor.to(self.device, non_blocking=self.device.type == "cuda")
                    for tensor in (batch, batch_truth, masks, batch_hidden)
                )
                imputation, brits_loss, reconstruction, consistency = self.network(
                    batch, masks, return_components=True
                )
                mit = masked_imputation_mae(
                    imputation, batch_truth, batch_hidden, self.config.mit_reduction
                )
                loss = brits_loss + self.config.mit_weight * mit
                if not torch.isfinite(loss):
                    raise RuntimeError("BRITS training loss is non-finite.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.gradient_clip
                    )
                optimizer.step()
                totals += torch.stack((loss, reconstruction, consistency, mit)).detach()

            train_finished = synchronized_time(self.device)
            mean_loss, reconstruction_mean, consistency_mean, mit_mean = (totals / len(loader)).tolist()
            validation = self._validation_metrics(val_set) if val_set is not None else None
            validation_mse = None if validation is None else validation["mse"]
            validation_rmse = None if validation is None else validation["rmse"]
            # GeoLink reports RMSE, so checkpoint selection uses the same
            # metric by default. MSE remains available for PyPOTS parity.
            score = (
                validation[self.config.validation_metric]
                if validation is not None
                else mean_loss
            )
            if not np.isfinite(score):
                raise RuntimeError(f"BRITS produced a non-finite selection score at epoch {epoch + 1}.")
            validation_finished = synchronized_time(self.device)
            history_point = {
                "train_seconds": train_finished - epoch_started,
                "validation_seconds": validation_finished - train_finished,
                "epoch_seconds": validation_finished - epoch_started,
                "epoch": epoch + 1,
                "loss": mean_loss,
                "reconstruction_loss": reconstruction_mean,
                "consistency_loss": consistency_mean,
                "mit_loss": mit_mean,
                "validation_mse": validation_mse,
                "validation_rmse": validation_rmse,
            }
            if validation is not None:
                for scenario, metrics in validation["by_scenario"].items():
                    key = scenario.lower().replace("-", "_").replace(" ", "_")
                    history_point[f"validation_{key}_mse"] = metrics["mse"]
                    history_point[f"validation_{key}_rmse"] = metrics["rmse"]
            self.training_history.append(history_point)
            elapsed = time.perf_counter() - training_started
            LOGGER.info(
                "BRITS epoch %d/%d | loss=%.6f | validation_mse=%s | "
                "validation_rmse=%s | elapsed=%.1fs",
                epoch + 1,
                self.config.epochs,
                mean_loss,
                f"{validation_mse:.6f}" if validation_mse is not None else "n/a",
                f"{validation_rmse:.6f}" if validation_rmse is not None else "n/a",
                elapsed,
            )

            stopped = stopping.update(score, self.network, epoch + 1)
            self.best_epoch = stopping.best_epoch
            if stopped:
                LOGGER.info("BRITS early stopping at epoch %d | best epoch=%d | score=%.6f",
                            epoch + 1, self.best_epoch, stopping.best_score)
                break
        stopping.restore(self.network)

    def _validation_metrics(self, dataset):
        return validation_metrics(dataset, self.predict, self.config.batch_size, "BRITS")

    def _validation_rmse(self, dataset):
        """Retain the previous helper API for callers that only need RMSE."""

        return self._validation_metrics(dataset)["rmse"]

    def _predict_batch(self, values, depth=None):
        batch = torch.from_numpy(values).to(self.device)
        observed = torch.isfinite(batch).float()
        imputation, _ = self.network(torch.nan_to_num(batch), observed, compute_loss=False)
        return imputation.cpu().numpy()

    def predict(self, dataset):
        return predict_batches(self.network, self._predict_batch, dataset, self.config.batch_size)


class BRITS(AbstractModel):
    """Bidirectional recurrent imputation for segmented well logs."""

    name = "brits"

    def __init__(self, config: BRITSConfig | None = None):
        super().__init__(config or BRITSConfig())

    def _build_backend(self):
        return _BRITSBackend(self.config)
