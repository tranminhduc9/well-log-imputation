"""Core BRITS model for multivariate well-log imputation."""

from dataclasses import dataclass
from copy import deepcopy
import logging
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import AbstractModel, ModelConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BRITSConfig(ModelConfig):
    hidden_size: int = 64
    consistency_weight: float = 0.1
    min_delta: float = 1e-4
    # Keep PyPOTS-compatible optimization by default. Experiments can opt in
    # to regularization (the final GeoLink notebook uses 1e-5).
    weight_decay: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative.")
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

    def forward(self, values, masks, deltas):
        batch_size, sequence_length, _ = values.shape
        hidden = values.new_zeros(batch_size, self.hidden_size)
        cell = values.new_zeros(batch_size, self.hidden_size)
        loss = values.new_tensor(0.0)
        imputations = []

        for step in range(sequence_length):
            x = values[:, step]
            mask = masks[:, step]
            delta = deltas[:, step]

            hidden = hidden * self.decay_h(delta)
            history = self.history(hidden)
            loss = loss + _masked_mae(history, x, mask)

            completed = mask * x + (1 - mask) * history
            feature = self.feature(completed)
            loss = loss + _masked_mae(feature, x, mask)

            weight = torch.sigmoid(self.combine(torch.cat((self.decay_x(delta), mask), 1)))
            estimate = weight * feature + (1 - weight) * history
            loss = loss + _masked_mae(estimate, x, mask)

            completed = mask * x + (1 - mask) * estimate
            hidden, cell = self.rnn(torch.cat((completed, mask), 1), (hidden, cell))
            imputations.append(completed)

        return torch.stack(imputations, 1), loss / (sequence_length * 3)


class BRITSNetwork(nn.Module):
    def __init__(self, n_features, hidden_size, consistency_weight):
        super().__init__()
        self.forward_rits = RITS(n_features, hidden_size)
        self.backward_rits = RITS(n_features, hidden_size)
        self.consistency_weight = consistency_weight

    def forward(self, values, masks):
        forward, forward_loss = self.forward_rits(values, masks, _deltas(masks))

        reverse_values = torch.flip(values, (1,))
        reverse_masks = torch.flip(masks, (1,))
        backward, backward_loss = self.backward_rits(
            reverse_values,
            reverse_masks,
            _deltas(reverse_masks),
        )
        backward = torch.flip(backward, (1,))

        consistency = torch.mean(torch.abs(forward - backward))
        loss = forward_loss + backward_loss + self.consistency_weight * consistency
        return (forward + backward) / 2, loss


def _masked_mae(prediction, target, mask):
    return torch.sum(torch.abs(prediction - target) * mask) / (torch.sum(mask) + 1e-5)


def _deltas(masks):
    """Unit depth steps since the latest observation for every log."""

    deltas = torch.zeros_like(masks)
    for step in range(1, masks.shape[1]):
        deltas[:, step] = (
            1 + (1 - masks[:, step - 1]) * deltas[:, step - 1]
        )
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
        self.training_history = []
        self.best_epoch = None

    def fit(self, train_set, val_set=None):
        values = np.asarray(train_set["X"], dtype=np.float32)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(values)),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.training_history = []
        self.best_epoch = None
        best_score = float("inf")
        patience_score = float("inf")
        best_state = None
        epochs_without_improvement = 0
        training_started = time.perf_counter()
        for epoch in range(self.config.epochs):
            self.network.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                masks = torch.isfinite(batch).float()
                batch = torch.nan_to_num(batch)
                _, loss = self.network(batch, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            mean_loss = epoch_loss / len(loader)
            validation = self._validation_metrics(val_set) if val_set is not None else None
            validation_mse = None if validation is None else validation["mse"]
            validation_rmse = None if validation is None else validation["rmse"]
            # PyPOTS uses MSE as BRITS' default validation metric. For a
            # multi-scenario monitor we use the unweighted mean scenario MSE.
            score = validation_mse if validation_mse is not None else mean_loss
            if not np.isfinite(score):
                raise RuntimeError(f"BRITS produced a non-finite selection score at epoch {epoch + 1}.")
            history_point = {
                "epoch": epoch + 1,
                "loss": mean_loss,
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

            if score < best_score:
                best_score = score
                best_state = deepcopy(self.network.state_dict())
                self.best_epoch = epoch + 1
            if patience_score - score > self.config.min_delta:
                patience_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    LOGGER.info(
                        "BRITS early stopping at epoch %d | best epoch=%d | score=%.6f",
                        epoch + 1,
                        self.best_epoch,
                        best_score,
                    )
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _validation_metrics(self, dataset):
        scenarios = dataset.get("validation_scenarios")
        if scenarios is not None:
            if not scenarios:
                raise ValueError("BRITS validation_scenarios must not be empty.")
            by_scenario = {
                name: self._validation_metrics(data)
                for name, data in scenarios.items()
            }
            return {
                "mse": float(np.mean([metrics["mse"] for metrics in by_scenario.values()])),
                "rmse": float(np.mean([metrics["rmse"] for metrics in by_scenario.values()])),
                "by_scenario": by_scenario,
            }
        if "X_intact" not in dataset or "indicating_mask" not in dataset:
            raise ValueError("BRITS validation requires X_intact and indicating_mask.")
        inputs = np.asarray(dataset["X"], dtype=np.float32)
        truth = np.asarray(dataset["X_intact"], dtype=np.float32)
        indicating_mask = np.asarray(dataset["indicating_mask"], dtype=bool)
        squared_error = 0.0
        count = 0
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), self.config.batch_size):
                end = start + self.config.batch_size
                batch = torch.from_numpy(inputs[start:end]).to(self.device)
                observed = torch.isfinite(batch).float()
                predictions, _ = self.network(torch.nan_to_num(batch), observed)
                valid = indicating_mask[start:end] & np.isfinite(truth[start:end])
                difference = predictions.cpu().numpy()[valid] - truth[start:end][valid]
                squared_error += float(np.square(difference.astype(np.float64)).sum())
                count += int(valid.sum())
        if count == 0:
            raise ValueError("BRITS validation has no masked values to score.")
        mse = float(squared_error / count)
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "by_scenario": {}}

    def _validation_rmse(self, dataset):
        """Retain the previous helper API for callers that only need RMSE."""

        return self._validation_metrics(dataset)["rmse"]

    def predict(self, dataset):
        values = np.asarray(dataset["X"], dtype=np.float32)
        result = []
        self.network.eval()

        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                batch = torch.from_numpy(values[start : start + self.config.batch_size]).to(self.device)
                masks = torch.isfinite(batch).float()
                imputation, _ = self.network(torch.nan_to_num(batch), masks)
                result.append(imputation.cpu().numpy())

        return {"imputation": np.concatenate(result)}


class BRITS(AbstractModel):
    """Bidirectional recurrent imputation for segmented well logs."""

    name = "brits"

    def __init__(self, config: BRITSConfig | None = None):
        super().__init__(config or BRITSConfig())

    def _build_backend(self):
        return _BRITSBackend(self.config)
