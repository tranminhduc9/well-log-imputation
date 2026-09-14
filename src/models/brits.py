"""Core BRITS model for multivariate well-log imputation."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import AbstractModel, ModelConfig


@dataclass(frozen=True)
class BRITSConfig(ModelConfig):
    hidden_size: int = 64
    consistency_weight: float = 0.1


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diagonal=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.register_buffer("mask", torch.eye(input_size) if diagonal else None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, delta):
        weight = self.weight if self.mask is None else self.weight * self.mask
        return torch.exp(-torch.relu(nn.functional.linear(delta, weight, self.bias)))


class FeatureRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("mask", 1 - torch.eye(n_features))
        nn.init.xavier_uniform_(self.weight)

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

        return torch.stack(imputations, 1), loss / sequence_length


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
    """Depth steps since the latest observation for every log."""

    deltas = torch.ones_like(masks)
    for step in range(1, masks.shape[1]):
        deltas[:, step] += (1 - masks[:, step]) * deltas[:, step - 1]
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
        )

        self.network.train()
        for _ in range(self.config.epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                masks = torch.isfinite(batch).float()
                batch = torch.nan_to_num(batch)
                _, loss = self.network(batch, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
