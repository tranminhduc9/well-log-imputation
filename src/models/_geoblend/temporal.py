"""Interpolation-residual dilated convolution model for offline log imputation."""
from copy import deepcopy
from dataclasses import dataclass
import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import AbstractModel, ModelConfig
from src.preprocessing.pipeline import create_missing_mask

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoTCNConfig(ModelConfig):
    channels: int = 48
    dilations: tuple = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    min_delta: float = 0.0001
    max_segments_per_epoch: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.channels < 4 or self.kernel_size < 3 or self.kernel_size % 2 != 1:
            raise ValueError('Invalid GeoTCN channels/kernel size.')
        if not self.dilations or min(self.dilations) < 1 or self.max_segments_per_epoch < 0:
            raise ValueError('Invalid GeoTCN dilations/training size.')


def interpolation_features(values):
    """Bidirectional interpolation from observed inputs, with boundary fallback."""
    values = np.asarray(values, dtype=np.float32)
    observed = np.isfinite(values)
    steps = values.shape[1]
    position = np.arange(steps)[None, :, None]
    left = np.maximum.accumulate(np.where(observed, position, -1), axis=1)
    right = np.minimum.accumulate(np.where(observed, position, steps)[:, ::-1], axis=1)[:, ::-1]
    clean = np.where(observed, values, 0)
    left_value = np.take_along_axis(clean, left.clip(0, steps - 1), axis=1)
    right_value = np.take_along_axis(clean, right.clip(0, steps - 1), axis=1)
    fraction = (position - left) / np.maximum(right - left, 1)
    filled = left_value + fraction * (right_value - left_value)
    filled = np.where(left < 0, right_value, np.where(right >= steps, left_value, filled))
    empty = ~observed.any(axis=1, keepdims=True)
    filled = np.where(empty, 0, filled).astype(np.float32)
    left_gap = np.where(left < 0, steps, position - left)
    right_gap = np.where(right >= steps, steps, right - position)
    nearest = np.minimum(left_gap, right_gap)
    gate = np.minimum(nearest / 8., 1.).astype(np.float32)
    gate = np.where(empty, 1., gate).astype(np.float32)
    inputs = np.concatenate((filled, clean, observed.astype(np.float32),
                             np.log1p(left_gap) / np.log1p(steps),
                             np.log1p(right_gap) / np.log1p(steps)), axis=2).astype(np.float32)
    return inputs, filled, gate


class _ResidualBlock(nn.Module):
    def __init__(self, channels, kernel, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel,
                              padding=dilation * (kernel // 2), dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.project = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        h = self.conv(x)
        h = torch.nn.functional.gelu(self.norm(h.transpose(1, 2)).transpose(1, 2))
        return x + self.project(h)


class GeoTCNNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = nn.Conv1d(5 * config.n_features, config.channels, 1)
        self.blocks = nn.Sequential(*[_ResidualBlock(config.channels, config.kernel_size, d)
                                     for d in config.dilations])
        self.output = nn.Conv1d(config.channels, config.n_features, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs, filled, gate):
        correction = self.output(self.blocks(self.input(inputs.transpose(1, 2)))).transpose(1, 2)
        return filled + gate * correction


class _GeoTCNBackend:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.device in ('cuda', 'gpu') and torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config.seed)
        self.network = GeoTCNNetwork(config).to(self.device)
        self.training_history = []
        self.best_epoch = None

    def fit(self, train_set, val_set=None):
        truth = np.asarray(train_set['X'], dtype=np.float32)
        if not np.isfinite(truth).all():
            raise ValueError('GeoTCN requires finite intact training segments.')
        if val_set is None:
            raise ValueError('GeoTCN requires validation for checkpoint selection.')
        validation = val_set.get('validation_scenarios', {'validation': val_set})
        prepared = {name: (interpolation_features(data['X']), data) for name, data in validation.items()}
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=.5)
        best, patience_best, stale, best_state = float('inf'), float('inf'), 0, None
        generator = torch.Generator().manual_seed(self.config.seed)
        for epoch in range(self.config.epochs):
            started = time.perf_counter()
            rng = np.random.default_rng(self.config.seed + epoch)
            take = rng.permutation(len(truth))
            if self.config.max_segments_per_epoch:
                take = take[:self.config.max_segments_per_epoch]
            target = truth[take]
            hidden = create_missing_mask(target, random_state=self.config.seed + epoch)
            masked = target.copy()
            masked[hidden] = np.nan
            arrays = (*interpolation_features(masked), target, hidden.astype(np.float32))
            loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in arrays)),
                                batch_size=self.config.batch_size, shuffle=True, generator=generator)
            self.network.train()
            total = 0.
            for batch in loader:
                inputs, filled, gate, labels, mask = (x.to(self.device) for x in batch)
                output = self.network(inputs, filled, gate)
                error = output - labels
                # Equal segment weighting prevents long gaps dominating by count.
                loss = (((error.abs() + .5 * error.square()) * mask).sum((1, 2)) /
                        mask.sum((1, 2)).clamp_min(1)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite GeoTCN loss.')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
                optimizer.step()
                total += loss.item()
            scores = {}
            for name, (features, dataset) in prepared.items():
                predictions = self._predict_features(features)
                error = (predictions - dataset['X_intact'])[dataset['indicating_mask'].astype(bool)]
                scores[f'validation_{name}_rmse'] = float(np.sqrt(np.mean(error.astype(float)**2)))
            score = float(np.mean(list(scores.values())))
            if not np.isfinite(score):
                raise RuntimeError('Non-finite GeoTCN validation score.')
            self.training_history.append(dict(epoch=epoch + 1, loss=total / len(loader),
                                               validation_rmse=score, **scores))
            if score < best:
                best, best_state, self.best_epoch = score, deepcopy(self.network.state_dict()), epoch + 1
            if score < patience_best - self.config.min_delta:
                patience_best, stale = score, 0
            else:
                stale += 1
            scheduler.step(score)
            LOGGER.info('GeoTCN epoch %d/%d | validation %.6f | %.1fs', epoch + 1,
                        self.config.epochs, score, time.perf_counter() - started)
            if stale >= self.config.patience:
                break
        self.network.load_state_dict(best_state)

    def _predict_features(self, features):
        outputs = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(features[0]), self.config.batch_size):
                batch = [torch.from_numpy(a[start:start + self.config.batch_size]).to(self.device) for a in features]
                outputs.append(self.network(*batch).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, dataset):
        values = np.asarray(dataset['X'], dtype=np.float32)
        predicted = self._predict_features(interpolation_features(values))
        return {'imputation': np.where(np.isfinite(values), values, predicted)}


class GeoTCN(AbstractModel):
    name = 'geo_tcn'

    def __init__(self, config=None):
        super().__init__(config or GeoTCNConfig())

    def _build_backend(self):
        return _GeoTCNBackend(self.config)
