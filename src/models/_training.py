"""Batching, masked validation and early stopping for imputation experiments."""

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def synchronized_time(device):
    """Wall-clock boundary with completed CUDA work for timing measurements."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def training_loader(truth, observed, hidden, batch_size, depth=None, seed=None):
    arrays = [np.where(observed, truth, 0), np.where(np.isfinite(truth), truth, 0), observed, hidden]
    if depth is not None:
        arrays.append(depth)
    tensors = [torch.from_numpy(np.asarray(array, dtype=np.float32)) for array in arrays]
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True, generator=generator)


def validation_metrics(dataset, predict, batch_size, name):
    """Pooled masked MSE/RMSE per scenario, then an equal scenario average."""
    scenarios = dataset.get('validation_scenarios')
    if scenarios is not None:
        if not scenarios:
            raise ValueError(f'{name} validation_scenarios must not be empty.')
        scores = {scenario: validation_metrics(data, predict, batch_size, name)
                  for scenario, data in scenarios.items()}
        return {metric: float(np.mean([score[metric] for score in scores.values()]))
                for metric in ('mse', 'rmse')} | {'by_scenario': scores}
    if 'X_intact' not in dataset or 'indicating_mask' not in dataset:
        raise ValueError(f'{name} validation requires X_intact and indicating_mask.')
    values = np.asarray(dataset['X'], dtype=np.float32)
    truth = np.asarray(dataset['X_intact'], dtype=np.float32)
    mask = np.asarray(dataset['indicating_mask'], dtype=bool)
    squared_error, count = 0., 0
    for start in range(0, len(values), batch_size):
        end = start + batch_size
        inputs = {'X': values[start:end]}
        if 'depth' in dataset:
            inputs['depth'] = np.asarray(dataset['depth'])[start:end]
        prediction = predict(inputs)['imputation']
        valid = mask[start:end] & np.isfinite(truth[start:end])
        residual = (prediction[valid] - truth[start:end][valid]).astype(np.float64)
        squared_error += float(np.square(residual).sum())
        count += int(valid.sum())
    if not count:
        raise ValueError(f'{name} validation has no masked values to score.')
    mse = squared_error / count
    return {'mse': mse, 'rmse': float(np.sqrt(mse)), 'by_scenario': {}}


def predict_batches(network, predict_batch, dataset, batch_size):
    values = np.asarray(dataset['X'], dtype=np.float32)
    depth = dataset.get('depth')
    if depth is not None:
        depth = np.asarray(depth)
    network.eval()
    with torch.no_grad():
        parts = [predict_batch(values[start:start + batch_size],
                               None if depth is None else depth[start:start + batch_size])
                 for start in range(0, len(values), batch_size)]
    return {'imputation': np.concatenate(parts) if parts else values.copy()}


@dataclass
class EarlyStopping:
    """Keep the lowest-score weights; count epochs without a min_delta improvement."""
    patience: int
    min_delta: float
    best_score: float = float('inf')
    patience_score: float = float('inf')
    best_state: dict | None = None
    best_epoch: int | None = None
    stale_epochs: int = 0

    def update(self, score, network, epoch):
        if score < self.best_score:
            self.best_score, self.best_epoch = score, epoch
            self.best_state = deepcopy(network.state_dict())
        if score < self.patience_score - self.min_delta:
            self.patience_score, self.stale_epochs = score, 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience

    def restore(self, network):
        if self.best_state is not None:
            network.load_state_dict(self.best_state)
