"""Explicit validation-calibrated ensemble, never presented as a single network."""
import itertools
import numpy as np
from src.models.model import AbstractModel, ModelConfig
from src.models.geoboost import gap_groups


class _GeoBlendBackend:
    def __init__(self, config):
        self.config = config
        self.members = []
        self.weights = None
        self.calibration = []

    def fit(self, train_set, val_set=None):
        if not self.members or val_set is None:
            raise ValueError('GeoBlend needs fitted members and validation data.')
        self.weights = np.full((self.config.n_features, 4, len(self.members)), 1 / len(self.members))
        grid = np.array([w for w in itertools.product(range(5), repeat=len(self.members)) if sum(w) == 4]) / 4
        predictions, targets, groups = [[] for _ in range(3)]
        for data in val_set['validation_scenarios'].values():
            payload = {'X': data['X']}
            if 'depth' in data:
                payload['depth'] = data['depth']
            estimates = np.stack([member.impute(payload) for member in self.members], axis=-1)
            kind = self._groups(data['X'])
            predictions.append(estimates)
            targets.append(data['X_intact'])
            groups.append(kind)
        predictions, targets, groups = map(np.concatenate, (predictions, targets, groups))
        for target in range(self.config.n_features):
            for group in range(4):
                selected = groups[..., target] == group
                if not selected.any():
                    continue
                candidates = predictions[..., target, :][selected]
                truth = targets[..., target][selected]
                loss = ((candidates @ grid.T - truth[:, None])**2).mean(axis=0)
                weight = grid[loss.argmin()]
                self.weights[target, group] = weight
                self.calibration.append(dict(log_index=target, gap_group=group, count=int(selected.sum()),
                                             weights=weight.tolist(), validation_rmse=float(np.sqrt(loss.min()))))

    @staticmethod
    def _groups(values):
        groups = np.full(values.shape, -1, dtype=int)
        for segment in range(len(values)):
            for target in range(values.shape[2]):
                groups[segment, :, target] = gap_groups(np.isfinite(values[segment, :, target]))
        return groups

    def predict(self, dataset):
        payload = {'X': dataset['X']}
        if 'depth' in dataset:
            payload['depth'] = dataset['depth']
        estimates = np.stack([member.impute(payload) for member in self.members], axis=-1)
        result = np.asarray(dataset['X']).copy()
        groups = self._groups(result)
        for target in range(result.shape[2]):
            selected = groups[..., target] >= 0
            result[..., target][selected] = np.sum(estimates[..., target, :][selected] *
                                                  self.weights[target, groups[..., target][selected]], axis=1)
        return {'imputation': result}


class GeoBlend(AbstractModel):
    name = 'geo_blend'

    def __init__(self, config=None):
        super().__init__(config or ModelConfig())

    def _build_backend(self):
        return _GeoBlendBackend(self.config)
