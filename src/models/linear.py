"""Bidirectional linear interpolation; train-normalized mean for empty logs."""
from src.models.model import AbstractModel
from src.models.geo_tcn import interpolation_features


class _LinearBackend:
    def predict(self, dataset):
        return {'imputation': interpolation_features(dataset['X'])[1]}


class Linear(AbstractModel):
    name = 'linear'
    requires_training = False

    def _build_backend(self):
        return _LinearBackend()
