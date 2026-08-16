"""Random Forest imputer implementation and model adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import AbstractModel


class RandomForestImputer:
    """Train one Random Forest regressor for each well-log feature."""

    def __init__(self, num_models: int, **kwargs: Any) -> None:
        self.num_models = num_models
        self.models = {
            feature: RandomForestRegressor(**kwargs)
            for feature in range(num_models)
        }

    def fit(self, train_set, val_set=None) -> None:
        values = np.copy(train_set["X"])
        for feature, model in self.models.items():
            target = values[:, :, feature].reshape(-1)
            predictors = np.delete(values, feature, axis=2).reshape(
                -1, self.num_models - 1
            )
            model.fit(predictors, target)

    def predict(self, test_set):
        values = np.nan_to_num(np.copy(test_set["X"]))
        missing = np.asarray(test_set["indicating_mask"], dtype=bool)
        imputed_features = []

        for feature, model in self.models.items():
            target = values[:, :, feature : feature + 1]
            imputed = np.zeros_like(target)
            imputed[~missing[:, :, feature]] = target[~missing[:, :, feature]]
            predictors = np.delete(values, feature, axis=2)[missing[:, :, feature]]
            if len(predictors):
                imputed[missing[:, :, feature]] = model.predict(predictors).reshape(-1, 1)
            imputed_features.append(imputed)

        return {"imputation": np.concatenate(imputed_features, axis=2)}


class RandomForestModel(AbstractModel):
    name = "rf"

    def _build_backend(self):
        return RandomForestImputer(
            self.config.n_features,
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
        )
