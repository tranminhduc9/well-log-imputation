"""XGBoost imputer implementation and model adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBRegressor

from .base import AbstractModel


class XGBoostImputer:
    """Train one XGBoost regressor for each well-log feature."""

    def __init__(self, num_models: int, **kwargs: Any) -> None:
        self.num_models = num_models
        self.models = {
            feature: XGBRegressor(**kwargs) for feature in range(num_models)
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


class XGBoostModel(AbstractModel):
    name = "xgboost"

    def _build_backend(self):
        return XGBoostImputer(
            self.config.n_features,
            n_estimators=26,
            min_child_weight=6.0,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=1,
            reg_alpha=0.0,
            learning_rate=0.1,
        )
