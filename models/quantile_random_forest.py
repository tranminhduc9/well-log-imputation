"""Quantile Random Forest imputer with predictive intervals."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import AbstractModel


class QuantileRandomForest:
    """Impute with tree-prediction medians and empirical quantile bounds."""

    def __init__(
        self,
        num_models: int,
        device=None,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        prediction_chunk_size: int = 50_000,
        **kwargs,
    ) -> None:
        self.num_models = num_models
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.prediction_chunk_size = prediction_chunk_size
        self.models = {
            i: RandomForestRegressor(**kwargs) for i in range(num_models)
        }

    @staticmethod
    def _feature_matrix(X: np.ndarray, target: int) -> np.ndarray:
        features = np.delete(X, target, axis=2).reshape(-1, X.shape[2] - 1)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, train_set: dict[str, np.ndarray], val_set=None) -> None:
        # Keep targets intact but retain artificial gaps in predictor features.
        X = np.asarray(train_set["X"], dtype=np.float32)
        X_intact = np.asarray(train_set["X_intact"], dtype=np.float32)
        for target, model in self.models.items():
            y = X_intact[:, :, target].reshape(-1)
            x = self._feature_matrix(X, target)
            valid = np.isfinite(y)
            if not np.any(valid):
                raise ValueError(f"Feature {target} has no finite training targets.")
            model.fit(x[valid], y[valid])

    def _tree_quantiles(self, model, x: np.ndarray):
        medians, lowers, uppers = [], [], []
        for start in range(0, len(x), self.prediction_chunk_size):
            chunk = x[start : start + self.prediction_chunk_size]
            tree_predictions = np.stack(
                [tree.predict(chunk) for tree in model.estimators_], axis=0
            )
            lowers.append(np.quantile(tree_predictions, self.lower_quantile, axis=0))
            medians.append(np.quantile(tree_predictions, 0.5, axis=0))
            uppers.append(np.quantile(tree_predictions, self.upper_quantile, axis=0))
        return tuple(np.concatenate(values) for values in (medians, lowers, uppers))

    def predict(self, val_set: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        X = np.asarray(val_set["X"], dtype=np.float32)
        mask = np.asarray(val_set["indicating_mask"], dtype=bool)
        observed = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        imputation, lower, upper = observed.copy(), observed.copy(), observed.copy()

        for target, model in self.models.items():
            missing = mask[:, :, target]
            if not np.any(missing):
                continue
            x = np.delete(X, target, axis=2)[missing]
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            median, q_low, q_high = self._tree_quantiles(model, x)
            imputation[:, :, target][missing] = median
            lower[:, :, target][missing] = q_low
            upper[:, :, target][missing] = q_high

        return {"imputation": imputation, "lower": lower, "upper": upper}


class QuantileRandomForestModel(AbstractModel):
    name = "qrf"

    def _build_backend(self):
        return QuantileRandomForest(
            self.config.n_features,
            # QRF trains one forest per feature. These bounded defaults keep
            # the full depth-sample benchmark practical on CPU.
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            min_samples_split=10,
            max_samples=0.5,
            n_jobs=-1,
            random_state=17076,
        )
