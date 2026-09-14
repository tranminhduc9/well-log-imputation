"""XGBoost baseline for well-log imputation."""

from dataclasses import dataclass
import logging
import time

import numpy as np
from xgboost import XGBRegressor

from src.models.model import AbstractModel, ModelConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class XGBoostConfig(ModelConfig):
    """Hyperparameters shared by the per-log regressors."""

    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_jobs: int = -1
    device: str = "cpu"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_estimators <= 0 or self.max_depth <= 0:
            raise ValueError("n_estimators and max_depth must be positive.")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("subsample and colsample_bytree must be in (0, 1].")


class _XGBoostBackend:
    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self.models: list[XGBRegressor] = []
        self.training_history = []

    def _features(self, values: np.ndarray, target: int) -> np.ndarray:
        """Use the other logs and normalized position within each segment."""

        rows = values.reshape(-1, values.shape[-1])
        other_logs = np.delete(rows, target, axis=1)
        depth = np.tile(
            np.linspace(0.0, 1.0, values.shape[1]),
            values.shape[0],
        )[:, None]
        return np.column_stack((other_logs, depth))

    def fit(self, train_set, val_set=None) -> None:
        values = np.asarray(train_set["X"], dtype=float)
        targets = values.reshape(-1, values.shape[-1])
        self.models = []
        self.training_history = []

        for target in range(values.shape[-1]):
            started = time.perf_counter()
            LOGGER.info(
                "XGBoost log %d/%d started",
                target + 1,
                values.shape[-1],
            )
            observed = np.isfinite(targets[:, target])
            if not np.any(observed):
                raise ValueError(f"Training log {target} contains no observed values.")

            model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                device=self.config.device,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                n_jobs=self.config.n_jobs,
                random_state=self.config.seed,
            )
            train_features = self._features(values, target)[observed]
            train_targets = targets[observed, target]
            evaluation_sets = [(train_features, train_targets)]

            if val_set is not None and {
                "X",
                "X_intact",
                "indicating_mask",
            }.issubset(val_set):
                validation_values = np.asarray(val_set["X"], dtype=float)
                validation_targets = np.asarray(
                    val_set["X_intact"], dtype=float
                ).reshape(-1, values.shape[-1])
                validation_mask = np.asarray(
                    val_set["indicating_mask"], dtype=bool
                )[..., target].reshape(-1)
                if np.any(validation_mask):
                    evaluation_sets.append(
                        (
                            self._features(validation_values, target)[validation_mask],
                            validation_targets[validation_mask, target],
                        )
                    )

            model.fit(
                train_features,
                train_targets,
                eval_set=evaluation_sets,
                verbose=False,
            )
            self.models.append(model)
            history = model.evals_result()
            self.training_history.append(
                {
                    "log_index": target,
                    "train_rmse": history["validation_0"]["rmse"],
                    "validation_rmse": history.get("validation_1", {}).get(
                        "rmse", []
                    ),
                }
            )
            LOGGER.info(
                "XGBoost log %d/%d finished in %.1f seconds",
                target + 1,
                values.shape[-1],
                time.perf_counter() - started,
            )

    def predict(self, dataset):
        values = np.asarray(dataset["X"], dtype=float)
        imputation = values.copy()

        for target, model in enumerate(self.models):
            missing = np.isnan(values[..., target]).reshape(-1)
            if np.any(missing):
                column = imputation[..., target].reshape(-1)
                column[missing] = model.predict(
                    self._features(values, target)[missing]
                )

        return {"imputation": imputation}


class XGBoost(AbstractModel):
    """Train one XGBoost regressor for each well-log feature."""

    name = "xgboost"

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        super().__init__(config or XGBoostConfig())

    def _build_backend(self):
        return _XGBoostBackend(self.config)
