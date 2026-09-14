"""AbstractModel is a common interface for all well-log imputation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.data.metrics import compute_imputation_metrics
from src.data.loader import DataLoader


Dataset = Mapping[str, Any]
DatasetInput = Dataset | np.ndarray | DataLoader
Prediction = dict[str, np.ndarray]


@dataclass(frozen=True)
class ModelConfig:
    """Settings commonly needed by classical and neural models."""

    seq_len: int = 256
    n_features: int = 4
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    seed: int = 912
    device: str = "gpu"
    output_dir: Path | str = Path("artifacts")
    optimizer: Any = None

    def __post_init__(self) -> None:
        positive_settings = {
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "learning_rate": self.learning_rate,
        }
        invalid = [name for name, value in positive_settings.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Model settings must be positive: " + ", ".join(invalid)
            )

        object.__setattr__(self, "output_dir", Path(self.output_dir))


class AbstractModel(ABC):
    """Common interface used by every well-log imputation model.

    A dataset may be passed as a three-dimensional array, a mapping containing
    ``"X"``, or a data loader exposing ``load()``. Evaluation datasets also
    contain ``"X_intact"`` (ground truth) and ``"indicating_mask"`` (the
    artificial gaps to score).
    """

    name: ClassVar[str] = "model"
    requires_training: ClassVar[bool] = True
    requires_nan_training_input: ClassVar[bool] = False

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._backend = self._build_backend()
        self._is_fitted = not self.requires_training

    @abstractmethod
    def _build_backend(self) -> Any:
        """Create the concrete model (for example, SAITS or XGBoost)."""

    @property
    def backend(self) -> Any:
        """Return the wrapped model for model-specific advanced use."""

        return self._backend

    @property
    def is_fitted(self) -> bool:
        """Whether this adapter is ready to make predictions."""

        return self._is_fitted

    def fit(
        self,
        train_set: DatasetInput,
        val_set: DatasetInput | None = None,
    ) -> AbstractModel:
        """Train from arrays, mappings or loaders and return ``self``."""

        train_data = self._prepare_dataset(train_set)
        validation_data = (
            None if val_set is None else self._prepare_dataset(val_set)
        )

        fit_method = getattr(self._backend, "fit", None)
        if not callable(fit_method):
            raise TypeError(f"Backend for '{self.name}' must implement fit().")

        fit_method(train_set=train_data, val_set=validation_data)
        self._is_fitted = True
        return self

    def predict(
        self,
        data: DatasetInput,
        mask: np.ndarray | None = None,
    ) -> Prediction:
        """Impute a dataset and return at least an ``imputation`` array.

        ``mask=True`` marks values that should be hidden and reconstructed.
        Passing an explicit mask is a convenient shortcut when ``data`` is a
        raw array from the preprocessing pipeline.
        """

        if not self._is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted first.")

        dataset = self._prepare_dataset(data, mask)
        predict_method = getattr(self._backend, "predict", None)
        if not callable(predict_method):
            raise TypeError(f"Backend for '{self.name}' must implement predict().")

        raw_prediction = predict_method(dataset)
        prediction = (
            {"imputation": np.asarray(raw_prediction)}
            if not isinstance(raw_prediction, Mapping)
            else {key: np.asarray(value) for key, value in raw_prediction.items()}
        )
        if "imputation" not in prediction:
            raise TypeError(
                f"Model '{self.name}' must return an array or a mapping "
                "containing 'imputation'."
            )

        expected_shape = np.asarray(dataset["X"]).shape
        if prediction["imputation"].shape != expected_shape:
            raise ValueError(
                "Imputation shape must match input shape: "
                f"expected {expected_shape}, got {prediction['imputation'].shape}."
            )
        return prediction

    def impute(
        self,
        data: DatasetInput,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Shortcut that returns only the imputed values."""

        return self.predict(data, mask)["imputation"]

    def evaluate(
        self,
        data: DatasetInput,
        mask: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        """Compute standard errors on artificially missing positions only."""

        dataset = self._prepare_dataset(data, mask)
        if "X_intact" not in dataset or "indicating_mask" not in dataset:
            raise ValueError(
                "Evaluation requires X_intact and indicating_mask, or a raw "
                "array together with mask."
            )

        imputation = self.predict(dataset)["imputation"].astype(float, copy=False)
        return compute_imputation_metrics(
            truth=np.asarray(dataset["X_intact"]),
            imputation=imputation,
            indicating_mask=np.asarray(dataset["indicating_mask"]),
        )

    def _prepare_dataset(
        self,
        data: DatasetInput,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Normalize raw arrays and mappings to the shared dataset format."""

        if isinstance(data, DataLoader):
            dataset = data.load()
        elif isinstance(data, Mapping):
            dataset = dict(data)
        else:
            dataset = {"X": data}
        if "X" not in dataset:
            raise ValueError("Dataset must contain an 'X' array.")

        values = np.asarray(dataset["X"])
        expected_tail = (self.config.seq_len, self.config.n_features)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                "X must have shape (segments, sequence_length, features); "
                f"expected (*, {expected_tail[0]}, {expected_tail[1]}), "
                f"got {values.shape}."
            )
        dataset["X"] = values

        if mask is not None:
            missing = np.asarray(mask, dtype=bool)
            if missing.shape != values.shape:
                raise ValueError(
                    f"Mask shape {missing.shape} does not match X shape {values.shape}."
                )
            intact = np.asarray(dataset.get("X_intact", values)).copy()
            masked_values = intact.astype(float, copy=True)
            masked_values[missing] = np.nan
            dataset.update(
                X=masked_values,
                X_intact=intact,
                indicating_mask=missing,
            )

        for key in ("X_intact", "indicating_mask"):
            if key in dataset and np.asarray(dataset[key]).shape != values.shape:
                raise ValueError(f"{key} must have the same shape as X.")
        return dataset
