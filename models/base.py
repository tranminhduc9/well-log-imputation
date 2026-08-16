"""Common contract implemented by every imputation model adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping


Dataset = Mapping[str, Any]
Prediction = dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    """Runtime settings shared by the concrete model adapters."""

    seq_len: int = 256
    n_features: int = 4
    batch_size: int = 32
    epochs: int = 50
    patience: int = 15
    optimizer: Any = None
    learning_rate: float = 1e-3
    device: str = "cpu"
    output_dir: Path | str = Path(".")

    def __post_init__(self) -> None:
        positive_values = {
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "learning_rate": self.learning_rate,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Model settings must be positive: {', '.join(invalid)}")
        object.__setattr__(self, "output_dir", Path(self.output_dir))


class AbstractModel(ABC):
    """Uniform ``fit``/``predict`` facade for all benchmark models.

    A concrete model only has to implement :meth:`_build_backend` in its own
    module. The facade keeps third-party and repository-native implementations
    behind the same interface used by ``main.py``.
    """

    name: ClassVar[str]
    requires_training: ClassVar[bool] = True

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._backend = self._build_backend()

    @abstractmethod
    def _build_backend(self) -> Any:
        """Create and return the concrete imputation backend."""

    @property
    def backend(self) -> Any:
        """Expose the wrapped backend for advanced use and diagnostics."""

        return self._backend

    def fit(self, train_set: Dataset, val_set: Dataset | None = None) -> None:
        """Train the wrapped model using the benchmark dictionary format."""

        fit = getattr(self._backend, "fit", None)
        if fit is None:
            raise TypeError(f"Backend for model '{self.name}' does not implement fit().")
        fit(train_set=train_set, val_set=val_set)

    def predict(self, test_set: Dataset) -> Prediction:
        """Return an imputation dictionary from the wrapped model."""

        predict = getattr(self._backend, "predict", None)
        if predict is None:
            raise TypeError(f"Backend for model '{self.name}' does not implement predict().")
        prediction = predict(test_set)
        if not isinstance(prediction, dict) or "imputation" not in prediction:
            raise TypeError(
                f"Model '{self.name}' must return a dictionary containing 'imputation'."
            )
        return prediction

    def impute(self, test_set: Dataset) -> Any:
        """Compatibility shortcut returning only the imputed array."""

        return self.predict(test_set)["imputation"]
