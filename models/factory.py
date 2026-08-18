"""Lazy factory for the benchmark's model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .base import AbstractModel, ModelConfig


@dataclass(frozen=True)
class _ModelRegistration:
    module: str
    class_name: str
    uses_optimizer: bool = False


_MODEL_REGISTRY: dict[str, _ModelRegistration] = {
    "locf": _ModelRegistration("locf", "LOCFModel"),
    "rf": _ModelRegistration("random_forest", "RandomForestModel"),
    "qrf": _ModelRegistration("quantile_random_forest", "QuantileRandomForestModel"),
    "quantilerf": _ModelRegistration("quantile_random_forest", "QuantileRandomForestModel"),
    "xgboost": _ModelRegistration("xgboost", "XGBoostModel"),
    "saits": _ModelRegistration("saits", "SAITSModel", uses_optimizer=True),
    "unet": _ModelRegistration("unet", "UNetModel", uses_optimizer=True),
    "bayesnn": _ModelRegistration("bayesian_nn", "BayesianNNModel"),
    "bayessnn": _ModelRegistration("bayesian_nn", "BayesianNNModel"),
    "np": _ModelRegistration("neural_process", "NPModel"),
    "anp_standard": _ModelRegistration("attentive_neural_process", "StandardANPModel"),
    "anp": _ModelRegistration("attention_neural_process", "ANPModel"),
}

MODEL_NAMES = tuple(_MODEL_REGISTRY)


class ModelFactory:
    """Create fresh instances of one configured model for each fold."""

    def __init__(
        self,
        model_name: str,
        seq_len: int = 256,
        n_features: int = 4,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 50,
        optimizer: Any = None,
        device: str = "cpu",
        output_dir: Path | str = ".",
        learning_rate: float = 1e-3,
    ) -> None:
        normalized_name = model_name.lower()
        if normalized_name not in _MODEL_REGISTRY:
            available = ", ".join(MODEL_NAMES)
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}."
            )

        self.model_name = normalized_name
        self.config = ModelConfig(
            seq_len=seq_len,
            n_features=n_features,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            optimizer=optimizer,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir,
        )

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return MODEL_NAMES

    @classmethod
    def uses_optimizer(cls, model_name: str) -> bool:
        try:
            return _MODEL_REGISTRY[model_name.lower()].uses_optimizer
        except KeyError as error:
            raise ValueError(f"Unknown model '{model_name}'.") from error

    def create(self) -> AbstractModel:
        registration = _MODEL_REGISTRY[self.model_name]
        module = import_module(f"{__package__}.{registration.module}")
        model_class = getattr(module, registration.class_name)
        if not issubclass(model_class, AbstractModel):
            raise TypeError(
                f"Registered model {registration.class_name} must implement AbstractModel."
            )
        return model_class(self.config)

    def instantiate(self) -> AbstractModel:
        """Backward-compatible alias for :meth:`create`."""

        return self.create()


def instantiate(model_name: str, **kwargs: Any) -> AbstractModel:
    """Functional shortcut for creating a configured model."""

    return ModelFactory(model_name, **kwargs).create()
