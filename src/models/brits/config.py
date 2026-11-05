"""BRITS settings; MIT is optional extra supervision for artificial gaps."""

from dataclasses import dataclass
from src.models.base import ModelConfig


@dataclass(frozen=True)
class BRITSConfig(ModelConfig):
    hidden_size: int = 64
    consistency_weight: float = 0.1
    mit_weight: float = 1.0
    mit_reduction: str = "segment"
    masking_strategy: str = "mixed"
    masking_rate: float = 0.2
    gradient_clip: float = 1.0
    validation_metric: str = "rmse"
    min_delta: float = 1e-4
    weight_decay: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.mit_reduction not in {"segment", "point"}:
            raise ValueError("mit_reduction must be 'segment' or 'point'.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative.")
        if self.mit_weight < 0:
            raise ValueError("mit_weight must be non-negative.")
        if self.masking_strategy not in {"mixed", "random", "none"}:
            raise ValueError(
                "masking_strategy must be 'mixed', 'random', or 'none'."
            )
        if not 0 < self.masking_rate < 1:
            raise ValueError("masking_rate must be in (0, 1).")
        if self.gradient_clip < 0:
            raise ValueError("gradient_clip must be non-negative.")
        if self.validation_metric not in {"mse", "rmse"}:
            raise ValueError("validation_metric must be 'mse' or 'rmse'.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
