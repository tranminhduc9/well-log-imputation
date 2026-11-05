"""Settings for one XGBoost regressor per target log."""

from dataclasses import dataclass
from src.models.base import ModelConfig


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
