"""SAITS adapter exposing the common fit/predict/impute interface."""

from src.models.base import AbstractModel
from .config import SAITSConfig
from .training import _SAITSBackend


class SAITS(AbstractModel):
    """Self-attention imputation for segmented well logs."""

    name = "saits"

    def __init__(self, config: SAITSConfig | None = None) -> None:
        super().__init__(config or SAITSConfig())

    def _build_backend(self) -> _SAITSBackend:
        return _SAITSBackend(self.config)
