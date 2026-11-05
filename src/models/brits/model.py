"""BRITS adapter exposing the common fit/predict/impute interface."""

from src.models.base import AbstractModel
from .config import BRITSConfig
from .training import _BRITSBackend


class BRITS(AbstractModel):
    """Bidirectional recurrent imputation for segmented well logs."""

    name = "brits"

    def __init__(self, config: BRITSConfig | None = None):
        super().__init__(config or BRITSConfig())

    def _build_backend(self):
        return _BRITSBackend(self.config)
