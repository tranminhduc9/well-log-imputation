"""FlexLogNet adapter for the common imputation interface."""

from src.models.base import AbstractModel
from .config import FlexLogNetConfig
from .training import _FlexLogNetBackend


class FlexLogNet(AbstractModel):
    name = "flexlognet"

    def __init__(self, config: FlexLogNetConfig | None = None):
        super().__init__(config or FlexLogNetConfig())

    def _build_backend(self):
        return _FlexLogNetBackend(self.config)
