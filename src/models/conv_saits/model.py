"""Conv-SAITS adapter exposing the common fit/predict/impute interface."""

from src.models.base import AbstractModel
from .config import ConvSAITSConfig
from .conv_encoder import DecoupledFeatureEncoder
from .network import ConvSAITSNetwork, DiagonallyMaskedAttentionLayer
from .training import _ConvSAITSBackend


class ConvSAITS(AbstractModel):
    """Dual-stage convolution-and-attention imputer for segmented well logs."""

    name = "conv_saits"

    def __init__(self, config: ConvSAITSConfig | None = None) -> None:
        super().__init__(config or ConvSAITSConfig())

    def _build_backend(self) -> _ConvSAITSBackend:
        return _ConvSAITSBackend(self.config)


CONV_SAITS = ConvSAITS
