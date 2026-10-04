"""Conv-SAITS architecture, configuration and optional research variants."""

from .config import ConvSAITSConfig
from .model import (
    DecoupledFeatureEncoder, DiagonallyMaskedAttentionLayer,
    ConvSAITS, ConvSAITSNetwork, CONV_SAITS, _ConvSAITSBackend,
)
from .ablation import CrossLogAttention, ConditionalLogDecoder, gap_features

__all__ = [
    "ConvSAITS", "CONV_SAITS", "ConvSAITSConfig", "ConvSAITSNetwork",
    "DecoupledFeatureEncoder", "DiagonallyMaskedAttentionLayer",
    "CrossLogAttention", "ConditionalLogDecoder", "gap_features",
]
