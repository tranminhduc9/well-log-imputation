"""Public model construction API.

Importing :mod:`models` stays lightweight. Concrete implementations are loaded
only when :class:`ModelFactory` creates the selected model.
"""

from .base import AbstractModel, ModelConfig
from .factory import MODEL_NAMES, ModelFactory, instantiate

# Keep the original public name working for notebooks or downstream scripts.
Factory = ModelFactory

__all__ = [
    "AbstractModel",
    "Factory",
    "MODEL_NAMES",
    "ModelConfig",
    "ModelFactory",
    "instantiate",
]
