"""Per-log XGBoost baseline."""

from .config import XGBoostConfig
from .model import XGBoost, _XGBoostBackend

__all__ = ["XGBoost", "XGBoostConfig"]
