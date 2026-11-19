"""Per-log XGBoost baseline."""

from .config import XGBoostConfig
from .model import XGBoost

__all__ = ["XGBoost", "XGBoostConfig"]
