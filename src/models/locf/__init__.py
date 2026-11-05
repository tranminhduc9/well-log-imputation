"""Training-free last-observation baseline."""

from .model import LOCF, _LOCFBackend

__all__ = ["LOCF"]
