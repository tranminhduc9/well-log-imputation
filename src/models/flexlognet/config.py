"""Configuration for the 256-point FlexLogNet adaptation."""

from dataclasses import dataclass
import math

from src.models.base import ModelConfig


@dataclass(frozen=True)
class FlexLogNetConfig(ModelConfig):
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    resample_factors: tuple[int, ...] = (4, 8, 16)
    kernel_size: int = 7
    stats_hidden: int = 128
    dropout: float = 0.1
    ssim_weight: float = 1.0
    statistics_weight: float = 0.5
    mean_weight: float = 4.0
    min_delta: float = 0.0
    gradient_clip: float | None = 1.0
    epsilon: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "encoder_channels", tuple(self.encoder_channels))
        object.__setattr__(self, "resample_factors", tuple(self.resample_factors))
        if not self.encoder_channels or any(type(c) is not int or c <= 0 for c in self.encoder_channels):
            raise ValueError("encoder_channels must contain positive integers.")
        if self.n_features < 2 or self.seq_len < 2 ** (len(self.encoder_channels) + 1):
            raise ValueError("FlexLogNet needs at least two logs and two points at the deepest level.")
        if type(self.kernel_size) is not int or self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if type(self.stats_hidden) is not int or self.stats_hidden <= 0:
            raise ValueError("stats_hidden must be a positive integer.")
        if len(set(self.resample_factors)) != len(self.resample_factors) or any(
            type(f) is not int or f <= 1 or f > self.seq_len for f in self.resample_factors
        ):
            raise ValueError("resample_factors must be unique integers in [2, seq_len].")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        for name in ("ssim_weight", "statistics_weight", "mean_weight", "min_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be None or finite and positive.")
